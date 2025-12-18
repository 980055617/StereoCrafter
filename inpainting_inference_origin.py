import os
from pathlib import Path
import cv2
import numpy as np
from fire import Fire

import torch
from decord import VideoReader, cpu

from transformers import CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKLTemporalDecoder,
)
from diffusers import UNetSpatioTemporalConditionModel

from utils.training_batches import chunk_frame_ranges
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :overlap_size, :]
    return b


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=8,
    **kargs,
):
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    tile_size = (
        int((height + tile_overlap[0] *  (tile_num - 1)) / tile_num), 
        int((width  + tile_overlap[1] * (tile_num - 1)) / tile_num)
    )
    tile_stride = (
        (tile_size[0] - tile_overlap[0]), 
        (tile_size[1] - tile_overlap[1])
        )
    
    cols = []
    for i in range(0, tile_num):
        rows = []
        for j in range(0, tile_num):

            cond_tile = cond_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]
            mask_tile = mask_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]

            tile = process_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs,
            ).frames[0]

            rows.append(tile)
        cols.append(rows)

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress,
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress,
    )

    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    x = torch.cat(pixels, dim=2)
    return x


def write_video_opencv(input_frames, fps, output_video_path):

    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    out = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, 
        (width, height)
    )

    for i in range(num_frames):
        out.write(input_frames[i, :, :, ::-1])

    out.release()



def main(
    pre_trained_path,
    unet_path,
    input_video_path,
    save_dir,
    frames_chunk=23,
    overlap=3,
    tile_num=1,
    decode_chunk_size: int = 2,
):
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, 
        subfolder="vae", 
        variant="fp16", 
        torch_dtype=torch.float16
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        # variant="fp16",
        torch_dtype=torch.float16
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to("cuda")

    os.makedirs(save_dir, exist_ok=True)
    video_name = input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"

    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    num_frames = len(video_reader)
    if num_frames == 0:
        raise ValueError(f"No frames found in video: {input_video_path}")

    first = video_reader[0].asnumpy()
    height_raw, width_raw = first.shape[0] // 2, first.shape[1] // 2
    height = height_raw // 128 * 128
    width = width_raw // 128 * 128

    def _load_chunk(start: int, end: int):
        indices = list(range(start, end))
        batch = video_reader.get_batch(indices).asnumpy()  # [T,H,W,C]
        frames = torch.tensor(batch).permute(0, 3, 1, 2).float()  # [T,C,H,W]
        left = frames[:, :, :height_raw, :width_raw]
        mask = frames[:, :, height_raw:, :width_raw]
        warped = frames[:, :, height_raw:, width_raw:]

        left = left[:, :, :height, :width] / 255.0
        warped = warped[:, :, :height, :width] / 255.0
        mask = mask[:, :, :height, :width] / 255.0
        mask = mask.mean(dim=1, keepdim=True)
        return left, warped, mask

    step = max(frames_chunk - overlap, 1)
    frame_ranges = list(chunk_frame_ranges(num_frames, frames_chunk, overlap)) if frames_chunk > 0 else [(0, num_frames)]

    generated_prev = None
    stem = Path(input_video_path).stem
    # "_train" が末尾についていれば落とし、数字部分のみの名前にする
    video_name = stem[:-6] if stem.endswith("_train") else stem

    # SBS をストリーミングで書き出す
    width_sbs = width * 2
    writer = cv2.VideoWriter(
        os.path.join(save_dir, f"{video_name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width_sbs, height),
    )

    for start, end in frame_ranges:
        if end - start <= 0:
            continue
        frames_left, frames_warped, frames_mask = _load_chunk(start, end)

        input_frames_i = frames_warped.clone()
        mask_frames_i = frames_mask

        if generated_prev is not None and overlap > 0 and start > 0:
            ov = min(overlap, generated_prev.shape[0], input_frames_i.shape[0])
            if ov > 0:
                input_frames_i[:ov] = generated_prev[-ov:]

        video_latents = spatial_tiled_process(
            input_frames_i,
            mask_frames_i,
            pipeline,
            tile_num,
            spatial_n_compress=8,
            min_guidance_scale=1.01,
            max_guidance_scale=1.01,
            decode_chunk_size=8,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=8,
        )

        video_latents = video_latents.unsqueeze(0)
        if video_latents == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        video_frames = pipeline.decode_latents(
            video_latents,
            num_frames=video_latents.shape[1],
            decode_chunk_size=decode_chunk_size,
        )
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32)
                / 255.0
            )
        generated = torch.stack(video_frames)

        append_gen = generated if start == 0 else generated[overlap:]
        append_left = frames_left if start == 0 else frames_left[overlap:]

        generated_prev = generated

        frames_sbs = torch.cat([append_left, append_gen], dim=3)
        frames_sbs_np = (
            (frames_sbs * 255)
            .permute(0, 2, 3, 1)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )
        for frame in frames_sbs_np:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR

    writer.release()


if __name__ == "__main__":
    Fire(main)
