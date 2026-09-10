#!/usr/bin/env python3
"""Run the reference StereoCrafter inpainting pipeline with a matched crop.

This is a comparison adapter, not a reference baseline file. It leaves the
`origin` scripts untouched while allowing the reference pipeline to be profiled
at the same per-eye crop as the current Mamba 0160 inference config.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from decord import VideoReader, cpu
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from fire import Fire
from transformers import CLIPVisionModelWithProjection

from inpainting_inference_origin import spatial_tiled_process, write_video_opencv
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
from utils.module_timing import install_cuda_module_timer, write_cuda_module_timing


def center_crop_frames(frames: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    if frames.dim() != 4:
        raise ValueError(f"expected [T,C,H,W], got shape={tuple(frames.shape)}")
    h = int(frames.shape[2])
    w = int(frames.shape[3])
    if crop_h > h or crop_w > w:
        raise ValueError(f"crop {crop_h}x{crop_w} exceeds source {h}x{w}")
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return frames[:, :, top : top + crop_h, left : left + crop_w]


def main(
    pre_trained_path: str = "weights/stable-video-diffusion-img2vid-xt-1-1/",
    unet_path: str = "weights/StereoCrafter/",
    input_video_path: str = "video_data/splatting/0160_splatting_results.mp4",
    save_dir: str = "outputs/diagnose_0160/reference_matched_inpainting",
    frames_chunk: int = 14,
    overlap: int = 3,
    tile_num: int = 1,
    target_height: int = 576,
    target_width: int = 1024,
    max_profile_chunks: int | None = None,
    module_profile_json: str | None = None,
    module_profile_include: str | None = None,
) -> None:
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch.float16,
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
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
    ).to("cuda")
    module_profile_state = None
    module_profile_handles = []
    if module_profile_json:
        module_profile_state, module_profile_handles = install_cuda_module_timer(
            pipeline.unet,
            include=module_profile_include,
            default_include=["*.attn1"],
        )
        print(
            "[profile] module timing enabled: "
            f"modules={len(module_profile_state['modules'])} output={module_profile_json}"
        )

    os.makedirs(save_dir, exist_ok=True)
    video_name = (
        os.path.basename(input_video_path)
        .replace(".mp4", "")
        .replace("_splatting_results", "")
        + "_inpainting_results"
    )

    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    frame_indices = list(range(len(video_reader)))
    frames = video_reader.get_batch(frame_indices)
    num_frames = len(video_reader)

    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()
    height = frames.shape[2] // 2
    width = frames.shape[3] // 2
    frames_left = center_crop_frames(frames[:, :, :height, :width], int(target_height), int(target_width))
    frames_mask = center_crop_frames(frames[:, :, height:, :width], int(target_height), int(target_width))
    frames_warped = center_crop_frames(frames[:, :, height:, width:], int(target_height), int(target_width))

    frames = torch.cat([frames_warped, frames_left, frames_mask], dim=0) / 255.0
    frames_warped, frames_left, frames_mask = torch.chunk(frames, chunks=3, dim=0)
    frames_mask = frames_mask.mean(dim=1, keepdim=True)

    results = []
    generated = None
    processed_chunks = 0
    for i in range(0, num_frames, frames_chunk - overlap):
        if i + overlap >= frames_warped.shape[0]:
            break

        if generated is not None and i + frames_chunk > frames_warped.shape[0]:
            cur_i = max(frames_warped.shape[0] + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        input_frames_i = frames_warped[cur_i : cur_i + frames_chunk].clone()
        mask_frames_i = frames_mask[cur_i : cur_i + frames_chunk]

        if generated is not None:
            input_frames_i[:cur_overlap] = generated[-cur_overlap:]

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

        video_latents = video_latents.unsqueeze(0).to(pipeline.vae.dtype)
        video_frames = pipeline.decode_latents(
            video_latents,
            num_frames=video_latents.shape[1],
            decode_chunk_size=2,
        )
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j, img in enumerate(video_frames):
            video_frames[j] = torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        generated = torch.stack(video_frames)
        if i != 0:
            generated = generated[cur_overlap:]
        results.append(generated)
        processed_chunks += 1
        if max_profile_chunks is not None and processed_chunks >= int(max_profile_chunks):
            print(f"[profile] stopping after max_profile_chunks={max_profile_chunks}")
            break

    frames_output = torch.cat(results, dim=0).cpu()
    frames_left_output = frames_left[: frames_output.shape[0]]

    frames_sbs = torch.cat([frames_left_output, frames_output], dim=3)
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    frames_sbs = (frames_sbs * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    write_video_opencv(frames_sbs, fps, frames_sbs_path)

    vid_left = (frames_left_output * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    vid_right = (frames_output * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    vid_left[:, :, :, 1] = 0
    vid_left[:, :, :, 2] = 0
    vid_right[:, :, :, 0] = 0

    vid_anaglyph = vid_left + vid_right
    vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    write_video_opencv(vid_anaglyph, fps, vid_anaglyph_path)

    if module_profile_state is not None and module_profile_json:
        payload = write_cuda_module_timing(
            module_profile_state,
            module_profile_json,
            metadata={
                "saveDir": save_dir,
                "maxProfileChunks": max_profile_chunks,
                "processedChunks": processed_chunks,
                "framesChunk": frames_chunk,
            },
        )
        for handle in module_profile_handles:
            handle.remove()
        print(
            "[profile] module timing wrote "
            f"{module_profile_json} totalProfiledMs={payload['totalProfiledMs']:.3f}"
        )


if __name__ == "__main__":
    Fire(main)
