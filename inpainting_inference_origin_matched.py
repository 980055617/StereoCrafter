"""Matched-crop, guidance-overridable runner for the origin (non-Mamba)
reference pipeline.

Does not edit any `origin`-named file. Re-implements the orchestration from
`inpainting_inference_origin.main()` (same `spatial_tiled_process` pipeline,
imported, not duplicated) but:
- center-crops inputs to `target_height`/`target_width` before generation, so
  the output is directly comparable to the Mamba matched-crop presets and to
  `scripts/evaluate_inpainting_train_tile.py` (same crop convention as
  `inpainting_inference.py`'s `target_height`/`target_width` handling).
- exposes `min_guidance_scale`/`max_guidance_scale` as CLI overrides instead
  of the hardcoded 1.01/1.01 in `inpainting_inference_origin.py`.

Used to check whether a quality failure mode observed on Mamba-replaced
checkpoints (e.g. over-saturation/painted-over structure at higher guidance)
also appears on the origin reference, to tell apart "Mamba-specific fragility"
from "shared pipeline/recipe fragility".
"""

from __future__ import annotations

import os

import numpy as np
import torch
from decord import VideoReader, cpu
from fire import Fire
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from inpainting_inference_origin import spatial_tiled_process, write_video_opencv
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid


def _center_crop_chw(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    src_h, src_w = int(frames.shape[2]), int(frames.shape[3])
    if height > src_h or width > src_w:
        raise ValueError(f"Crop {height}x{width} exceeds source {src_h}x{src_w}")
    top = (src_h - height) // 2
    left = (src_w - width) // 2
    return frames[:, :, top : top + height, left : left + width]


def main(
    pre_trained_path: str = "weights/stable-video-diffusion-img2vid-xt-1-1/",
    unet_path: str = "weights/StereoCrafter/",
    input_video_path: str = "video_data/splatting/0160_splatting_results.mp4",
    save_dir: str = "outputs/diagnose_0160/origin_matched",
    frames_chunk: int = 14,
    overlap: int = 3,
    tile_num: int = 1,
    target_height: int = 576,
    target_width: int = 1024,
    min_guidance_scale: float = 1.01,
    max_guidance_scale: float = 1.01,
    num_inference_steps: int = 8,
) -> None:
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path, subfolder="image_encoder", variant="fp16", torch_dtype=torch.float16
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, subfolder="vae", variant="fp16", torch_dtype=torch.float16
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=torch.float16
    )
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path, image_encoder=image_encoder, vae=vae, unet=unet, torch_dtype=torch.float16
    )
    pipeline = pipeline.to("cuda")

    os.makedirs(save_dir, exist_ok=True)
    video_name = (
        input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "")
        + "_inpainting_results"
    )

    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    frame_indices = list(range(len(video_reader)))
    frames = video_reader.get_batch(frame_indices)
    num_frames = len(video_reader)

    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()
    height, width = frames.shape[2] // 2, frames.shape[3] // 2
    frames_left = frames[:, :, :height, :width]
    frames_mask = frames[:, :, height:, :width]
    frames_warpped = frames[:, :, height:, width:]

    frames_left = frames_left / 255.0
    frames_mask = (frames_mask / 255.0).mean(dim=1, keepdim=True)
    frames_warpped = frames_warpped / 255.0

    frames_left = _center_crop_chw(frames_left, target_height, target_width)
    frames_mask = _center_crop_chw(frames_mask, target_height, target_width)
    frames_warpped = _center_crop_chw(frames_warpped, target_height, target_width)

    results = []
    generated = None
    for i in range(0, num_frames, frames_chunk - overlap):
        if i + overlap >= frames_warpped.shape[0]:
            break
        if generated is not None and i + frames_chunk > frames_warpped.shape[0]:
            cur_i = max(frames_warpped.shape[0] + overlap - frames_chunk, 0)
            cur_overlap = i - cur_i + overlap
        else:
            cur_i = i
            cur_overlap = overlap

        input_frames_i = frames_warpped[cur_i : cur_i + frames_chunk].clone()
        mask_frames_i = frames_mask[cur_i : cur_i + frames_chunk]

        if generated is not None:
            input_frames_i[:cur_overlap] = generated[-cur_overlap:]

        video_latents = spatial_tiled_process(
            input_frames_i,
            mask_frames_i,
            pipeline,
            tile_num,
            spatial_n_compress=8,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            decode_chunk_size=8,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=num_inference_steps,
        )

        video_latents = video_latents.unsqueeze(0)
        if video_latents == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        video_frames = pipeline.decode_latents(video_latents, num_frames=video_latents.shape[1], decode_chunk_size=2)
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
        generated = torch.stack(video_frames)
        if i != 0:
            generated = generated[cur_overlap:]
        results.append(generated)

    frames_output = torch.cat(results, dim=0).cpu()
    frames_sbs = torch.cat([frames_left, frames_output], dim=3)
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    frames_sbs_np = (frames_sbs * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    write_video_opencv(frames_sbs_np, fps, frames_sbs_path)
    print(f"Wrote {frames_sbs_path}")


if __name__ == "__main__":
    Fire(main)
