import inspect
import os
import warnings
from typing import Any

import numpy as np
import torch
from fire import Fire
from transformers import CLIPVisionModelWithProjection
from diffusers.models.unets.unet_spatio_temporal_condition import (
    UNetSpatioTemporalConditionModel,
)
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import (
    AutoencoderKLTemporalDecoder,
)

from pipelines.mamba_stereo_video_inpainting_pipeline import (
    MambaStableVideoDiffusionInpaintingPipeline as _Pipe,
    tensor2vid,
)
from utils.inpainting import read_and_prepare_video, spatial_tiled_process, write_video_opencv
from utils.config_utils import load_json_config
from utils.model_io import resolve_unet_state_path
from utils.training_pipeline import enable_vae_memory_helpers
from utils.diffusers_mamba_time_patch import apply_mamba_time_patch
from blocks.mamba_diffusers_adapter import materialize_mamba_time_embed_proj_from_state_dict

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch.library.impl_abstract.*register_fake.*",
)

apply_mamba_time_patch()


def _scheduler_debug_info(scheduler: Any) -> dict[str, Any]:
    cfg = getattr(scheduler, "config", None)
    return {
        "class": scheduler.__class__.__name__,
        "prediction_type": getattr(cfg, "prediction_type", None),
        "num_train_timesteps": getattr(cfg, "num_train_timesteps", None),
        "beta_schedule": getattr(cfg, "beta_schedule", None),
        "rescale_betas_zero_snr": getattr(cfg, "rescale_betas_zero_snr", None),
        "steps_offset": getattr(cfg, "steps_offset", None),
    }


def _center_crop_frames(frames: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    """Center-crop [T,C,H,W] frames to match training fixed-crop behavior."""
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("target_height/target_width must be positive.")
    if frames.dim() != 4:
        raise ValueError(f"Expected 4D tensor [T,C,H,W], got shape={tuple(frames.shape)}")
    h = int(frames.shape[2])
    w = int(frames.shape[3])
    if crop_h > h or crop_w > w:
        raise ValueError(
            f"Requested crop {crop_h}x{crop_w} exceeds source {h}x{w}. "
            "Use a smaller target size."
        )
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    return frames[:, :, top : top + crop_h, left : left + crop_w]


def main(
    pre_trained_path: str,
    unet_path: str,
    input_video_path: str,
    save_dir: str,
    frames_chunk: int = 11,
    overlap: int = 3,
    tile_num: int = 1,
    num_inference_steps: int = 8,
    *,
    precision: str = "fp16",
    decode_chunk_size: int = 2,
    unet_state_path: str | None = None,
    noise_seed: int | None = None,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 1.0,
    target_height: int | None = None,
    target_width: int | None = None,
    overlap_prev_weight: float = 1.0,
):
    prec = (precision or "fp16").lower()
    overlap_prev_weight = float(overlap_prev_weight)
    torch_dtype = torch.float16 if prec == "fp16" else (torch.bfloat16 if prec == "bf16" else torch.float32)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch_dtype
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, 
        subfolder="vae", 
        variant="fp16", 
        torch_dtype=torch_dtype
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        # variant="fp16",
        torch_dtype=torch_dtype
    )
    expected_unet_dir = os.path.join(unet_path, "unet_diffusers")
    print(
        f"Loaded UNet from {unet_path} (subfolder='unet_diffusers', exists={os.path.isdir(expected_unet_dir)})"
    )
    print(
        "UNet config: in_channels=%s, out_channels=%s, num_frames=%s, cross_attention_dim=%s"
        % (
            getattr(unet.config, "in_channels", None),
            getattr(unet.config, "out_channels", None),
            getattr(unet.config, "num_frames", None),
            getattr(unet.config, "cross_attention_dim", None),
        )
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = _Pipe.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch_dtype,
    )
    enable_vae_memory_helpers(pipeline)
    print(f"[sched][inference] {_scheduler_debug_info(pipeline.scheduler)}")
    try:
        pipeline.scheduler.set_timesteps(int(num_inference_steps), device="cpu")
        ts = pipeline.scheduler.timesteps
        ts_head = ts[: min(8, len(ts))].detach().cpu().tolist() if isinstance(ts, torch.Tensor) else list(ts[:8])
        print(f"[sched][inference][timesteps_head] {ts_head}")
    except Exception as err:
        print(f"[sched][inference][timesteps_head] unavailable: {err}")

    # Optionally load a fine‑tuned UNet state_dict (.pt) produced by training
    unet_state_path = resolve_unet_state_path(unet_state_path)
    if unet_state_path is not None and os.path.isfile(unet_state_path):
        # load unet weights safely and cast back to desired dtype
        try:
            raw_state = torch.load(unet_state_path, map_location="cpu", weights_only=True)  # torch>=2.4
        except TypeError:
            raw_state = torch.load(unet_state_path, map_location="cpu")

        # unwrap training checkpoints that contain optimizer/scheduler, etc.
        state_dict = None
        if isinstance(raw_state, dict):
            for key in ("model", "unet", "state_dict"):
                if key in raw_state and isinstance(raw_state[key], dict):
                    state_dict = raw_state[key]
                    break
            # if it already looks like a state_dict, use it directly
            if state_dict is None and all(isinstance(v, torch.Tensor) for v in raw_state.values()):
                state_dict = raw_state
        if state_dict is None:
            state_dict = raw_state

        # Materialize lazy Mamba time projection modules so their keys are loadable.
        created = materialize_mamba_time_embed_proj_from_state_dict(pipeline.unet, state_dict)
        if created > 0:
            print(f"Materialized Mamba time_embed_proj modules before load: {created}")

        # safer to load on fp32 then cast down if needed
        try:
            pipeline.unet.to(dtype=torch.float32)
        except Exception:
            pass
        missing, unexpected = pipeline.unet.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] Missing keys when loading UNet: {len(missing)} (showing first 5): {missing[:5]}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading UNet: {len(unexpected)} (showing first 5): {unexpected[:5]}")
        # cast back to requested precision to avoid dtype mismatch during matmuls
        try:
            pipeline.unet.to(dtype=torch_dtype)
        except Exception:
            pass

    if hasattr(pipeline, "vae"):
        target_dtype = torch.float16 if prec == "fp16" else (torch.bfloat16 if prec == "bf16" else torch.float32)
        pipeline.vae.to(dtype=target_dtype)

    pipeline = pipeline.to("cuda")
    generator = None
    if noise_seed is not None:
        seed_val = int(noise_seed)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)

    os.makedirs(save_dir, exist_ok=True)
    video_name = input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"

    fps, frames_left, frames_warped, frames_mask = read_and_prepare_video(input_video_path)
    if target_height is not None and target_width is not None:
        h = int(target_height)
        w = int(target_width)
        if h > 0 and w > 0:
            # Match training fixed crop: center crop instead of resize.
            frames_left = _center_crop_frames(frames_left, h, w)
            frames_warped = _center_crop_frames(frames_warped, h, w)
            frames_mask = _center_crop_frames(frames_mask, h, w)
    num_frames = frames_warped.shape[0]

    results = []
    generated = None
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

            try:
                prev = generated[-cur_overlap:]
                if overlap_prev_weight >= 1.0:
                    input_frames_i[:cur_overlap] = prev
                elif overlap_prev_weight <= 0.0:
                    pass
                else:
                    alpha = float(overlap_prev_weight)
                    input_frames_i[:cur_overlap] = alpha * prev + (1.0 - alpha) * input_frames_i[:cur_overlap]
            except Exception as e:
                print(e)
                print(
                    f"i: {i}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, input_frames_i: {input_frames_i.shape}, generated: {generated.shape}"
                )

        video_latents = spatial_tiled_process(
            input_frames_i,
            mask_frames_i,
            pipeline,
            tile_num,
            spatial_n_compress=8,
            min_guidance_scale=float(min_guidance_scale),
            max_guidance_scale=float(max_guidance_scale),
            decode_chunk_size=decode_chunk_size,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        video_latents = video_latents.unsqueeze(0)

        # VAEのデータ型とlatentsのデータ型を合わせる
        video_latents = video_latents.to(pipeline.vae.dtype)

        video_frames = pipeline.decode_latents(video_latents, num_frames=video_latents.shape[1], decode_chunk_size=decode_chunk_size)
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32)
                / 255.0
            )
        generated = torch.stack(video_frames)
        if i != 0:
            generated = generated[cur_overlap:]
        results.append(generated)

    frames_output = torch.cat(results, dim=0).cpu()


    frames_sbs = torch.cat([frames_left, frames_output], dim=3)
    frames_sbs_path = os.path.join(save_dir, f"{video_name}_sbs.mp4")
    frames_sbs = (frames_sbs * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    write_video_opencv(frames_sbs, fps, frames_sbs_path)


    vid_left = (frames_left * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()
    vid_right = (frames_output * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()

    vid_left[:, :, :, 1] = 0
    vid_left[:, :, :, 2] = 0
    vid_right[:, :, :, 0] = 0

    vid_anaglyph = vid_left + vid_right
    vid_anaglyph_path = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
    write_video_opencv(vid_anaglyph, fps, vid_anaglyph_path)


def run(config: str | None = None, config_dir: str = "config", **overrides: Any) -> None:
    """Fire entry point with optional JSON config loading (similar to training)."""
    config_identifier = config or os.environ.get("STEREOCRAFT_INFERENCE_CONFIG")
    config_values: dict[str, Any] = {}
    if config_identifier:
        config_path, payload = load_json_config(config_identifier, config_dir)
        config_values.update(payload)
        print(f"Loaded inference config from {config_path}")
    config_values.update(overrides)

    signature = inspect.signature(main)
    required = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect._empty
    ]
    missing = [name for name in required if name not in config_values]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required inference parameters: {missing_list}")

    unexpected_keys = set(config_values) - set(signature.parameters.keys())
    if unexpected_keys:
        unexpected_list = ", ".join(sorted(unexpected_keys))
        raise ValueError(f"Unknown inference parameters: {unexpected_list}")

    main(**config_values)


if __name__ == "__main__":
    Fire(run)
