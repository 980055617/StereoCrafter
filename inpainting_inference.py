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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from pipelines.mamba_stereo_video_inpainting_pipeline import (
    MambaStableVideoDiffusionInpaintingPipeline as _Pipe,
    tensor2vid,
)
from utils.inpainting import read_and_prepare_video, spatial_tiled_process, write_video_opencv
from utils.config_utils import load_json_config
from utils.model_io import resolve_unet_state_path
from utils.training_pipeline import enable_vae_memory_helpers

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch.library.impl_abstract.*register_fake.*",
)


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
    use_mamba: bool = False,
    unet_state_path: str | None = None,
    noise_seed: int | None = None,
):
    prec = (precision or "fp16").lower()
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
    # Align inference scheduler with training (DDPM-based forward process).
    if getattr(pipeline, "scheduler", None) is not None:
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

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
                input_frames_i[:cur_overlap] = generated[-cur_overlap:]
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
            min_guidance_scale=1.01,
            max_guidance_scale=1.01,
            decode_chunk_size=8,
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
