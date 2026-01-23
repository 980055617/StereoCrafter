#!/usr/bin/env python3
# =============================================
# File: /workspace/stereocraft/scripts/count_pipeline_parameters.py
# ---------------------------------------------
# 目的: パイプラインのパラメータ数集計
# =============================================

"""Compare parameter counts between base and Mamba pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from fire import Fire
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

from pipelines.mamba_stereo_video_inpainting_pipeline import (
    MambaStableVideoDiffusionInpaintingPipeline,
)
from pipelines.stereo_video_inpainting import (
    StableVideoDiffusionInpaintingPipeline,
)


PipelineCounts = Dict[str, int]


def _resolve_dtype(precision: str) -> torch.dtype:
    prec = (precision or "fp16").lower()
    if prec == "fp16":
        return torch.float16
    if prec == "bf16":
        return torch.bfloat16
    if prec == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision '{precision}'. Pick from ['fp16', 'bf16', 'fp32'].")


def _load_components(
    pre_trained_path: str,
    unet_path: str,
    torch_dtype: torch.dtype,
) -> Tuple[CLIPVisionModelWithProjection, AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel]:
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch_dtype,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch_dtype,
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    return image_encoder, vae, unet


def _count_parameters(pipeline) -> PipelineCounts:
    counts: PipelineCounts = {}
    total = 0
    # `components` contains only registered modules; we skip items without parameters.
    for name, module in pipeline.components.items():
        if not hasattr(module, "parameters"):
            counts[name] = 0
            continue
        param_count = sum(param.numel() for param in module.parameters())
        counts[name] = param_count
        total += param_count

    counts["total"] = total
    return counts


@dataclass
class InspectResult:
    label: str
    counts: PipelineCounts

    def pretty(self) -> str:
        lines = [f"{self.label}"]
        total = self.counts.get("total", 0)
        total_m = total / 1_000_000 if total else 0.0
        lines.append(f"  total parameters: {total:,} ({total_m:.2f}M)")
        for name, value in sorted(self.counts.items()):
            if name == "total":
                continue
            value_m = value / 1_000_000 if value else 0.0
            lines.append(f"  - {name}: {value:,} ({value_m:.2f}M)")
        return "\n".join(lines)


def inspect(
    pre_trained_path: str,
    unet_path: str,
    *,
    precision: str = "fp16",
) -> None:
    """
    Print parameter counts for both pipeline variants.

    Args:
        pre_trained_path: Base Stable Video Diffusion checkpoint directory.
        unet_path: UNet checkpoint directory (fine-tuned weights live under `unet_diffusers`).
        precision: One of ['fp16', 'bf16', 'fp32']; controls torch dtype used when loading modules.
    """
    torch_dtype = _resolve_dtype(precision)

    results = []
    for _, label, pipeline_cls in (
        (False, "Pipeline (use_mamba = False)", StableVideoDiffusionInpaintingPipeline),
        (True, "Pipeline (use_mamba = True)", MambaStableVideoDiffusionInpaintingPipeline),
    ):
        image_encoder, vae, unet = _load_components(pre_trained_path, unet_path, torch_dtype)
        pipeline = pipeline_cls.from_pretrained(
            pre_trained_path,
            image_encoder=image_encoder,
            vae=vae,
            unet=unet,
            torch_dtype=torch_dtype,
        )
        counts = _count_parameters(pipeline)
        results.append(InspectResult(label=label, counts=counts))
        # Free up memory before constructing the next pipeline variant
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    for result in results:
        print(result.pretty())
        print()

    diff = results[1].counts["total"] - results[0].counts["total"]
    diff_m = diff / 1_000_000
    print(f"Δ total parameters (use_mamba=True - use_mamba=False): {diff:+,} ({diff_m:+.2f}M)")


def main():
    Fire(inspect)


if __name__ == "__main__":
    main()
