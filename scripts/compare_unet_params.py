#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================
# File: /workspace/stereocraft/scripts/compare_unet_params.py
# ---------------------------------------------
# 目的: UNetパラメータ数の比較
# =============================================

"""Compare parameter counts between base and Mamba-adapted UNet."""
import argparse
import os
import json
import torch
from typing import Tuple

from diffusers import UNetSpatioTemporalConditionModel

from blocks.mamba_diffusers_adapter import (
    replace_unet_spatiotemporal_transformer_with_mamba,
)


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_unet_base(base_dir: str) -> UNetSpatioTemporalConditionModel:
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        base_dir, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    return unet


def load_unet_adapted(adapted_dir: str, base_dir_for_config: str = None) -> UNetSpatioTemporalConditionModel:
    # 1) Load config (prefer adapted_dir/config.json; fallback to base_dir/unet)
    cfg_path = os.path.join(adapted_dir, "config.json")
    if os.path.exists(cfg_path):
        unet = UNetSpatioTemporalConditionModel.from_config(json.load(open(cfg_path, "r")))
    else:
        if base_dir_for_config is None:
            raise FileNotFoundError("config.json not found in adapted_dir; please pass a base_dir with --base_dir_for_config")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            base_dir_for_config, subfolder="unet", low_cpu_mem_usage=True, torch_dtype=torch.float32
        )

    # 2) Read adapter config if present
    adapter_cfg = {
        "mamba_d_state": 128,
        "mamba_expand": 1,
        "temporal_chunk": 32,
        "spatial_chunk": 2048,
        "mem_eff": True,
        "keep_spatial_mixer": True,
    }
    ac_path = os.path.join(adapted_dir, "mamba_adapter.json")
    if os.path.exists(ac_path):
        try:
            with open(ac_path, "r") as f:
                ac = json.load(f)
            adapter_cfg.update(
                {
                    "mamba_d_state": int(ac.get("mamba_d_state", adapter_cfg["mamba_d_state"])),
                    "mamba_expand": int(ac.get("mamba_expand", adapter_cfg["mamba_expand"])),
                    "temporal_chunk": int(ac.get("temporal_chunk", adapter_cfg["temporal_chunk"])),
                    "spatial_chunk": int(ac.get("spatial_chunk", adapter_cfg["spatial_chunk"])),
                    "mem_eff": bool(ac.get("mem_eff", adapter_cfg["mem_eff"])),
                }
            )
        except Exception:
            pass

    # 3) Apply adapter
    replace_unet_spatiotemporal_transformer_with_mamba(
        unet,
        use_mem_eff_path=adapter_cfg["mem_eff"] and torch.cuda.is_available(),
        temporal_chunk_size=adapter_cfg["temporal_chunk"],
        spatial_chunk_size=adapter_cfg["spatial_chunk"],
        d_state=adapter_cfg["mamba_d_state"],
        expand=adapter_cfg["mamba_expand"],
        keep_spatial_mixer=adapter_cfg["keep_spatial_mixer"],
    )

    # 4) Load adapted state_dict (safetensors/bin)
    cand = [
        os.path.join(adapted_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(adapted_dir, "model.safetensors"),
        os.path.join(adapted_dir, "diffusion_pytorch_model.bin"),
        os.path.join(adapted_dir, "pytorch_model.bin"),
    ]
    sd_path_st = next((p for p in cand if p.endswith(".safetensors") and os.path.exists(p)), None)
    sd_path_pt = next((p for p in cand if p.endswith(".bin") and os.path.exists(p)), None)
    if sd_path_st is not None:
        from safetensors.torch import load_file as load_sft
        sd = load_sft(sd_path_st, device="cpu")
    elif sd_path_pt is not None:
        sd = torch.load(sd_path_pt, map_location="cpu")
    else:
        raise FileNotFoundError(f"No state dict found in {adapted_dir}")
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    if missing:
        print(f"[adapted] missing keys: {len(missing)} (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"[adapted] unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}")
    return unet


def sizeof_mb(n_params: int, bytes_per_param: int = 4) -> float:
    return n_params * bytes_per_param / (1024**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="Path to base weights dir (contains subfolder 'unet')")
    ap.add_argument("--adapted_dir", type=str, required=True, help="Path to adapted UNet dir (unet_diffusers)")
    ap.add_argument("--base_dir_for_config", type=str, default=None, help="Optional fallback for config if adapted has none")
    args = ap.parse_args()

    # Base
    base_unet = load_unet_base(args.base_dir)
    base_total, base_trainable = count_params(base_unet)

    # Adapted (Mamba)
    adapted_unet = load_unet_adapted(args.adapted_dir, args.base_dir_for_config or args.base_dir)
    ad_total, ad_trainable = count_params(adapted_unet)

    print("=== Parameter Counts ===")
    print(f"Base UNet total     : {base_total:,} ({sizeof_mb(base_total):.2f} MB @fp32)")
    print(f"Base UNet trainable : {base_trainable:,}")
    print(f"Adapted UNet total  : {ad_total:,} ({sizeof_mb(ad_total):.2f} MB @fp32)")
    print(f"Adapted trainable   : {ad_trainable:,}")
    diff = ad_total - base_total
    sign = "+" if diff >= 0 else "-"
    print(f"Diff (adapted-base) : {sign}{abs(diff):,} params ({sign}{sizeof_mb(abs(diff)):.2f} MB @fp32)")

    # Optional: rough breakdown of modules containing 'Mamba'
    mamba_params = sum(p.numel() for n, p in adapted_unet.named_parameters() if "Mamba" in n or "mamba" in n.lower())
    if mamba_params > 0:
        print(f"Mamba-related params: {mamba_params:,} (~{sizeof_mb(mamba_params):.2f} MB @fp32)")


if __name__ == "__main__":
    main()
