# =============================================
# File: /workspace/stereocraft/scripts/extract_unet_from_train_state.py
# ---------------------------------------------
# 目的: train_stateからUNet抽出
# =============================================

import argparse
import os
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract UNet state_dict from a train_state_*.pt checkpoint"
    )
    parser.add_argument("train_state_path", help="Path to train_state_*.pt")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for UNet state_dict (.pt). Defaults to <train_state>_unet.pt",
    )
    args = parser.parse_args()

    src = args.train_state_path
    if not os.path.isfile(src):
        print(f"Input checkpoint not found: {src}", file=sys.stderr)
        return 1

    try:
        ckpt = torch.load(src, map_location="cpu")
    except Exception as err:
        print(f"Failed to load checkpoint: {err}", file=sys.stderr)
        return 1

    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print("Checkpoint does not contain 'model' key (UNet state_dict).", file=sys.stderr)
        return 1

    out_path = args.out or f"{os.path.splitext(src)[0]}_unet.pt"
    try:
        torch.save(ckpt["model"], out_path)
    except Exception as err:
        print(f"Failed to save UNet state_dict: {err}", file=sys.stderr)
        return 1

    print(f"Saved UNet state_dict to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
