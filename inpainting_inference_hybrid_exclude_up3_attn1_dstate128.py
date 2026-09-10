"""Run selective hybrid inference for a d_state=128 Mamba2 checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fire import Fire

from inpainting_inference_hybrid_exclude_up3_attn1 import (
    DEFAULT_CONFIG,
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
)


DEFAULT_SAVE_DIR = "outputs/diagnose_0160/hybrid_exclude_up3_attn1_dstate128_preset"
DEFAULT_UNET_STATE_PATH = (
    "weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128FromFullGated/"
    "MambaCrafter_LATEST/train_state_final_mamba_only.pt"
)
DEFAULT_D_STATE = 128


def _resolve_latest_checkpoint(path: str) -> str:
    marker = "MambaCrafter_LATEST"
    if marker not in path:
        return path
    raw = Path(path)
    parts = raw.parts
    try:
        marker_index = parts.index(marker)
    except ValueError:
        return path
    base = Path(*parts[:marker_index])
    suffix = Path(*parts[marker_index + 1 :])
    matches = sorted(base.glob("MambaCrafter_*"), key=lambda item: item.stat().st_mtime)
    if not matches:
        return path
    return str(matches[-1] / suffix)


def main(
    config: str = DEFAULT_CONFIG,
    save_dir: str = DEFAULT_SAVE_DIR,
    unet_state_path: str = DEFAULT_UNET_STATE_PATH,
    include_patterns: str = DEFAULT_INCLUDE,
    exclude_patterns: str = DEFAULT_EXCLUDE,
    d_state: int = DEFAULT_D_STATE,
    bidirectional_mode: str | None = None,
    **overrides: Any,
) -> None:
    """Run matched inference with the d_state=128 selective up3.attn1 preset."""

    os.environ["MAMBA_SELF_ATTN_D_STATE"] = str(int(d_state))
    unet_state_path = _resolve_latest_checkpoint(unet_state_path)

    from inpainting_inference_hybrid_exclude_up3_attn1 import main as base_main

    base_main(
        config=config,
        save_dir=save_dir,
        unet_state_path=unet_state_path,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        bidirectional_mode=bidirectional_mode,
        **overrides,
    )


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Usage: python inpainting_inference_hybrid_exclude_up3_attn1_dstate128.py "
            "[--save_dir PATH] [--config PATH] [--unet_state_path PATH] "
            "[--include_patterns PATTERN] [--exclude_patterns PATTERN] "
            "[--d_state 128] [--bidirectional_mode both|fwd|bwd] "
            "[inpainting override args...]\n\n"
            "Defaults:\n"
            f"  config: {DEFAULT_CONFIG}\n"
            f"  save_dir: {DEFAULT_SAVE_DIR}\n"
            f"  unet_state_path: {DEFAULT_UNET_STATE_PATH}\n"
            f"  include_patterns: {DEFAULT_INCLUDE}\n"
            f"  exclude_patterns: {DEFAULT_EXCLUDE}\n"
            f"  d_state: {DEFAULT_D_STATE}\n"
            "  bidirectional_mode: both unless MAMBA_BIDIRECTIONAL_MODE is set\n"
        )
        raise SystemExit(0)
    Fire(main)
