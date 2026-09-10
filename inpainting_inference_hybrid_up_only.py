"""Run the current 0160 hybrid up-only Mamba inference preset.

This preset preserves the strongest current Mamba baseline:

- load the full gated/residual exported Mamba checkpoint
- replace only `up_blocks.*` self-attention with Mamba
- keep down/mid self-attention as the reference attention path
- use the matched 1024x576 per-eye inference config
"""

from __future__ import annotations

import os
import sys
from typing import Any

from fire import Fire


DEFAULT_CONFIG = "config/0160_overfit_inference_matched.json"
DEFAULT_SAVE_DIR = "outputs/diagnose_0160/hybrid_up_only_preset"
DEFAULT_UNET_STATE_PATH = (
    "weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/"
    "train_state_final_mamba_only.pt"
)
DEFAULT_INCLUDE = "up_blocks.*"


def main(
    config: str = DEFAULT_CONFIG,
    save_dir: str = DEFAULT_SAVE_DIR,
    unet_state_path: str = DEFAULT_UNET_STATE_PATH,
    include_patterns: str = DEFAULT_INCLUDE,
    exclude_patterns: str | None = None,
    bidirectional_mode: str | None = None,
    **overrides: Any,
) -> None:
    """Run matched inference with the hybrid up-only replacement filter."""

    os.environ["MAMBA_SELF_ATTN_INCLUDE"] = include_patterns
    if exclude_patterns is None:
        os.environ.pop("MAMBA_SELF_ATTN_EXCLUDE", None)
    else:
        os.environ["MAMBA_SELF_ATTN_EXCLUDE"] = exclude_patterns
    if bidirectional_mode is not None:
        os.environ["MAMBA_BIDIRECTIONAL_MODE"] = bidirectional_mode

    from inpainting_inference import run

    run(
        config=config,
        save_dir=save_dir,
        unet_state_path=unet_state_path,
        expected_partial_unet_state=True,
        **overrides,
    )


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Usage: python inpainting_inference_hybrid_up_only.py "
            "[--save_dir PATH] [--config PATH] [--unet_state_path PATH] "
            "[--include_patterns PATTERN] [--bidirectional_mode both|fwd|bwd] "
            "[inpainting override args...]\n\n"
            "Defaults:\n"
            f"  config: {DEFAULT_CONFIG}\n"
            f"  save_dir: {DEFAULT_SAVE_DIR}\n"
            f"  unet_state_path: {DEFAULT_UNET_STATE_PATH}\n"
            f"  include_patterns: {DEFAULT_INCLUDE}\n"
            "  bidirectional_mode: both unless MAMBA_BIDIRECTIONAL_MODE is set\n"
        )
        raise SystemExit(0)
    Fire(main)
