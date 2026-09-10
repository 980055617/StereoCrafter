"""Train the selective hybrid candidate that excludes up3.attn1 from Mamba.

This wrapper codifies the current best replacement policy from the 0160/0204
layer-sensitivity sweep:

- replace `up_blocks.*` self-attention with gated/residual Mamba
- keep `up_blocks.3.attentions.1.*` on the reference attention path
- default to the confirmed full-gated e102 model-only seed

Run with DeepSpeed, for example:

MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' MAMBA_SELF_ATTN_EXCLUDE='up_blocks.3.attentions.1.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29526 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from fire import Fire


DEFAULT_CONFIG = "config/0160_overfit_gated_residual_mamba.json"
DEFAULT_RESUME_FROM = (
    "weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/"
    "train_state_epoch000102_model_only_no_ds.pt"
)
DEFAULT_SAVE_DIR = "weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1FromFullGated/"
DEFAULT_INCLUDE = "up_blocks.*"
DEFAULT_EXCLUDE = "up_blocks.3.attentions.1.*"
DEFAULT_STAGE_EPOCHS = [50, 150, 103]


def main(
    config: str = DEFAULT_CONFIG,
    resume_from: str = DEFAULT_RESUME_FROM,
    save_dir: str = DEFAULT_SAVE_DIR,
    include_patterns: str = DEFAULT_INCLUDE,
    exclude_patterns: str = DEFAULT_EXCLUDE,
    stage_epochs: list[int] | str = DEFAULT_STAGE_EPOCHS,
    **overrides: Any,
) -> None:
    """Resume from the full-gated e102 seed with the selective up3.attn1 exclusion."""

    os.environ["MAMBA_SELF_ATTN_INCLUDE"] = include_patterns
    os.environ["MAMBA_SELF_ATTN_EXCLUDE"] = exclude_patterns

    from inpainting_train_gated_residual_mamba import main as gated_train_main

    merged_overrides = {
        "resume_from": resume_from,
        "resume_into_source_dir": False,
        "save_dir": save_dir,
        "stage_epochs": stage_epochs,
        "mamba_gate_schedule": "linear",
        "mamba_gate_start": 1.0,
        "mamba_gate_end": 1.0,
        "save_interval_epochs": 1,
        **overrides,
    }
    gated_train_main(config=config, **merged_overrides)


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Usage: python inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py "
            "[--config PATH] [--resume_from PATH] [--save_dir PATH] "
            "[--include_patterns PATTERN] [--exclude_patterns PATTERN] "
            "[training override args...]\n\n"
            "Defaults:\n"
            f"  config: {DEFAULT_CONFIG}\n"
            f"  resume_from: {DEFAULT_RESUME_FROM}\n"
            f"  save_dir: {DEFAULT_SAVE_DIR}\n"
            f"  include_patterns: {DEFAULT_INCLUDE}\n"
            f"  exclude_patterns: {DEFAULT_EXCLUDE}\n"
            f"  stage_epochs: {DEFAULT_STAGE_EPOCHS}\n"
            "  mamba_gate_start/end: 1.0 / 1.0\n"
        )
        raise SystemExit(0)
    Fire(main)
