"""Train a lighter d_state=128 Mamba2 variant for the selective up3.attn1 policy.

This is a new architecture-size branch. The current baseline uses Mamba2 with
`d_state=256`; this wrapper keeps the same selective replacement policy but
initializes `d_state=128` Mamba blocks and resumes only compatible checkpoint
weights from the full-gated e102 seed.
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
DEFAULT_SAVE_DIR = "weights/Overfit0160GatedResidualMambaUpOnlyExcludeUp3Attn1DState128FromFullGated/"
DEFAULT_INCLUDE = "up_blocks.*"
DEFAULT_EXCLUDE = "up_blocks.3.attentions.1.*"
DEFAULT_D_STATE = 128
DEFAULT_STAGE_EPOCHS = [50, 150, 103]
DEFAULT_RESUME_IGNORE = ".fwd.,.bwd.,.time_embed_proj.,.mamba_gate"


def main(
    config: str = DEFAULT_CONFIG,
    resume_from: str = DEFAULT_RESUME_FROM,
    save_dir: str = DEFAULT_SAVE_DIR,
    include_patterns: str = DEFAULT_INCLUDE,
    exclude_patterns: str = DEFAULT_EXCLUDE,
    d_state: int = DEFAULT_D_STATE,
    stage_epochs: list[int] | str = DEFAULT_STAGE_EPOCHS,
    resume_ignore_key_patterns: str = DEFAULT_RESUME_IGNORE,
    **overrides: Any,
) -> None:
    """Run one d_state=128 gated/residual continuation epoch from the e102 seed."""

    os.environ["MAMBA_SELF_ATTN_INCLUDE"] = include_patterns
    os.environ["MAMBA_SELF_ATTN_EXCLUDE"] = exclude_patterns
    os.environ["MAMBA_SELF_ATTN_D_STATE"] = str(int(d_state))

    from inpainting_train_gated_residual_mamba import main as gated_train_main

    merged_overrides = {
        "resume_from": resume_from,
        "resume_into_source_dir": False,
        "resume_ignore_mismatched_shapes": True,
        "resume_ignore_key_patterns": resume_ignore_key_patterns,
        "save_dir": save_dir,
        "stage_epochs": stage_epochs,
        "mamba_gate_schedule": "linear",
        "mamba_gate_start": 0.0,
        "mamba_gate_end": 1.0,
        "save_interval_epochs": 1,
        **overrides,
    }
    gated_train_main(config=config, **merged_overrides)


if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Usage: python inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1_dstate128.py "
            "[--config PATH] [--resume_from PATH] [--save_dir PATH] "
            "[--include_patterns PATTERN] [--exclude_patterns PATTERN] "
            "[--d_state 128] [training override args...]\n\n"
            "Defaults:\n"
            f"  config: {DEFAULT_CONFIG}\n"
            f"  resume_from: {DEFAULT_RESUME_FROM}\n"
            f"  save_dir: {DEFAULT_SAVE_DIR}\n"
            f"  include_patterns: {DEFAULT_INCLUDE}\n"
            f"  exclude_patterns: {DEFAULT_EXCLUDE}\n"
            f"  d_state: {DEFAULT_D_STATE}\n"
            f"  stage_epochs: {DEFAULT_STAGE_EPOCHS}\n"
            f"  resume_ignore_key_patterns: {DEFAULT_RESUME_IGNORE}\n"
            "  mamba_gate_start/end: 0.0 / 1.0\n"
        )
        raise SystemExit(0)
    Fire(main)
