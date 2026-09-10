"""Train a gated/residual Mamba replacement only for up-block attn1 modules.

Run with DeepSpeed, for example:

MAMBA_SELF_ATTN_INCLUDE='up_blocks.*' DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29509 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only.py

The include env var is also set by this wrapper; it is shown in the command so
the run folder and shell history make the ablation explicit.
"""

from __future__ import annotations

import os
from typing import Any

from fire import Fire


DEFAULT_CONFIG = "config/0160_overfit_gated_residual_mamba.json"
DEFAULT_RESUME_FROM = "weights/Overfit0160/MambaCrafter_20260507_174452/train_state_epoch000100.pt"
DEFAULT_SAVE_DIR = "weights/Overfit0160GatedResidualMambaUpOnly/"
DEFAULT_INCLUDE = "up_blocks.*"
DEFAULT_STAGE_EPOCHS = [50, 150, 102]


def main(
    config: str = DEFAULT_CONFIG,
    resume_from: str = DEFAULT_RESUME_FROM,
    save_dir: str = DEFAULT_SAVE_DIR,
    include_patterns: str = DEFAULT_INCLUDE,
    stage_epochs: list[int] | str = DEFAULT_STAGE_EPOCHS,
    **overrides: Any,
) -> None:
    """Resume from the 0160 stage2/early-stage3 checkpoint and train only up-block Mamba."""

    os.environ["MAMBA_SELF_ATTN_INCLUDE"] = include_patterns

    from inpainting_train_gated_residual_mamba import main as gated_train_main

    merged_overrides = {
        "resume_from": resume_from,
        "resume_into_source_dir": False,
        "save_dir": save_dir,
        "stage_epochs": stage_epochs,
        "mamba_gate_schedule": "linear",
        "mamba_gate_start": 0.05,
        "mamba_gate_end": 1.0,
        "save_interval_epochs": 1,
        **overrides,
    }
    gated_train_main(config=config, **merged_overrides)


if __name__ == "__main__":
    Fire(main)
