"""Continue the best gated/residual Mamba overfit run for 50 more epochs.

Run with DeepSpeed, for example:

DS_ZERO_GRAD_FN_MODE=enable_grad deepspeed --num_gpus=2 --master_port=29508 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_continue_50.py
"""

from __future__ import annotations

from typing import Any

from fire import Fire


DEFAULT_CONFIG = "config/0160_overfit_gated_residual_mamba.json"
DEFAULT_RESUME_FROM = (
    "weights/Overfit0160GatedResidualMambaContinue50/"
    "MambaCrafter_20260530_124218/train_state_epoch000111.pt"
)
DEFAULT_SAVE_DIR = "weights/Overfit0160GatedResidualMambaContinue50/"
DEFAULT_STAGE_EPOCHS = [50, 150, 152]


def main(
    config: str = DEFAULT_CONFIG,
    resume_from: str = DEFAULT_RESUME_FROM,
    save_dir: str = DEFAULT_SAVE_DIR,
    stage_epochs: list[int] | str = DEFAULT_STAGE_EPOCHS,
    **overrides: Any,
) -> None:
    """Resume from the current best gated run and train stage3 through epoch152."""

    from inpainting_train_gated_residual_mamba import main as gated_train_main

    merged_overrides = {
        "resume_from": resume_from,
        "resume_into_source_dir": False,
        "save_dir": save_dir,
        "stage_epochs": stage_epochs,
        "mamba_gate_schedule": "none",
        "mamba_gate_start": 1.0,
        "mamba_gate_end": 1.0,
        **overrides,
    }
    gated_train_main(config=config, **merged_overrides)


if __name__ == "__main__":
    Fire(main)
