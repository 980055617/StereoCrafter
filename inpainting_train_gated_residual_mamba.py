"""Train with a gated residual attn1->Mamba replacement, then export Mamba-only weights."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import torch
from fire import Fire


def _load_config(config: str | None, config_dir: str) -> dict[str, Any]:
    if not config:
        return {}
    path = Path(config)
    if not path.exists():
        path = Path(config_dir) / config
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _resolve_latest_run_dir(save_dir: str) -> Path | None:
    base = Path(save_dir)
    if (base / "train_state_final.pt").exists():
        return base
    matches = sorted(glob.glob(str(base / "MambaCrafter_*")), key=os.path.getmtime)
    if not matches:
        return None
    return Path(matches[-1])


def _strip_gated_reference_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if ".origin_attn." not in key and not key.endswith(".mamba_gate")
    }


def export_mamba_only_checkpoint(run_dir: str | Path) -> None:
    run_path = Path(run_dir)
    ckpt_path = run_path / "train_state_final.pt"
    if not ckpt_path.exists():
        return
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(model, dict):
        return
    stripped_model = _strip_gated_reference_state(model)

    slim_ckpt = dict(ckpt)
    slim_ckpt["model"] = stripped_model
    slim_ckpt["gated_reference_stripped"] = True
    slim_ckpt["recommended_inference_state"] = "train_state_final_mamba_only.pt"
    torch.save(slim_ckpt, str(run_path / "train_state_final_mamba_only.pt"))
    torch.save(stripped_model, str(run_path / "unet_final_mamba_only.pt"))
    print(f"Exported Mamba-only checkpoint: {run_path / 'train_state_final_mamba_only.pt'}")


def main(
    config: str | None = None,
    config_dir: str = "train_config",
    export_mamba_only: bool = True,
    **overrides: Any,
) -> None:
    os.environ["MAMBA_SELF_ATTN_REPLACEMENT"] = "gated_residual"
    os.environ.setdefault("MAMBA_SELF_ATTN_INITIAL_GATE", str(overrides.get("mamba_gate_start", 0.0)))

    from inpainting_train import main as train_main

    train_main(config=config, config_dir=config_dir, **overrides)

    local_rank = str(os.environ.get("LOCAL_RANK", "")).strip()
    if local_rank not in {"", "0"} or not export_mamba_only:
        return
    config_values = _load_config(config, config_dir)
    config_values.update(overrides)
    save_dir = str(config_values.get("save_dir", "")).strip()
    if not save_dir:
        return
    run_dir = _resolve_latest_run_dir(save_dir)
    if run_dir is not None:
        export_mamba_only_checkpoint(run_dir)


if __name__ == "__main__":
    Fire(main)
