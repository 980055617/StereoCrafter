# =============================================
# File: /workspace/stereocraft/utils/model_io.py
# ---------------------------------------------
# 目的: モデル重みパス解決
# =============================================

from __future__ import annotations

from pathlib import Path


def resolve_unet_state_path(unet_state_path: str | None) -> str | None:
    if unet_state_path is None:
        return None
    candidate = Path(unet_state_path).expanduser()
    if candidate.is_dir():
        pt_files = sorted(
            p for p in candidate.iterdir() if p.is_file() and p.suffix == ".pt"
        )
        if len(pt_files) == 1:
            return str(pt_files[0])
        raise ValueError(
            f"unet_state_path directory has {len(pt_files)} .pt files; "
            "please pass the specific .pt file path."
        )
    if not candidate.exists():
        raise FileNotFoundError(f"unet_state_path not found: {candidate}")
    return str(candidate)
