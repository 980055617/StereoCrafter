# =============================================
# File: /workspace/stereocraft/utils/training_precision.py
# ---------------------------------------------
# 目的: 精度/AMP設定の解決
# =============================================

"""Precision/AMP helpers for training."""

from typing import Tuple

import torch


class PrecisionConfigError(ValueError):
    """Raised when an unsupported precision keyword is provided."""
    pass


def resolve_precision(precision: str, device: torch.device) -> Tuple[torch.dtype, bool, torch.amp.GradScaler]:
    """Map precision keyword to dtype/autocast/scaler."""
    precision_key = (precision or "").lower()
    dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if precision_key not in dtype_mapping:
        raise PrecisionConfigError(f"Unsupported precision '{precision}'. Choose from {list(dtype_mapping.keys())}.")

    torch_dtype = dtype_mapping[precision_key]
    use_amp = device.type == "cuda" and precision_key in ("fp16", "bf16")
    scaler = torch.amp.GradScaler('cuda', enabled=precision_key == "fp16")
    return torch_dtype, use_amp, scaler
