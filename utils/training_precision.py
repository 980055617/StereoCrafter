"""Precision/AMP utilities.

"fp16" / "bf16" / "fp32" の指定を、学習スクリプトでそのまま使える形
(torch.dtype / autocast の enabled / GradScaler) に変換します。

注意:
- GradScaler は fp16 のときのみ有効 (bf16/fp32 では無効)。
- autocast は CUDA かつ (fp16/bf16) の時に有効。
"""

from typing import Tuple

import torch


class PrecisionConfigError(ValueError):
    """Raised when an unsupported precision keyword is provided."""
    pass


def resolve_precision(precision: str, device: torch.device) -> Tuple[torch.dtype, bool, torch.amp.GradScaler]:
    """Map precision keyword to dtype/autocast usage and create a GradScaler.

    Args:
        precision: "fp16" | "bf16" | "fp32"。
        device: 使用デバイス (autocast の可否判定に使用)。

    Returns:
        (torch_dtype, use_amp, scaler):
            - torch_dtype: 使用する dtype。
            - use_amp: autocast を有効化するか。
            - scaler: fp16 のとき有効な GradScaler (それ以外は無効)。
    """
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
