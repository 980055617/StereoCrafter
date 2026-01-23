# =============================================
# File: /workspace/stereocraft/blocks/mamba_utils.py
# ---------------------------------------------
# 目的: FiLM条件付けのユーティリティ
# =============================================

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    """Generate FiLM gamma/beta from encoder_hidden_states."""
    def __init__(self, ctx_dim: int, d_model: int):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.Linear(ctx_dim, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model * 2)
        )

    def forward(self, ctx: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ctx is None:
            return None, None
        # ctx: (B, N, C) or (B, C)
        if ctx.dim() == 3:
            pooled = ctx.mean(dim=1)  # (B, C)
        else:
            pooled = ctx  # (B, C)
        gb = self.to_gb(pooled)  # (B, 2*d)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


def apply_film(x: torch.Tensor, gamma: Optional[torch.Tensor], beta: Optional[torch.Tensor]) -> torch.Tensor:
    if gamma is None or beta is None:
        return x
    shape = [x.shape[0], x.shape[1]] + [1] * (x.dim() - 2)
    return x * gamma.view(*shape) + beta.view(*shape)
