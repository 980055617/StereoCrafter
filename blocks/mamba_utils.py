# =============================================
# File: blocks/mamba_utils.py
# ---------------------------------------------
# 目的:
#  - 時間埋め込み (A/B)
#  - FiLM 条件注入 (A)
#  - AlphaBlender (B)
#  - 補助ユーティリティ
# =============================================
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbeddings(nn.Module):
    """
    Diffusers 相当のシンプルな timestep → embedding.
    返す埋め込みサイズは emb_dim.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4), nn.SiLU(), nn.Linear(emb_dim * 4, emb_dim)
        )

    @staticmethod
    def _build_sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
        # t: (B*T,) or (B,) の整数/実数ステップ
        device, dtype = t.device, t.dtype
        half = dim // 2
        exponents = torch.arange(half, device=device, dtype=dtype)
        freqs = torch.exp(-math.log(10000) * exponents / max(half, 1))  # (half,)
        args = t[:, None] * freqs[None, :]  # (N, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: (B,) or (B*T,) 実数でもOK
        sin = self._build_sinusoidal(timesteps, self.emb_dim)
        return self.proj(sin)


class FiLMConditioner(nn.Module):
    """
    encoder_hidden_states (B, N, C_ctx) を受け取り、
    gamma/beta (B, d_model) を生成して FiLM で注入。
    """
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


class AlphaBlender(nn.Module):
    """
    image_only_indicator が与えられたらそれを α として使い、
    無い場合は学習可能スカラ α∈[0,1] を使う。
    """
    def __init__(self, init_alpha: float = 0.5):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, temporal: torch.Tensor, spatial: torch.Tensor, image_only_indicator: Optional[torch.Tensor] = None) -> torch.Tensor:
        # temporal, spatial: (B,C,T,H,W)
        if image_only_indicator is not None:
            # 想定: (B,) or (B,1) or (B,T)
            alpha = image_only_indicator
            while alpha.dim() < temporal.dim():
                alpha = alpha.unsqueeze(-1)
            # 形: (B,1, T,1,1) 等にしてブロードキャスト
            return temporal * alpha + spatial * (1 - alpha)
        # 学習可能 α スカラ
        alpha = torch.sigmoid(self.raw)
        return temporal * alpha + spatial * (1 - alpha)


def apply_film(x: torch.Tensor, gamma: Optional[torch.Tensor], beta: Optional[torch.Tensor]) -> torch.Tensor:
    if gamma is None or beta is None:
        return x
    # x: (B, C, ...), gamma/beta: (B, C)
    # ブロードキャスト
    shape = [x.shape[0], x.shape[1]] + [1] * (x.dim() - 2)
    return x * gamma.view(*shape) + beta.view(*shape)


