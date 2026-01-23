# =============================================
# File: /workspace/stereocraft/blocks/mamba_spatial.py
# ---------------------------------------------
# 目的: 空間ミキサのConvブロック
# =============================================

import torch
import torch.nn as nn


class SpatialMixer(nn.Module):
    def __init__(self, channels: int, hidden_mult: int = 2):
        super().__init__()
        hidden = channels * hidden_mult
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        return self.block(x)
