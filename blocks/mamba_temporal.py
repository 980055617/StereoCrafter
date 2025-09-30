# =============================================
# File: blocks/mamba_temporal.py
# ---------------------------------------------
# 目的: 時間モデリング（C）
#  方針: (B*H*W, T, C) を Mamba2 に投入
# =============================================
import torch
import torch.nn as nn

from mamba_ssm import Mamba2

try:
    import torch
    if not torch.cuda.is_available():
        import mamba_ssm.modules.mamba2 as m2
        m2.causal_conv1d_fn = None
        m2.causal_conv1d_update = None
except Exception:
    pass

class TemporalMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        use_mem_eff_path: bool = True,
    ):
        super().__init__()
        self.core = Mamba2(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
        )

    def forward(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        # x_bt_c: (B*H*W, T, C)
        # Mamba2 は (B, L, C)。ここでは B'=(B*H*W), L=T
        return self.core(x_bt_c)

