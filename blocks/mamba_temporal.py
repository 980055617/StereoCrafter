# =============================================
# File: blocks/mamba_temporal.py
# ---------------------------------------------
# 目的: 時間モデリング（C）
#  方針: (B*H*W, T, C) を Mamba2 に投入
# =============================================
import torch
import torch.nn as nn

from mamba_ssm import Mamba2

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

    def _cuda_forward(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        """Execute Mamba2 on the tensor's device, respecting mem-eff fallbacks."""
        use_mem_eff = bool(getattr(self.core, "use_mem_eff_path", False))
        if not use_mem_eff:
            try:
                return self.core(x_bt_c)
            except RuntimeError as e:
                msg = str(e)
                if "causal_conv1d with channel last layout requires strides" in msg:
                    old = self.core.use_mem_eff_path
                    self.core.use_mem_eff_path = True
                    try:
                        return self.core(x_bt_c)
                    finally:
                        self.core.use_mem_eff_path = old
                raise
        return self.core(x_bt_c)

    def forward(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        # x_bt_c: (B*H*W, T, C)
        # Mamba2 は (B, L, C)。ここでは B'=(B*H*W), L=T
        # CUDA 環境で非メモリ効率パスが stride 制約で失敗する場合、自動で mem‑eff パスへフォールバック。
        if x_bt_c.is_cuda and hasattr(self, "core"):
            device = x_bt_c.device
            # Triton kernel inside Mamba2 expects the current device to match tensor.device.
            with torch.cuda.device(device):
                return self._cuda_forward(x_bt_c)
        # CPU などは通常経路
        return self.core(x_bt_c)
