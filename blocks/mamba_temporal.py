# =============================================
# File: /workspace/stereocraft/blocks/mamba_temporal.py
# ---------------------------------------------
# 目的: 時間方向Mamba2ブロック
# =============================================

import os

import torch
import torch.nn as nn

from mamba_ssm import Mamba2


_TORCH_COMPILE_GLOBAL_DISABLED = False


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
        self._compiled_core = None
        self._compile_failed = False

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return bool(default)
        value = value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _core_for_forward(self):
        global _TORCH_COMPILE_GLOBAL_DISABLED
        if not self._env_bool("MAMBA_SELF_ATTN_TORCH_COMPILE", False):
            return self.core
        if self._compile_failed or _TORCH_COMPILE_GLOBAL_DISABLED:
            return self.core
        if self._compiled_core is None:
            if not hasattr(torch, "compile"):
                self._compile_failed = True
                _TORCH_COMPILE_GLOBAL_DISABLED = True
                return self.core
            mode = os.getenv("MAMBA_SELF_ATTN_TORCH_COMPILE_MODE", "reduce-overhead")
            try:
                self._compiled_core = torch.compile(self.core, mode=mode)
            except Exception as exc:
                self._compile_failed = True
                _TORCH_COMPILE_GLOBAL_DISABLED = True
                print(f"[TemporalMamba][compile][warn] disabled after compile failure: {exc}")
                return self.core
        return self._compiled_core

    def _cuda_forward(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        """Execute Mamba2 on the tensor's device, respecting mem-eff fallbacks."""
        global _TORCH_COMPILE_GLOBAL_DISABLED
        use_mem_eff = bool(getattr(self.core, "use_mem_eff_path", False))
        core = self._core_for_forward()
        if not use_mem_eff:
            try:
                return core(x_bt_c)
            except RuntimeError as e:
                msg = str(e)
                if "causal_conv1d with channel last layout requires strides" in msg:
                    old = self.core.use_mem_eff_path
                    self.core.use_mem_eff_path = True
                    try:
                        return self._core_for_forward()(x_bt_c)
                    finally:
                        self.core.use_mem_eff_path = old
                raise
        try:
            return core(x_bt_c)
        except Exception as exc:
            if self._compiled_core is not None:
                self._compile_failed = True
                _TORCH_COMPILE_GLOBAL_DISABLED = True
                self._compiled_core = None
                print(f"[TemporalMamba][compile][warn] disabled after forward failure: {exc}")
                return self.core(x_bt_c)
            raise

    def forward(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        # x_bt_c: (B*H*W, T, C) -> Mamba2 expects (B, L, C)
        if x_bt_c.is_cuda and hasattr(self, "core"):
            device = x_bt_c.device
            # Triton kernel inside Mamba2 expects the current device to match tensor.device.
            with torch.cuda.device(device):
                return self._cuda_forward(x_bt_c)
        # CPU などは通常経路
        return self._core_for_forward()(x_bt_c)
