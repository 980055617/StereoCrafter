import types
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from mamba_temporal.adapters.temporal_mamba_adapter import TemporalMambaAdapter


class _BFState:
    """
    UNet の forward 直前に B, F をキャプチャし、登録済みアダプタへ配布するための状態管理。
    """
    def __init__(self):
        self.B: Optional[int] = None
        self.F: Optional[int] = None
        self.adapters: List[TemporalMambaAdapter] = []


def _infer_BF_from_sample(sample: torch.Tensor, unet) -> Tuple[int, int]:
    """
    sample 形状から (B,F) を推定。
    - [B, F, C, H, W] の場合は直読。
    - [B*F, C, H, W] の場合は、unet に事前に設定された _B, _F を使う。
      → pipeline側から set_unet_BF(unet, B, F) を呼ぶ実装にするのが安全。
    """
    if sample.dim() == 5:  # [B, F, C, H, W]
        return int(sample.shape[0]), int(sample.shape[1])

    if sample.dim() == 4:  # [B*F, C, H, W]
        B = getattr(unet, "_B", None)
        F = getattr(unet, "_F", None)
        if B is None or F is None:
            raise RuntimeError(
                "Cannot infer (B,F) from [B*F, C, H, W]. "
                "Set unet._B and unet._F before calling UNet, or pass 5D input."
            )
        if B * F != int(sample.shape[0]):
            raise RuntimeError(f"Mismatch: B*F={B*F}, sample[0]={int(sample.shape[0])}")
        return B, F

    raise ValueError(f"Unexpected sample shape: {tuple(sample.shape)}")


def set_unet_BF(unet: nn.Module, B: int, F: int):
    """
    pipeline 側から（UNet呼び出し直前に）B,F を明示セットしたいとき用。
    """
    setattr(unet, "_B", int(B))
    setattr(unet, "_F", int(F))

def attach_temporal_mamba_adapters(
    unet: nn.Module,
    where=("pre_down","post_mid"),
    d_state=128,
    headdim=64,
    num_groups=32,
):
    handles = {}
    state = _BFState()

    def _pre_hook(module, inputs):
        sample = inputs[0]
        B, F = _infer_BF_from_sample(sample, unet)
        state.B, state.F = B, F
        for a in state.adapters:
            a.set_batch_frames(B, F)

    handles["unet_pre"] = unet.register_forward_pre_hook(_pre_hook)

    # --- アダプタ作成 ---
    adapters = {}
    if "pre_down" in where:
        C0 = unet.conv_in.out_channels
        adapters["pre_down"] = TemporalMambaAdapter(C0, d_state=d_state, headdim=headdim, num_groups=num_groups)
        state.adapters.append(adapters["pre_down"])
    if "post_mid" in where:
        Cmid = getattr(getattr(unet, "up_blocks", [None])[0], "in_channels", None)
        if Cmid is None:
            Cmid = unet.config.block_out_channels[-1]
        adapters["post_mid"] = TemporalMambaAdapter(Cmid, d_state=d_state, headdim=headdim, num_groups=num_groups)
        state.adapters.append(adapters["post_mid"])

    # --- ★ サブモジュール登録（これで unet.to(device) に追随） ---
    if not hasattr(unet, "_mamba_temporal"):
        unet._mamba_temporal = nn.ModuleDict()
    for k, m in adapters.items():
        unet._mamba_temporal[k] = m  # register

    # --- フック（呼び出し直前に強制的に device/dtype を合わせる） ---
    if "pre_down" in where:
        def _hook_after_conv_in(module, inp, out):
            x = out
            if x.dim() == 5:
                B, F = int(x.shape[0]), int(x.shape[1])
                x = x.view(B*F, x.shape[2], x.shape[3], x.shape[4])
            adapter = unet._mamba_temporal["pre_down"]
            # ★ここで強制的に合わせる（device & dtype）
            adapter.to(device=x.device, dtype=x.dtype)
            return adapter(x)
        handles["conv_in"] = unet.conv_in.register_forward_hook(_hook_after_conv_in)

    if "post_mid" in where:
        def _hook_after_mid(module, inp, out):
            x = out
            adapter = unet._mamba_temporal["post_mid"]
            adapter.to(device=x.device, dtype=x.dtype)   # ★同様
            return adapter(x)
        handles["mid_block"] = unet.mid_block.register_forward_hook(_hook_after_mid)

    return handles

def detach_hooks(handles: Dict[str, torch.utils.hooks.RemovableHandle]):
    for k, h in list(handles.items()):
        try:
            h.remove()
        except Exception:
            pass
        handles.pop(k, None)
