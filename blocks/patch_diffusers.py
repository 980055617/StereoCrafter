# blocks/patch_diffusers.py
from __future__ import annotations
import torch.nn as nn
from typing import Callable, Optional
from .mamba_spatiotemporal import MambaSpatioTemporalModel


def _get_parent_and_attr(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _guess_in_channels(parent: nn.Module, attr: str, fallback: int) -> int:
    """
    置換元 Transformer の直前/内部の層から in_channels を推定する。
    - proj_in があればその out_channels
    - module 側に in_channels/inner_dim/hidden_size があればそれ
    - それ以外は fallback
    （※ ログ専用化に伴い、ここも推定のみで変更は行わない）
    """
    mod = getattr(parent, attr)
    for k in ("in_channels", "inner_dim", "hidden_size"):
        if hasattr(mod, k) and isinstance(getattr(mod, k), int):
            return int(getattr(mod, k))
    proj_in = getattr(mod, "proj_in", None)
    if proj_in is not None and hasattr(proj_in, "out_channels"):
        return int(getattr(proj_in, "out_channels"))
    return int(fallback)


def replace_transformers_with_mamba(
    unet: nn.Module,
    keep_mid_attn: bool = True,
    d_model: Optional[int] = None,
    is_target: Optional[Callable[[str, nn.Module], bool]] = None,
) -> int:
    """
    UNet 内の TransformerSpatioTemporalModel を MambaSpatioTemporalModel に置換。
    - Mid を残したい場合は keep_mid_attn=True。
    - d_model を指定しなければ in==out 前提で Mamba 側が内部決定。
    - 返り値: 置換した数
    """
    count = 0
    fallback_c = (
        unet.config.block_out_channels[-1]
        if hasattr(unet, "config") and hasattr(unet.config, "block_out_channels")
        else 320
    )

    for name, module in list(unet.named_modules()):
        if module.__class__.__name__ != "TransformerSpatioTemporalModel":
            continue
        if keep_mid_attn and "mid_block" in name:
            continue
        if is_target and not is_target(name, module):
            continue

        parent, attr = _get_parent_and_attr(unet, name)
        in_ch = _guess_in_channels(parent, attr, fallback=fallback_c)

        # 置換（ここは仕様通り差し替え。テンソルはここでは扱わない）
        mamba = MambaSpatioTemporalModel(
            in_channels=in_ch,
            d_model=d_model,  # None -> 実装側で in==out に整合
        )
        setattr(parent, attr, mamba)
        count += 1

    return count
