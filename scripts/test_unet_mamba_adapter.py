#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================
# File: /workspace/stereocraft/scripts/test_unet_mamba_adapter.py
# ---------------------------------------------
# 目的: Mambaアダプタの基本テスト
# =============================================

"""Basic tests for Mamba adapter replacement in UNet."""

import torch
from diffusers.models import UNetSpatioTemporalConditionModel

from blocks.mamba_diffusers_adapter import replace_unet_spatiotemporal_transformer_with_mamba


def _device_dtype():
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    dt = torch.float16 if use_cuda else torch.float32
    return dev, dt


def _make_unet(in_channels=8, out_channels=4, ctx_dim=1024, num_frames=8, use_cross_attn=True):
    # 小さめ構成で作成（down/up のブロック数と block_out_channels の長さを一致させる）
    if use_cross_attn:
        down_block_types = (
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        )
        up_block_types = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        )
    else:
        down_block_types = (
            "DownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        )
        up_block_types = (
            "UpBlockSpatioTemporal",
            "UpBlockSpatioTemporal",
        )
    block_out_channels = (320, 640)
    num_attention_heads = (5, 10)
    transformer_layers_per_block = (1, 1)

    model = UNetSpatioTemporalConditionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=ctx_dim,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        layers_per_block=1,
        num_attention_heads=num_attention_heads,
        transformer_layers_per_block=transformer_layers_per_block,
        num_frames=num_frames,
    )
    return model


def _inputs(B=1, F=4, C_in=8, H=32, W=32, ctx_dim=1024, dev=None, dt=torch.float32):
    x = torch.randn(B, F, C_in, H, W, device=dev, dtype=dt).requires_grad_(True)
    t = torch.tensor([10] * B, device=dev)
    ctx = torch.randn(B, 1, ctx_dim, device=dev, dtype=dt)
    # added_time_ids: (B, 3)
    add = torch.tensor([[6.0, 127.0, 0.0]], device=dev, dtype=dt).repeat(B, 1)
    return x, t, ctx, add


def _run_once(use_mem_eff: bool):
    dev, dt = _device_dtype()
    B, F, H, W = 1, 4, 32, 32
    C_in, C_out, ctx_dim = 8, 4, 1024

    unet = _make_unet(in_channels=C_in, out_channels=C_out, ctx_dim=ctx_dim, num_frames=F, use_cross_attn=True).to(dev).to(dtype=dt)
    # 置換（CPU では mem‑eff は無効推奨）
    replaced = replace_unet_spatiotemporal_transformer_with_mamba(
        unet,
        use_mem_eff_path=(torch.cuda.is_available() and use_mem_eff),
        temporal_chunk_size=8,
    )
    assert replaced > 0, "no modules were replaced"

    x, t, ctx, add = _inputs(B=B, F=F, C_in=C_in, H=H, W=W, ctx_dim=ctx_dim, dev=dev, dt=dt)

    y = unet(x, t, encoder_hidden_states=ctx, added_time_ids=add, return_dict=True)["sample"]
    assert y.shape == (B, F, C_out, H, W), f"unexpected shape: {tuple(y.shape)}"
    assert torch.isfinite(y).all(), "output contains NaN/Inf"

    # 逆伝播（軽く確認）
    loss = y.mean()
    loss.backward()
    assert torch.isfinite(x.grad).all(), "input grad has NaN/Inf"

    # 元の TransformerSpatioTemporalModel が残っていないことを確認
    def _has_orig_transformer(m):
        return m.__class__.__name__ == "TransformerSpatioTemporalModel"
    assert not any(_has_orig_transformer(m) for m in unet.modules()), "original Transformer remains"


def test_adapter_basic():
    # CPU/CUDA 共通: mem‑eff=False の一回
    _run_once(use_mem_eff=False)
    # CUDA 環境では mem‑eff=True でも実行
    if torch.cuda.is_available():
        _run_once(use_mem_eff=True)


def test_batch2_frames8():
    # B>1 / F>1 でも forward/backward が通ることを確認
    dev, dt = _device_dtype()
    B, F, H, W = 2, 8, 32, 32
    C_in, C_out, ctx_dim = 8, 4, 1024

    unet = _make_unet(in_channels=C_in, out_channels=C_out, ctx_dim=ctx_dim, num_frames=F, use_cross_attn=True).to(dev).to(dtype=dt)
    replaced = replace_unet_spatiotemporal_transformer_with_mamba(
        unet,
        use_mem_eff_path=(torch.cuda.is_available()),
        temporal_chunk_size=8,
    )
    assert replaced > 0

    x, t, ctx, add = _inputs(B=B, F=F, C_in=C_in, H=H, W=W, ctx_dim=ctx_dim, dev=dev, dt=dt)
    x.requires_grad_(True)
    y = unet(x, t, encoder_hidden_states=ctx, added_time_ids=add, return_dict=True)["sample"]
    assert y.shape == (B, F, C_out, H, W)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_zero_encoder_hidden_states():
    # Diffusers の UNet 実装は encoder_hidden_states を常に参照するため、
    # None は許容されない。クロスアテンション無し構成でも 0 埋めテンソルを渡して無条件ケースを模擬する。
    dev, dt = _device_dtype()
    B, F, H, W = 1, 4, 32, 32
    C_in, C_out, ctx_dim = 8, 4, 1024

    unet = _make_unet(in_channels=C_in, out_channels=C_out, ctx_dim=ctx_dim, num_frames=F, use_cross_attn=False).to(dev).to(dtype=dt)
    replace_unet_spatiotemporal_transformer_with_mamba(
        unet,
        use_mem_eff_path=(torch.cuda.is_available()),
        temporal_chunk_size=8,
    )

    x, t, _, add = _inputs(B=B, F=F, C_in=C_in, H=H, W=W, ctx_dim=ctx_dim, dev=dev, dt=dt)
    zero_ctx = torch.zeros(B, 1, ctx_dim, device=dev, dtype=dt)
    y = unet(x, t, encoder_hidden_states=zero_ctx, added_time_ids=add, return_dict=True)["sample"]
    assert y.shape == (B, F, C_out, H, W)
    assert torch.isfinite(y).all()


if __name__ == "__main__":
    test_adapter_basic()
    test_batch2_frames8()
    test_zero_encoder_hidden_states()
    print("[OK] UNet Mamba adapter test passed")
