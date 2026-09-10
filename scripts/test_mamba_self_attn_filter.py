#!/usr/bin/env python
"""Tests for selecting which attn1 modules are replaced with Mamba."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.nn as nn

from blocks.mamba_diffusers_adapter import (
    BiMambaSelfAttention,
    replace_unet_spatiotemporal_self_attn_with_mamba,
)


class FakeAttention(nn.Module):
    def __init__(self, dim: int = 64, dim_head: int = 8) -> None:
        super().__init__()
        self.query_dim = dim
        self.dim_head = dim_head
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)


class FakeTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = FakeAttention()


class TransformerSpatioTemporalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([FakeTransformerBlock()])
        self.temporal_transformer_blocks = nn.ModuleList()


def _attention_stack(count: int) -> nn.ModuleList:
    return nn.ModuleList([TransformerSpatioTemporalModel() for _ in range(count)])


class FakeBlock(nn.Module):
    def __init__(self, attention_count: int) -> None:
        super().__init__()
        self.attentions = _attention_stack(attention_count)


class FakeUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([FakeBlock(2), FakeBlock(2)])
        self.up_blocks = nn.ModuleList([FakeBlock(1)])
        self.mid_block = nn.Module()
        self.mid_block.attentions = _attention_stack(1)


@contextmanager
def _clean_filter_env():
    keys = [
        "MAMBA_SELF_ATTN_INCLUDE",
        "MAMBA_SELF_ATTN_EXCLUDE",
        "MAMBA_BIDIRECTIONAL_MODE",
        "MAMBA_INNER_PROFILE_JSON",
    ]
    old = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _attn1_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in model.named_modules()
        if name.endswith(".attn1")
    }


class CountingShift(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states + self.delta


def test_include_filter_replaces_only_matching_attn1_modules():
    with _clean_filter_env():
        model = FakeUNet()

        replaced = replace_unet_spatiotemporal_self_attn_with_mamba(
            model,
            d_state=4,
            expand=1,
            chunk_size=8,
            use_mem_eff_path=False,
            include_patterns=[
                "down_blocks.1",
                "mid_block.attentions.0",
            ],
        )

        attn1 = _attn1_modules(model)
        replaced_names = {
            name for name, module in attn1.items() if isinstance(module, BiMambaSelfAttention)
        }

        assert replaced == 3
        assert replaced_names == {
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn1",
            "mid_block.attentions.0.transformer_blocks.0.attn1",
        }


def test_exclude_filter_keeps_matching_attn1_modules_unreplaced():
    with _clean_filter_env():
        model = FakeUNet()

        replaced = replace_unet_spatiotemporal_self_attn_with_mamba(
            model,
            d_state=4,
            expand=1,
            chunk_size=8,
            use_mem_eff_path=False,
            exclude_patterns=["down_blocks.0"],
        )

        attn1 = _attn1_modules(model)
        replaced_names = {
            name for name, module in attn1.items() if isinstance(module, BiMambaSelfAttention)
        }

        assert replaced == 4
        assert all(not name.startswith("down_blocks.0") for name in replaced_names)
        assert len(replaced_names) == 4


def test_filter_env_vars_are_used_when_arguments_are_omitted():
    with _clean_filter_env():
        os.environ["MAMBA_SELF_ATTN_INCLUDE"] = "up_blocks.0,mid_block"
        os.environ["MAMBA_SELF_ATTN_EXCLUDE"] = "mid_block"
        model = FakeUNet()

        replaced = replace_unet_spatiotemporal_self_attn_with_mamba(
            model,
            d_state=4,
            expand=1,
            chunk_size=8,
            use_mem_eff_path=False,
        )

        attn1 = _attn1_modules(model)
        replaced_names = {
            name for name, module in attn1.items() if isinstance(module, BiMambaSelfAttention)
        }

        assert replaced == 1
        assert replaced_names == {"up_blocks.0.attentions.0.transformer_blocks.0.attn1"}


def test_bidirectional_mode_env_selects_forward_backward_or_both():
    with _clean_filter_env():
        attn = BiMambaSelfAttention(
            4,
            d_state=4,
            headdim=4,
            expand=1,
            chunk_size=8,
            use_mem_eff_path=False,
        )
        attn.fwd = CountingShift(2.0)
        attn.bwd = CountingShift(10.0)
        hidden = torch.zeros(1, 3, 4)

        os.environ["MAMBA_BIDIRECTIONAL_MODE"] = "fwd"
        out = attn(hidden)
        assert torch.allclose(out, torch.full_like(hidden, 2.0))
        assert attn.fwd.calls == 1
        assert attn.bwd.calls == 0

        os.environ["MAMBA_BIDIRECTIONAL_MODE"] = "bwd"
        out = attn(hidden)
        assert torch.allclose(out, torch.full_like(hidden, 10.0))
        assert attn.fwd.calls == 1
        assert attn.bwd.calls == 1

        os.environ["MAMBA_BIDIRECTIONAL_MODE"] = "both"
        out = attn(hidden)
        assert torch.allclose(out, torch.full_like(hidden, 6.0))
        assert attn.fwd.calls == 2
        assert attn.bwd.calls == 2


if __name__ == "__main__":
    test_include_filter_replaces_only_matching_attn1_modules()
    test_exclude_filter_keeps_matching_attn1_modules_unreplaced()
    test_filter_env_vars_are_used_when_arguments_are_omitted()
    test_bidirectional_mode_env_selects_forward_backward_or_both()
    print("[OK] Mamba self-attn filter tests passed")
