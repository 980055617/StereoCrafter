# =============================================
# File: blocks/mamba_spatiotemporal.py
# ---------------------------------------------
# 目的: A〜E をすべて反映した本体
#   - A 条件づけ (FiLM)
#   - B 時間埋め込み + AlphaBlender
#   - C 二系統 (空間/時間)
#   - D 入力正規化 + 出力残差
#   - E return_dict 互換
# 依存: mamba_utils, mamba_spatial, mamba_temporal
# =============================================
from __future__ import annotations
from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.resnet import AlphaBlender

from .mamba_utils import  FiLMConditioner, apply_film
from .mamba_spatial import SpatialMixer
from .mamba_temporal import TemporalMamba


class MambaSpatioTemporalModel(nn.Module):
    """
    TransformerSpatioTemporalModel 互換を目標にした Mamba 版。
    入出力: (B, C, T, H, W)

    Args:
        in_channels: UNet側から入る C。
        d_model: Mamba 内部次元。未指定なら in_channels。
        cross_attention_dim: encoder_hidden_states のチャネル数 (A)
        num_groups_gn: GroupNorm のグループ数 (D)
        keep_spatial_mixer: True なら空間ミキサを使う (C)
        temporal_chunk_size: 長い T に対する安定化 (E 実運用ヒント)
    """
    def __init__(
        self,
        in_channels: int,
        d_model: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        temporal_chunk_size: int = 64,
        use_mem_eff_path: bool = True,
        num_groups_gn: int = 32,
        keep_spatial_mixer: bool = True,
        spatial_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model or in_channels
        assert self.d_model % headdim == 0, f"d_model={self.d_model} must be divisible by headdim={headdim}"

        # D: 入り口正規化 & 1x1 投影（C を d_model に合わせる）
        self.norm_in = nn.GroupNorm(num_groups=num_groups_gn, num_channels=in_channels)
        self.in_proj = nn.Conv3d(in_channels, self.d_model, kernel_size=1)

        # B: timestep embedding
        self.time_proj = Timesteps(self.d_model, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(in_channels=self.d_model, time_embed_dim=self.d_model*4, out_dim=self.d_model)

        # A: 条件づけ (FiLM)
        self.cond = FiLMConditioner(cross_attention_dim, self.d_model) if (cross_attention_dim is not None) else None

        # C: 空間/時間の二系統
        self.keep_spatial_mixer = keep_spatial_mixer
        if keep_spatial_mixer:
            self.spatial = SpatialMixer(self.d_model)
        self.temporal = TemporalMamba(
            d_model=self.d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=temporal_chunk_size,
            use_mem_eff_path=use_mem_eff_path,
        )
        # 大きな空間次元(H*W)に対して、(B*H*W) 次元でチャンク実行するための上限
        self.spatial_chunk_size = spatial_chunk_size

        # B: ブレンド
        self.blender = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 出口: 1x1 で元チャネルに戻す + D: 残差を足す
        self.out_proj = nn.Conv3d(self.d_model, in_channels, kernel_size=1)

    def _apply_time_embed(self, x, timesteps):
        # x: (B,C,T,H,W)
        B, C, T, H, W = x.shape

        # tベクトルを用意（BかB*Tを受け取り、足りなければ0..T-1で埋める）
        if timesteps is None:
            # Diffusers' spatiotemporal blocks don't pass diffusion timestep; use frame indices like SVD.
            t = torch.arange(T, device=x.device).repeat(B)  # (B*T,)
        else:
            t_in = timesteps
            if not isinstance(t_in, torch.Tensor):
                t_in = torch.tensor(t_in, device=x.device)
            elif t_in.device != x.device:
                t_in = t_in.to(x.device)
            if t_in.dim() == 0:
                t_in = t_in[None]
            if t_in.dim() == 1:
                if t_in.numel() == B * T:
                    t = t_in
                elif t_in.numel() == B:
                    # ★ ここで各バッチのtimestepをフレーム数Tぶんに拡張
                    t = t_in.repeat_interleave(T)     # (B*T,)
                else:
                    t = torch.arange(T, device=x.device) \
                            .repeat(B)                 # (B*T,)
            else:
                t = torch.arange(T, device=x.device) \
                        .repeat(B)                 # (B*T,)

        # per-frameで埋め込み → (B*T, C) → (B,C,T,1,1) に整形して加算
        temb = self.time_proj(t)                  # (B*T, C) ここはfp32になりがち
        # TimestepEmbeddingの重みdtypeに合わせる（fp16運用/AMPでも衝突回避）
        target_dtype = self.time_embed.linear_1.weight.dtype
        temb = temb.to(target_dtype)
        temb = self.time_embed(temb)              # (B*T, C)
        temb = temb.view(B, T, C).permute(0,2,1).contiguous()[:, :, :, None, None]
        # Optional one-time debug
        if getattr(self, "_dbg_time_once", False) is False and os.getenv("MAMBA_DEBUG", "0") == "1":
            print("[MambaInner] time-embed applied:")
            print(f"  x: {(B, C, T, H, W)}  t: shape={(t.shape if isinstance(t, torch.Tensor) else None)}, dtype={getattr(t,'dtype',None)}")
            print(f"  temb: shape={temb.shape}, dtype={temb.dtype}")
            self._dbg_time_once = True
        return x + temb

    def _temporal_path(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,H,W) with C=d_model
        B, C, T, H, W = x.shape
        # (B*H*W, T, C)
        x_seq = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, T, C)
        # 空間次元をチャンクしてメモリを抑制
        if self.spatial_chunk_size is not None and (B * H * W) > self.spatial_chunk_size:
            chunks = []
            for i in range(0, B * H * W, self.spatial_chunk_size):
                part = x_seq[i : i + self.spatial_chunk_size]
                chunks.append(self.temporal(part))
            y_seq = torch.cat(chunks, dim=0)
        else:
            y_seq = self.temporal(x_seq)
        y = y_seq.view(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()  # (B,C,T,H,W)
        return y

    def _spatial_path(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,H,W) with C=d_model
        if not self.keep_spatial_mixer:
            return x
        B, C, T, H, W = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        y_2d = self.spatial(x_2d)
        y = y_2d.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        return y

    def forward(
        self,
        hidden_states: torch.Tensor,             # (B, C, T, H, W)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # (B, N, Cctx) or (B, Cctx)
        timestep: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **_: dict,
    ):
        assert hidden_states.dim() == 5, "expected (B,C,T,H,W)"
        dtype = hidden_states.dtype
        residual = hidden_states

        # D: 入力正規化 & 1x1 投影
        x = self.norm_in(hidden_states)
        x = self.in_proj(x)

        # B: 時間埋め込み加算
        x = self._apply_time_embed(x, timestep)

        # A: FiLM 条件注入（前段で適用）
        if self.cond is not None:
            gamma, beta = self.cond(encoder_hidden_states)
            # (B,C,T,H,W) に対してチャネル次元に適用
            x = apply_film(x, gamma, beta)

        # C: 二系統
        y_temporal = self._temporal_path(x)
        y_spatial = self._spatial_path(x)

        # B: ブレンディング
        B, _, T, _, _ = y_spatial.shape
        if image_only_indicator is None:
            image_only_indicator = torch.zeros(B, T, dtype=y_spatial.dtype, device=y_spatial.device)
        else:
            if image_only_indicator.dim() == 1 and image_only_indicator.shape[0] == B:
                image_only_indicator = image_only_indicator[:, None].expand(B, T)
            elif image_only_indicator.dim() == 2 and image_only_indicator.shape[0] == B and image_only_indicator.shape[1] == 1:
                image_only_indicator = image_only_indicator.expand(B, T)
            elif not (image_only_indicator.dim() == 2 and image_only_indicator.shape == (B, T)):
                raise ValueError(
                    f"image_only_indicator must be (B,), (B,1) or (B,T). Got {tuple(image_only_indicator.shape)} for (B,T)=({B},{T})"
                )
        y = self.blender(
            x_spatial=y_spatial,
            x_temporal=y_temporal,
            image_only_indicator=image_only_indicator,
        )

        # 出口 & D: 残差
        y = self.out_proj(y)
        y = y + residual

        if return_dict:
            return {"sample": y}
        return (y,)
