# blocks/mamba_spatiotemporal.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from mamba_ssm import Mamba2

class MambaSpatioTemporalModel(nn.Module):
    """
    Drop-in 置換用モジュール。
    - 入出力: (B, C, T, H, W) を基本とするが、4D (B, C, H, W) も T=1 として受け付ける。
    - 追加引数は互換性のため受けるが未使用（Mid は従来 Transformer を保持する方針）。
    """
    def __init__(
        self,
        in_channels: int,
        d_model: Optional[int] = None,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model or in_channels

        self.proj_in  = nn.Conv2d(in_channels, self.d_model, kernel_size=1)
        self.proj_out = nn.Conv2d(self.d_model, in_channels, kernel_size=1)

        with torch.no_grad():
            if self.d_model == in_channels:
                # 1x1 conv を恒等に近い初期化
                w = self.proj_in.weight.data
                self.proj_in.weight.zero_()
                for i in range(min(w.shape[0], w.shape[1])):
                    self.proj_in.weight.data[i, i, 0, 0] = 1.0
                self.proj_in.bias.zero_()

                w = self.proj_out.weight.data
                self.proj_out.weight.zero_()
                for i in range(min(w.shape[0], w.shape[1])):
                    self.proj_out.weight.data[i, i, 0, 0] = 1.0
                self.proj_out.bias.zero_()
            else:
                nn.init.kaiming_uniform_(self.proj_in.weight, a=math.sqrt(5))
                nn.init.zeros_(self.proj_in.bias)
                nn.init.zeros_(self.proj_out.weight)
                nn.init.zeros_(self.proj_out.bias)

        self.core = Mamba2(
            d_model=self.d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
        )

        # 将来の stateful 推論用（今回は未使用）
        self._state: Optional[Tuple[Any, Any]] = None

    def reset_state(self):
        self._state = None

    def forward(
        self,
        hidden_states: torch.Tensor,                # (B*C,T,H,W) or (B,T,C,H,W) or (B,C,H,W)
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[dict] = None,
        temb: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        **_: Any
    ) -> torch.Tensor:
        # 入力次元を記録（返却時の形を合わせる）
        _in_dim = hidden_states.dim()

        # --- B*T→(B,C,T,H,W) 復元のためのフラグとサイズ保持 ---
        flatten_back = False
        B = T = None

        if hidden_states.dim() == 4:
            # UNet は [B*T, C, H, W] を渡してくる。T は image_only_indicator から復元できることが多い
            if num_frames is None and image_only_indicator is not None:
                num_frames = int(image_only_indicator.shape[1])
            if num_frames is not None and num_frames > 1:
                T = int(num_frames)
                BT, C, H, W = hidden_states.shape
                if BT % T != 0:
                    raise ValueError(f"num_frames={T} に一致しない先頭次元: {BT}")
                B = BT // T
                # -> (B, T, C, H, W) -> (B, C, T, H, W)
                hidden_states = hidden_states.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
                flatten_back = True
            else:
                # T が分からない/1 の場合は 4D のまま扱う（後段互換のため 4D→4D で返す）
                pass
        elif hidden_states.dim() == 5:
            # (B, C, T, H, W) か (B, T, C, H, W) を受ける
            B_, A, B2, H, W = hidden_states.shape
            if A == self.in_channels:          # (B, C, T, H, W)
                pass
            elif B2 == self.in_channels:       # (B, T, C, H, W) -> (B, C, T, H, W)
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
            else:
                raise ValueError(f"in_channels={self.in_channels} と一致する軸が見つかりません: {tuple(hidden_states.shape)}")
        else:
            raise ValueError(f"Expected 4D or 5D, got {tuple(hidden_states.shape)}")

        # 以降の実処理は 2パスに分ける:
        # (A) 5D: (B,C,T,H,W) を時系列として扱う
        # (B) 4D: (BT,C,H,W) は T が不明なので T=1 とみなし、時系列長1として扱う
        if hidden_states.dim() == 5:
            Bc, Cc, Tc, Hc, Wc = hidden_states.shape
        else:
            # 4D 入力（T 不明）: T=1 として扱う
            BT, Cc, Hc, Wc = hidden_states.shape
            Bc, Tc = BT, 1
            hidden_states = hidden_states.view(Bc, Cc, Tc, Hc, Wc)

        # (B,C,T,H,W) -> (B*T,C,H,W) -> proj_in
        x = hidden_states.permute(0, 2, 1, 3, 4).contiguous().view(Bc * Tc, Cc, Hc, Wc)
        x = self.proj_in(x)  # (B*T, d_model, H, W)

        # -> (B*H*W, T, d_model)
        x = x.view(Bc, Tc, self.d_model, Hc, Wc).permute(0, 3, 4, 1, 2).contiguous().view(Bc * Hc * Wc, Tc, self.d_model)

        y = self.core(x)  # (B*H*W, T, d_model)

        # 逆変換 (B,T,d_model,H,W) -> (B,C,T,H,W)
        y = y.view(Bc, Hc, Wc, Tc, self.d_model).permute(0, 3, 4, 1, 2).contiguous()
        y = y.view(Bc * Tc, self.d_model, Hc, Wc)
        y = self.proj_out(y)
        y = y.view(Bc, Tc, Cc, Hc, Wc).permute(0, 2, 1, 3, 4).contiguous()

        # --- 呼び出し元契約に合わせて戻す ---
        if flatten_back:
            y = y.permute(0, 2, 1, 3, 4).contiguous().view(B * T, Cc, Hc, Wc)
        else:
            if _in_dim == 4:
                # 入力が 4D なら 4D で返す（Diffusers の期待に合わせる）
                y = y.view(Bc * Tc, Cc, Hc, Wc)
            # 入力が 5D なら 5D のまま返す

        # Diffusers の TransformerSpatioTemporalModel に合わせてタプル返却
        return (y,)