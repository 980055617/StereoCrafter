import math
import torch
import torch.nn as nn
from torch import Tensor
from math import gcd

from mamba_ssm import Mamba2

class TemporalMambaAdapter(nn.Module):
    """
    UNet 中間テンソル x:[B*F, C, H, W] に対し、F（フレーム）方向のみ Mamba2 で系列混合。
    - Mamba の d_model を「8整列要件」を満たす倍数に動的パディング（1x1 Convで拡張/縮小）
    - reshape 後は .contiguous() を強制
    - 残差: y = x + gamma * f(x), gamma は 0 初期化（恒等スタート）
    """
    def __init__(self, channels: int, d_state: int = 128, headdim: int = 64, num_groups: int = 32):
        super().__init__()
        self.channels = channels
        self.d_state = d_state          # ← 保存
        self.headdim = headdim          # ← 保存

        self.norm     = nn.GroupNorm(num_groups, channels)
        self.proj_in  = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # d_model は F に依存して動的に決めるため lazy init
        self._mamba: Mamba2 = None      # type: ignore

        # d_model が channels と一致しない場合に使う1x1の昇降次元
        self.expand   = nn.Identity()
        self.collapse = nn.Identity()

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gamma    = nn.Parameter(torch.zeros(1))  # 恒等開始

        # 実行時設定
        self._B = None
        self._F = None
        self._expanded_dim: int = channels

    @torch.no_grad()
    def set_batch_frames(self, B: int, F: int):
        self._B, self._F = B, F

    def _get_BF(self, x: Tensor):
        if self._B is None or self._F is None:
            raise RuntimeError("TemporalMambaAdapter: B,F are not set. Call set_batch_frames(B,F) before UNet forward.")
        return self._B, self._F

    def _min_aligned_dmodel(self, c_min: int, headdim: int, d_state: int, ngroups: int = 1) -> int:
        step = max(headdim // 2, 1)  # ensure (2*d_model) % headdim == 0
        m = ((c_min + step - 1) // step) * step
        while True:
            if (2 * m) % headdim != 0:
                m += step
                continue
            nheads = (2 * m) // headdim
            d_in_proj = 4 * m + 2 * ngroups * d_state + nheads
            if d_in_proj % 8 == 0:
                return m
            m += step
        
    def _ensure_mamba_ready(self, device, dtype, F: int):
        """
        Mamba の d_model を「causal_conv1d の stride×8 整列要件」を満たすよう決定。
        条件: rearrange('b s d -> b d s') 後に stride(0)=F*d, stride(2)=d が 8 の倍数。
        ⇒ d を multiple_of = 8 // gcd(F, 8) の倍数にすればOK。加えて安全側で 8 の倍数に寄せる。
        """
        expanded = self._min_aligned_dmodel(self.channels, self.headdim, self.d_state, ngroups=1)


        if self._mamba is not None and expanded == self._expanded_dim:
            # 既存インスタンスを使い回しつつ device/dtype を最新に
            self._mamba.to(device=device, dtype=dtype)
            if isinstance(self.expand, nn.Conv2d):
                self.expand.to(device=device, dtype=dtype)
            if isinstance(self.collapse, nn.Conv2d):
                self.collapse.to(device=device, dtype=dtype)
            return

        # expand / collapse を再構築（必要なときのみ）
        if expanded != self.channels:
            self.expand   = nn.Conv2d(self.channels, expanded, kernel_size=1, bias=True).to(device=device, dtype=dtype)
            self.collapse = nn.Conv2d(expanded, self.channels, kernel_size=1, bias=True).to(device=device, dtype=dtype)
        else:
            self.expand   = nn.Identity()
            self.collapse = nn.Identity()

        # Mamba を再構築（d_model=expanded）
        self._mamba = Mamba2(d_model=expanded, d_state=self.d_state, headdim=self.headdim)
        self._mamba.to(device=device, dtype=dtype)

        self._expanded_dim = expanded

    def forward(self, x: Tensor) -> Tensor:
        # x: [B*F, C, H, W]
        bf, C, H, W = x.shape
        B, F = self._get_BF(x)
        assert bf == B * F, f"shape mismatch: got {bf}, expected {B*F}"

        dev, dt = x.device, x.dtype
        self._ensure_mamba_ready(dev, dt, F)

        r = self.norm(x)
        r = self.proj_in(r)

        # 1) 必要ならチャネル拡張（d_model 整列）
        r = self.expand(r)                          # [B*F, C', H, W]
        Cexp = r.shape[1]

        # 2) 時間系列化（contiguous 保証）
        r = r.view(B, F, Cexp, H, W).permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, F, C']
        r = r.view(B * H * W, F, Cexp).contiguous()                        # [B*H*W, F, C']

        # 3) Mamba 本体（O(F)）
        r = self._mamba(r)

        # 4) 形状を戻す（contiguous 保証）
        r = r.view(B, H, W, F, Cexp).permute(0, 3, 4, 1, 2).contiguous().view(B * F, Cexp, H, W)

        # 5) チャネル縮小（元の C に戻す）
        r = self.collapse(r)

        r = self.proj_out(r)
        return x + self.gamma * r