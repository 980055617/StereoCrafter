from __future__ import annotations
from typing import Optional, Dict, Any

import os
import torch
import torch.nn as nn

from .mamba_spatiotemporal import MambaSpatioTemporalModel
from .mamba_utils import FiLMConditioner


class MambaSpatioTemporalAdapter(nn.Module):
    """
    Diffusers の TransformerSpatioTemporalModel 互換ラッパ。
    UNet の各ブロックが期待する API で呼び出され、内部で MambaSpatioTemporalModel を実行します。

    init シグネチャは TransformerSpatioTemporalModel と揃えています（未使用パラメータも受け取ります）。
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        # Mamba 側オプション
        d_state: int = 256,
        expand: int = 2,
        temporal_chunk_size: int = 64,
        use_mem_eff_path: bool = True,
        keep_spatial_mixer: bool = True,
        num_groups_gn: int = 32,
        spatial_chunk_size: int = 4096,
    ) -> None:
        super().__init__()

        d_model = in_channels
        assert (
            d_model % attention_head_dim == 0
        ), f"in_channels({d_model}) must be divisible by attention_head_dim({attention_head_dim})"

        self.inner = MambaSpatioTemporalModel(
            in_channels=in_channels,
            d_model=d_model,
            cross_attention_dim=cross_attention_dim,
            d_state=d_state,
            headdim=attention_head_dim,
            expand=expand,
            temporal_chunk_size=temporal_chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            num_groups_gn=num_groups_gn,
            keep_spatial_mixer=keep_spatial_mixer,
            spatial_chunk_size=spatial_chunk_size,
        )
        # one-time debug print guard
        self._debug_printed = False

    def forward(
        self,
        hidden_states: torch.Tensor,               # (B*T, C, H, W)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # (B*T, 1, Cctx) など
        image_only_indicator: Optional[torch.Tensor] = None,   # (B, T)
        return_dict: bool = True,
        **kwargs: Dict[str, Any],
    ):
        assert hidden_states.dim() == 4, "expected (B*T, C, H, W)"
        assert image_only_indicator is not None and image_only_indicator.dim() == 2, "need image_only_indicator (B,T)"

        bt, c, h, w = hidden_states.shape
        b, t = image_only_indicator.shape
        assert bt % t == 0 and bt == b * t, f"inconsistent shapes: (BT={bt}) vs (B={b}, T={t})"

        # (B*T,C,H,W) -> (B,C,T,H,W)
        x = hidden_states.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

        # encoder_hidden_states は (B*T, N=1, Cctx) が多いので、各バッチの先頭フレームのみを利用
        enc = encoder_hidden_states
        if enc is not None and enc.dim() >= 2:
            enc = enc[::t]  # (B, 1, Cctx) or (B, Cctx)
            # 必要なら FiLM を遅延初期化（cross_attention_dim を実行時に取得）
            if getattr(self.inner, "cond", None) is None:
                cctx = enc.shape[-1]
                self.inner.cond = FiLMConditioner(ctx_dim=cctx, d_model=self.inner.d_model)
                # inner の他パラメータに合わせてデバイス/精度を合わせる
                ref = next(self.inner.parameters())
                self.inner.cond.to(device=ref.device, dtype=ref.dtype)

        # Diffusers 側から渡ってくる追加引数
        timestep = kwargs.get("timestep", None)
        added_time_ids = kwargs.get("added_time_ids", None)

        # Optional debug (enable by: export MAMBA_DEBUG=1)
        if not self._debug_printed and os.getenv("MAMBA_DEBUG", "0") == "1":
            def _shape(x):
                return tuple(x.shape) if isinstance(x, torch.Tensor) else None
            print("[MambaAdapter] args: ")
            print(f"  hidden_states: bt={bt}, c={c}, h={h}, w={w}")
            print(f"  B={b}, T={t}")
            print(f"  encoder_hidden_states: present={enc is not None}, shape={_shape(enc)}")
            print(f"  image_only_indicator: shape={_shape(image_only_indicator)}")
            print(f"  timestep: present={timestep is not None}, shape={_shape(timestep) if isinstance(timestep, torch.Tensor) else None}, dtype={getattr(timestep,'dtype',None)}")
            print(f"  added_time_ids: present={added_time_ids is not None}, shape={_shape(added_time_ids)}, dtype={getattr(added_time_ids,'dtype',None)}")
            if added_time_ids is not None:
                print("  note: added_time_ids are currently not consumed by Mamba inner block (UNet may use them elsewhere).")
            self._debug_printed = True

        out = self.inner(
            x,
            encoder_hidden_states=enc,
            timestep=timestep,
            image_only_indicator=image_only_indicator,
            return_dict=True,
        )
        y = out["sample"]  # (B,C,T,H,W)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(bt, c, h, w)  # (B*T,C,H,W)

        if not return_dict:
            return (y,)
        return {"sample": y}


def replace_unet_spatiotemporal_transformer_with_mamba(
    unet: nn.Module,
    *,
    d_state: int = 256,
    expand: int = 2,
    temporal_chunk_size: int = 64,
    use_mem_eff_path: bool = True,
    keep_spatial_mixer: bool = True,
    num_groups_gn: int = 32,
    spatial_chunk_size: Optional[int] = None,
) -> int:
    """
    UNetSpatioTemporalConditionModel 配下にある TransformerSpatioTemporalModel を Mamba 版に置換する。

    Returns: 置換したモジュール数
    """
    replaced = 0

    # Environment overrides to help with stability/OOM without code changes
    def _env_bool(name: str) -> Optional[bool]:
        v = os.getenv(name)
        if v is None:
            return None
        v = v.strip().lower()
        if v in ("1", "true", "yes", "on"):  # enable
            return True
        if v in ("0", "false", "no", "off"):  # disable
            return False
        return None

    def _env_int(name: str) -> Optional[int]:
        v = os.getenv(name)
        try:
            return int(v) if v is not None and v != "" else None
        except Exception:
            return None

    # MAMBA_MEM_EFF or MAMBA_USE_MEM_EFF_PATH to override fused/mem‑eff path usage
    env_mem_eff = _env_bool("MAMBA_MEM_EFF")
    if env_mem_eff is None:
        env_mem_eff = _env_bool("MAMBA_USE_MEM_EFF_PATH")
    if env_mem_eff is not None:
        use_mem_eff_path = bool(env_mem_eff)

    # Temporal/spatial chunk sizes can be tuned via env for peak reduction
    env_tchunk = _env_int("MAMBA_TEMPORAL_CHUNK")
    if env_tchunk is not None and env_tchunk > 0:
        temporal_chunk_size = env_tchunk
    env_schunk = _env_int("MAMBA_SPATIAL_CHUNK")
    if env_schunk is not None and env_schunk > 0:
        spatial_chunk_size = env_schunk

    log_enabled = os.getenv("MAMBA_ADAPTER_LOG", "0") == "1" or os.getenv("MAMBA_DEBUG", "0") == "1"

    def _maybe_swap(parent: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}{name}" if prefix else name
            cls_name = child.__class__.__name__
            if cls_name == "TransformerSpatioTemporalModel":
                # 必要パラメータを抽出
                heads = int(getattr(child, "num_attention_heads", 8))
                head_dim = int(getattr(child, "attention_head_dim", 64))
                in_ch = int(getattr(child, "in_channels", None) or getattr(child, "out_channels", None) or 320)
                # cross_attention_dim を推定（UNet config からブロック位置で選択）
                cctx = None
                try:
                    cfg = getattr(unet, "config", None)
                    cad = getattr(cfg, "cross_attention_dim", None)
                    if isinstance(cad, int):
                        cctx = int(cad)
                    elif isinstance(cad, (list, tuple)):
                        # full_name 例: down_blocks.0.attentions.1... / up_blocks.2.attentions.0... / mid_block...
                        if full_name.startswith("down_blocks."):
                            idx = int(full_name.split(".")[1])
                            if 0 <= idx < len(cad):
                                cctx = int(cad[idx])
                        elif full_name.startswith("up_blocks."):
                            idx = int(full_name.split(".")[1])
                            rcad = list(reversed(cad))
                            if 0 <= idx < len(rcad):
                                cctx = int(rcad[idx])
                        elif full_name.startswith("mid_block"):
                            cctx = int(cad[-1])
                except Exception:
                    pass
                if cctx is None:
                    # フォールバック（SVD 既定）
                    cctx = 1024

                adapter = MambaSpatioTemporalAdapter(
                    num_attention_heads=heads,
                    attention_head_dim=head_dim,
                    in_channels=in_ch,
                    out_channels=in_ch,
                    num_layers=len(getattr(child, "transformer_blocks", [])) or 1,
                    cross_attention_dim=cctx,
                    d_state=d_state,
                    expand=expand,
                    temporal_chunk_size=temporal_chunk_size,
                    use_mem_eff_path=use_mem_eff_path,
                    keep_spatial_mixer=keep_spatial_mixer,
                    num_groups_gn=num_groups_gn,
                    spatial_chunk_size=spatial_chunk_size if spatial_chunk_size is not None else 1024,
                )
                # 置換元(child)のデバイス/精度に合わせる
                try:
                    ref_param = next(child.parameters())
                    adapter = adapter.to(device=ref_param.device, dtype=ref_param.dtype)
                except StopIteration:
                    pass

                if log_enabled:
                    # ログ: どこに/どんな設定で挿入されるか
                    dev = getattr(ref_param, "device", None) if 'ref_param' in locals() else None
                    dtype = getattr(ref_param, "dtype", None) if 'ref_param' in locals() else None
                    print("[MambaAdapter][replace] at="
                          f"{full_name} "
                          f"device={dev} dtype={dtype} "
                          f"params={{heads={heads}, head_dim={head_dim}, in_ch={in_ch}, cctx={cctx}, "
                          f"d_state={d_state}, expand={expand}, t_chunk={temporal_chunk_size}, "
                          f"mem_eff={use_mem_eff_path}, keep_spatial={keep_spatial_mixer}, gn_groups={num_groups_gn}, "
                          f"spatial_chunk={spatial_chunk_size if spatial_chunk_size is not None else 1024}}}")
                setattr(parent, name, adapter)
                replaced += 1
            else:
                # 再帰探索。パスを連結
                _maybe_swap(child, prefix=f"{full_name}.")
    _maybe_swap(unet)
    if log_enabled:
        print(f"[MambaAdapter][replace] total_replaced={replaced}")
    return replaced
