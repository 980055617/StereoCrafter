# =============================================
# File: /workspace/stereocraft/blocks/mamba_diffusers_adapter.py
# ---------------------------------------------
# 目的: Diffusers UNet向けMambaアダプタ
# =============================================

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import atexit
import json
import os
from fnmatch import fnmatchcase
from pathlib import Path
import torch
import torch.nn as nn

from .mamba_spatiotemporal import MambaSpatioTemporalModel
from .mamba_temporal import TemporalMamba
from .mamba_utils import FiLMConditioner


_INNER_PROFILE_EVENTS: list[dict[str, Any]] = []
_INNER_PROFILE_REGISTERED = False


def _bidirectional_mode() -> str:
    raw = os.getenv("MAMBA_BIDIRECTIONAL_MODE", "both").strip().lower()
    aliases = {
        "bi": "both",
        "bimamba": "both",
        "bidirectional": "both",
        "forward": "fwd",
        "backward": "bwd",
        "reverse": "bwd",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"both", "fwd", "bwd"}:
        raise ValueError(
            "MAMBA_BIDIRECTIONAL_MODE must be one of both, fwd, bwd "
            f"(got {raw!r})"
        )
    return mode


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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        parsed = int(value) if value is not None and value != "" else int(default)
    except Exception:
        parsed = int(default)
    return parsed


def _inner_profile_path() -> Optional[str]:
    value = os.getenv("MAMBA_INNER_PROFILE_JSON")
    return value if value else None


def _ensure_inner_profile_writer() -> None:
    global _INNER_PROFILE_REGISTERED
    if _INNER_PROFILE_REGISTERED:
        return
    _INNER_PROFILE_REGISTERED = True
    atexit.register(_write_inner_profile)


def _write_inner_profile() -> None:
    path = _inner_profile_path()
    if not path or not _INNER_PROFILE_EVENTS:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grouped: dict[str, dict[str, Any]] = {}
        for event in _INNER_PROFILE_EVENTS:
            module = str(event["module"])
            stage = str(event["stage"])
            module_entry = grouped.setdefault(module, {"module": module, "stages": {}})
            stage_entry = module_entry["stages"].setdefault(
                stage,
                {
                    "calls": 0,
                    "totalMs": 0.0,
                    "maxMs": 0.0,
                    "timingsMs": [],
                    "inputShapes": event.get("inputShapes", []),
                },
            )
            elapsed_ms = float(event["start"].elapsed_time(event["end"]))
            stage_entry["calls"] += 1
            stage_entry["totalMs"] += elapsed_ms
            stage_entry["maxMs"] = max(float(stage_entry["maxMs"]), elapsed_ms)
            stage_entry["timingsMs"].append(elapsed_ms)

        for module_entry in grouped.values():
            for stage_entry in module_entry["stages"].values():
                calls = int(stage_entry["calls"])
                stage_entry["avgMs"] = stage_entry["totalMs"] / max(calls, 1)

        payload = {
            "schema": "master_project.stereocrafter.bimamba_inner_timing.v1",
            "eventCount": len(_INNER_PROFILE_EVENTS),
            "modules": list(grouped.values()),
        }
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[MambaAdapter][inner-profile][warn] failed to write {path}: {exc}")


def _profile_stage(
    module_name: str,
    stage: str,
    input_shapes: list[list[int]],
    fn,
):
    path = _inner_profile_path()
    if not path or not torch.cuda.is_available():
        return fn()
    _ensure_inner_profile_writer()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    _INNER_PROFILE_EVENTS.append(
        {
            "module": module_name,
            "stage": stage,
            "start": start,
            "end": end,
            "inputShapes": input_shapes,
        }
    )
    return result


class LocalDetailResidual1D(nn.Module):
    """Small sequence-local residual branch for recovering local detail.

    The output projection is zero-initialized, so enabling this branch preserves
    the current Mamba behavior at initialization.
    """

    def __init__(self, dim: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=True,
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = x.transpose(1, 2).contiguous()
        x = self.depthwise(x)
        x = x.transpose(1, 2).contiguous()
        x = self.act(x)
        return self.out_proj(x)


class BiMambaSelfAttention(nn.Module):
    """Bidirectional Mamba self-attention replacement."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 256,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 1024,
        use_mem_eff_path: bool = True,
    ) -> None:
        super().__init__()
        self.fwd = TemporalMamba(
            d_model=dim,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
        )
        self.bwd = TemporalMamba(
            d_model=dim,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
        )
        self.supports_time_emb = True
        self._time_embed_dim = None
        self.time_embed_proj = None
        self.local_detail = (
            LocalDetailResidual1D(
                dim,
                kernel_size=_env_int("MAMBA_SELF_ATTN_LOCAL_DETAIL_KERNEL", 3),
            )
            if _env_bool("MAMBA_SELF_ATTN_LOCAL_DETAIL", False)
            else None
        )
        self._profile_name = self.__class__.__name__

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        # Keep signature compatibility; Mamba ignores encoder_hidden_states/attention_mask.
        _ = encoder_hidden_states, attention_mask, kwargs
        mode = _bidirectional_mode()
        module_name = getattr(self, "_profile_name", self.__class__.__name__)
        input_shapes = [list(hidden_states.shape)]

        if mode == "both":
            y_f = _profile_stage(module_name, "fwd", input_shapes, lambda: self.fwd(hidden_states))
            x_rev = _profile_stage(
                module_name,
                "reverse_input",
                input_shapes,
                lambda: torch.flip(hidden_states, dims=[1]).contiguous(),
            )
            y_rev = _profile_stage(module_name, "bwd", input_shapes, lambda: self.bwd(x_rev))
            y_b = _profile_stage(
                module_name,
                "reverse_output",
                input_shapes,
                lambda: torch.flip(y_rev, dims=[1]).contiguous(),
            )
            y = _profile_stage(module_name, "combine", input_shapes, lambda: 0.5 * (y_f + y_b))
        elif mode == "fwd":
            y = _profile_stage(module_name, "fwd", input_shapes, lambda: self.fwd(hidden_states))
        else:
            x_rev = _profile_stage(
                module_name,
                "reverse_input",
                input_shapes,
                lambda: torch.flip(hidden_states, dims=[1]).contiguous(),
            )
            y_rev = _profile_stage(module_name, "bwd", input_shapes, lambda: self.bwd(x_rev))
            y = _profile_stage(
                module_name,
                "reverse_output",
                input_shapes,
                lambda: torch.flip(y_rev, dims=[1]).contiguous(),
            )

        if time_emb is not None:
            def _apply_film() -> torch.Tensor:
                if self.time_embed_proj is None:
                    time_dim = int(time_emb.shape[-1])
                    self._time_embed_dim = time_dim
                    self.time_embed_proj = nn.Linear(time_dim, 2 * y.shape[-1], bias=True)
                    nn.init.zeros_(self.time_embed_proj.weight)
                    nn.init.zeros_(self.time_embed_proj.bias)
                    self.time_embed_proj.to(device=y.device, dtype=y.dtype)
                elif self._time_embed_dim is not None and int(time_emb.shape[-1]) != self._time_embed_dim:
                    raise ValueError(
                        f"time_emb dim changed: expected {self._time_embed_dim}, got {int(time_emb.shape[-1])}"
                    )

                film = self.time_embed_proj(time_emb).to(dtype=y.dtype, device=y.device)
                gamma, beta = film.chunk(2, dim=-1)
                return y * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

            y = _profile_stage(module_name, "film", input_shapes, _apply_film)

        if self.local_detail is not None:
            y = y + _profile_stage(
                module_name,
                "local_detail",
                input_shapes,
                lambda: self.local_detail(y),
            )

        return y


class GatedResidualMambaSelfAttention(BiMambaSelfAttention):
    """Train with a frozen attention reference while annealing toward Mamba-only."""

    def __init__(
        self,
        dim: int,
        origin_attn: nn.Module,
        *,
        d_state: int = 256,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 1024,
        use_mem_eff_path: bool = True,
        initial_gate: float = 0.0,
    ) -> None:
        super().__init__(
            dim,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
        )
        self.origin_attn = origin_attn
        for param in self.origin_attn.parameters():
            param.requires_grad_(False)
        self.origin_attn.eval()
        self.register_buffer("mamba_gate", torch.tensor(float(initial_gate)), persistent=True)
        self.reference_disabled = False
        self.origin_feature_distill_enabled = False
        self.origin_feature_distill_loss: Optional[torch.Tensor] = None

    def set_mamba_gate(self, value: float, *, disable_reference: bool = False) -> None:
        value = max(0.0, min(1.0, float(value)))
        self.mamba_gate.fill_(value)
        self.reference_disabled = bool(disable_reference or value >= 1.0)

    def set_origin_feature_distill(self, enabled: bool) -> None:
        self.origin_feature_distill_enabled = bool(enabled)
        self.origin_feature_distill_loss = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.origin_attn.eval()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        mamba_y = super().forward(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            time_emb=time_emb,
            **kwargs,
        )
        gate = float(self.mamba_gate.detach().float().item())
        self.origin_feature_distill_loss = None
        needs_reference = self.origin_feature_distill_enabled or not (self.reference_disabled or gate >= 1.0)
        if not needs_reference:
            return mamba_y
        with torch.no_grad():
            ref_y = self.origin_attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
        if self.origin_feature_distill_enabled:
            self.origin_feature_distill_loss = (mamba_y.float() - ref_y.detach().float()).pow(2).mean()
        if self.reference_disabled or gate >= 1.0:
            return mamba_y
        if gate <= 0.0:
            return ref_y
        return ref_y + self.mamba_gate.to(dtype=mamba_y.dtype, device=mamba_y.device) * (mamba_y - ref_y)


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
        if cross_attention_dim is not None:
            self.inner.cond = FiLMConditioner(
                ctx_dim=cross_attention_dim,
                d_model=self.inner.d_model,
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


def replace_unet_spatiotemporal_self_attn_with_mamba(
    unet: nn.Module,
    *,
    d_state: int = 256,
    expand: int = 2,
    chunk_size: int = 1024,
    use_mem_eff_path: bool = True,
    include_patterns: Optional[Sequence[str] | str] = None,
    exclude_patterns: Optional[Sequence[str] | str] = None,
) -> int:
    """
    Replace only the self-attn (attn1) inside TransformerSpatioTemporalModel blocks with Mamba.

    Cross-attn (attn2) stays intact.
    """
    replaced = 0

    def _env_bool(name: str) -> Optional[bool]:
        v = os.getenv(name)
        if v is None:
            return None
        v = v.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        return None

    def _env_int(name: str) -> Optional[int]:
        v = os.getenv(name)
        try:
            return int(v) if v is not None and v != "" else None
        except Exception:
            return None

    env_mem_eff = _env_bool("MAMBA_MEM_EFF")
    if env_mem_eff is None:
        env_mem_eff = _env_bool("MAMBA_USE_MEM_EFF_PATH")
    if env_mem_eff is not None:
        use_mem_eff_path = bool(env_mem_eff)

    env_d_state = _env_int("MAMBA_SELF_ATTN_D_STATE")
    if env_d_state is not None and env_d_state > 0:
        d_state = env_d_state

    env_expand = _env_int("MAMBA_SELF_ATTN_EXPAND")
    if env_expand is not None and env_expand > 0:
        expand = env_expand

    env_chunk = _env_int("MAMBA_SELF_ATTN_CHUNK")
    if env_chunk is not None and env_chunk > 0:
        chunk_size = env_chunk

    def _split_patterns(value: Optional[Sequence[str] | str]) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw = value.split(",")
        else:
            raw = []
            for item in value:
                raw.extend(str(item).split(","))
        return [item.strip() for item in raw if item and item.strip()]

    def _env_patterns(name: str) -> list[str]:
        return _split_patterns(os.getenv(name))

    include_filter = _split_patterns(include_patterns) or _env_patterns("MAMBA_SELF_ATTN_INCLUDE")
    exclude_filter = _split_patterns(exclude_patterns) or _env_patterns("MAMBA_SELF_ATTN_EXCLUDE")

    def _matches_pattern(path: str, pattern: str) -> bool:
        if any(ch in pattern for ch in "*?[]"):
            return fnmatchcase(path, pattern)
        return path == pattern or path.startswith(f"{pattern}.")

    def _should_replace(path: str) -> bool:
        if include_filter and not any(_matches_pattern(path, pattern) for pattern in include_filter):
            return False
        if exclude_filter and any(_matches_pattern(path, pattern) for pattern in exclude_filter):
            return False
        return True

    replacement_mode = os.getenv("MAMBA_SELF_ATTN_REPLACEMENT", "mamba").strip().lower()
    use_gated_residual = replacement_mode in {"gated", "gated_residual", "residual_gated"}
    initial_gate = _env_int("MAMBA_SELF_ATTN_INITIAL_GATE")
    initial_gate_float = 0.0 if initial_gate is None else float(initial_gate)
    initial_gate_raw = os.getenv("MAMBA_SELF_ATTN_INITIAL_GATE")
    if initial_gate_raw is not None:
        try:
            initial_gate_float = float(initial_gate_raw)
        except ValueError:
            initial_gate_float = 0.0

    log_enabled = os.getenv("MAMBA_ADAPTER_LOG", "0") == "1" or os.getenv("MAMBA_DEBUG", "0") == "1"

    def _swap_attn1(block: nn.Module, prefix: str) -> bool:
        nonlocal replaced
        attn1_path = f"{prefix}.attn1"
        if not _should_replace(attn1_path):
            if log_enabled:
                print(f"[MambaAdapter][self-attn][filter-skip] {attn1_path}")
            return False
        attn1 = getattr(block, "attn1", None)
        if attn1 is None:
            return False

        dim = getattr(attn1, "query_dim", None)
        if dim is None:
            norm = getattr(block, "norm1", None)
            if norm is not None and hasattr(norm, "normalized_shape"):
                shape = norm.normalized_shape
                dim = int(shape[0]) if isinstance(shape, (list, tuple)) else int(shape)
        if dim is None:
            return False

        adapter_kwargs = dict(
            d_state=d_state,
            headdim=getattr(attn1, "dim_head", 64),
            expand=expand,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
        )
        if use_gated_residual:
            adapter = GatedResidualMambaSelfAttention(
                dim,
                origin_attn=attn1,
                initial_gate=initial_gate_float,
                **adapter_kwargs,
            )
        else:
            adapter = BiMambaSelfAttention(dim, **adapter_kwargs)
        adapter._profile_name = attn1_path
        try:
            ref_param = next(attn1.parameters())
            adapter = adapter.to(device=ref_param.device, dtype=ref_param.dtype)
        except StopIteration:
            pass

        setattr(block, "attn1", adapter)
        replaced += 1
        if log_enabled:
            dev = getattr(ref_param, "device", None) if "ref_param" in locals() else None
            dtype = getattr(ref_param, "dtype", None) if "ref_param" in locals() else None
            print(
                "[MambaAdapter][self-attn] at="
                f"{prefix}.attn1 dim={dim} device={dev} dtype={dtype} "
                f"d_state={d_state} expand={expand} chunk={chunk_size} mem_eff={use_mem_eff_path} "
                f"mode={'gated_residual' if use_gated_residual else 'mamba'}"
            )
        return True

    def _maybe_swap(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}{name}" if prefix else name
            if child.__class__.__name__ == "TransformerSpatioTemporalModel":
                replaced_in_block = 0
                for i, block in enumerate(getattr(child, "transformer_blocks", [])):
                    if _swap_attn1(block, f"{full_name}.transformer_blocks.{i}"):
                        replaced_in_block += 1
                for i, _block in enumerate(getattr(child, "temporal_transformer_blocks", [])):
                    if log_enabled:
                        print(
                            "[MambaAdapter][self-attn][skip] "
                            f"{full_name}.temporal_transformer_blocks.{i}.attn1"
                        )
                if replaced_in_block:
                    setattr(child, "_mamba_self_attn", True)
            else:
                _maybe_swap(child, prefix=f"{full_name}.")

    _maybe_swap(unet)
    if log_enabled:
        print(f"[MambaAdapter][self-attn] total_replaced={replaced}")
    return replaced


def set_gated_mamba_gate(root: nn.Module, value: float, *, disable_reference: bool = False) -> int:
    """Set the gate on all gated residual Mamba attention modules."""
    updated = 0
    for module in root.modules():
        if isinstance(module, GatedResidualMambaSelfAttention):
            module.set_mamba_gate(value, disable_reference=disable_reference)
            updated += 1
    return updated


def set_origin_feature_distill(root: nn.Module, enabled: bool) -> int:
    """Enable/disable origin-attention feature distillation on gated modules."""
    updated = 0
    for module in root.modules():
        if isinstance(module, GatedResidualMambaSelfAttention):
            module.set_origin_feature_distill(enabled)
            updated += 1
    return updated


def collect_origin_feature_distill_loss(root: nn.Module) -> tuple[Optional[torch.Tensor], int]:
    """Return the mean feature distillation loss from the latest forward pass."""
    losses: list[torch.Tensor] = []
    for module in root.modules():
        if isinstance(module, GatedResidualMambaSelfAttention):
            loss = getattr(module, "origin_feature_distill_loss", None)
            if isinstance(loss, torch.Tensor):
                losses.append(loss)
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), len(losses)


def _resolve_module_by_path(root: nn.Module, path: str) -> Optional[nn.Module]:
    cur: object = root
    for token in path.split("."):
        if token.isdigit():
            idx = int(token)
            if isinstance(cur, (nn.ModuleList, list, tuple)) and 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            if not hasattr(cur, token):
                return None
            cur = getattr(cur, token)
    return cur if isinstance(cur, nn.Module) else None


def materialize_mamba_time_embed_proj_from_state_dict(unet: nn.Module, state_dict: Dict[str, torch.Tensor]) -> int:
    """Instantiate lazy `attn1.time_embed_proj` modules so checkpoint keys can load.

    BiMambaSelfAttention creates `time_embed_proj` lazily on first forward, which
    causes checkpoint keys to be treated as `unexpected` at load time. This helper
    creates those linear layers from state_dict shapes before `load_state_dict`.
    """
    created = 0
    suffix = ".time_embed_proj.weight"
    for key, weight in state_dict.items():
        if not key.endswith(suffix):
            continue
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        module_path = key[: -len(suffix)]
        module = _resolve_module_by_path(unet, module_path)
        if module is None:
            continue
        if not hasattr(module, "time_embed_proj"):
            continue
        if getattr(module, "time_embed_proj", None) is not None:
            continue
        in_dim = int(weight.shape[1])
        out_dim = int(weight.shape[0])
        proj = nn.Linear(in_dim, out_dim, bias=True)
        try:
            ref = next(module.parameters())
            proj = proj.to(device=ref.device, dtype=ref.dtype)
        except StopIteration:
            pass
        module.time_embed_proj = proj
        if hasattr(module, "_time_embed_dim"):
            module._time_embed_dim = in_dim
        created += 1
    return created
