import csv
import gc
import glob
import inspect
import json
import logging
import math
import os
import random
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Sequence, Union

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch.library.impl_abstract.*register_fake.*",
)

logger = logging.getLogger(__name__)


def _normalize_stage_overrides(raw: Any) -> dict[int, dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, (list, tuple)):
        if len(raw) != 3:
            raise ValueError("stage_overrides as a list must contain exactly 3 entries.")
        items = enumerate(raw, start=1)
    elif isinstance(raw, dict):
        items = raw.items()
    else:
        raise ValueError("stage_overrides must be a dict keyed by stage number, or a 3-entry list.")

    normalized: dict[int, dict[str, Any]] = {}
    for key, value in items:
        if value in (None, {}):
            continue
        try:
            stage_idx = int(key)
        except (TypeError, ValueError) as err:
            raise ValueError(f"stage_overrides key must be 1, 2, or 3; got {key!r}") from err
        if stage_idx < 1 or stage_idx > 3:
            raise ValueError(f"stage_overrides key must be 1, 2, or 3; got {stage_idx}")
        if not isinstance(value, dict):
            raise ValueError(f"stage_overrides[{stage_idx}] must be an object.")
        normalized[stage_idx] = dict(value)
    return normalized


def _merge_stage_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


import torch
import torch.nn.functional as F
# ---- force reentrant checkpoint (must be BEFORE diffusers imports) ----
import sys
import torch.utils.checkpoint as _cp

_ORIG_CP = _cp.checkpoint
_FORCE_REENTRANT_CP = False

def enable_force_reentrant_checkpoint(enabled: bool) -> None:
    global _FORCE_REENTRANT_CP
    _FORCE_REENTRANT_CP = bool(enabled)

def _patched_checkpoint(function, *args, **kwargs):
    if _FORCE_REENTRANT_CP:
        kwargs["use_reentrant"] = True
        kwargs.pop("determinism_check", None)
    return _ORIG_CP(function, *args, **kwargs)

_cp.checkpoint = _patched_checkpoint

def _patch_modules_holding_checkpoint_symbol() -> None:
    for m in list(sys.modules.values()):
        if m is None:
            continue
        if hasattr(m, "checkpoint") and getattr(m, "checkpoint") is _ORIG_CP:
            setattr(m, "checkpoint", _cp.checkpoint)

# optional: if private non-reentrant generator exists, redirect to reentrant
try:
    _ORIG_NRR = _cp._checkpoint_without_reentrant_generator  # type: ignore[attr-defined]
    def _patched_nrr_gen(function, *args, **kwargs):
        return _cp.checkpoint(function, *args, use_reentrant=True)
    _cp._checkpoint_without_reentrant_generator = _patched_nrr_gen  # type: ignore[attr-defined]
except Exception:
    pass
# -----------------------------------------------------------------------

def _format_cuda_stats(device: torch.device | None = None) -> str:
    if not torch.cuda.is_available():
        return "cuda:unavailable"
    indices = list(range(torch.cuda.device_count()))
    if device is not None and device.type == "cuda":
        idx = 0 if device.index is None else device.index
        indices = [idx]
    parts = []
    for idx in indices:
        try:
            alloc = torch.cuda.memory_allocated(idx) / (1024**2)
            rsv = torch.cuda.memory_reserved(idx) / (1024**2)
            peak = torch.cuda.max_memory_allocated(idx) / (1024**2)
            parts.append(f"cuda:{idx} alloc={alloc:.1f}MiB reserved={rsv:.1f}MiB peak_alloc={peak:.1f}MiB")
        except Exception as err:
            parts.append(f"cuda:{idx} stats_error={err}")
    return " | ".join(parts) if parts else "cuda:unknown"


def log_cuda_memory_stats(tag: str, *, device: torch.device | None = None, summary: bool = False) -> None:
    if not torch.cuda.is_available():
        logger.info("[CUDA:%s] cuda not available", tag)
        return
    try:
        logger.info("[CUDA:%s] %s", tag, _format_cuda_stats(device))
    except Exception as err:
        logger.warning("[CUDA:%s] failed to query cuda stats: %s", tag, err)
    if summary:
        try:
            summary_device = device if device is not None else torch.device("cuda")
            logger.info("[CUDA:%s] summary (abbrev):\n%s", tag, torch.cuda.memory_summary(summary_device, abbreviated=True))
        except Exception as err:
            logger.warning("[CUDA:%s] failed to get memory summary: %s", tag, err)


def cleanup_cuda(tag: str, *objs: Any) -> None:
    try:
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(idx)
                except Exception:
                    pass
            logger.info("[CLEANUP:before:%s] %s", tag, _format_cuda_stats())
    except Exception as err:
        logger.warning("[CLEANUP:before:%s] failed to query cuda stats: %s", tag, err)

    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    try:
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(idx)
                except Exception:
                    pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                for idx in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(idx)
            except Exception:
                pass
            for idx in range(torch.cuda.device_count()):
                try:
                    torch.cuda.synchronize(idx)
                except Exception:
                    pass
            logger.info("[CLEANUP:after:%s] %s", tag, _format_cuda_stats())
    except Exception as err:
        logger.warning("[CLEANUP:after:%s] failed: %s", tag, err)

from fire import Fire

from blocks.mamba_diffusers_adapter import BiMambaSelfAttention, GatedResidualMambaSelfAttention, MambaSpatioTemporalAdapter
from blocks.mamba_diffusers_adapter import materialize_mamba_time_embed_proj_from_state_dict
from blocks.mamba_diffusers_adapter import set_gated_mamba_gate
from utils.config_utils import load_json_config
from utils.training_batches import prepare_batches, estimate_num_chunks
from utils.training_env import get_compute_device, set_global_seed, setup_interrupt_handler
from utils.training_log_utils import (
    init_csv_log,
    resolve_run_save_dir,
    select_log_path,
    trim_train_log,
    trim_val_log,
    write_run_config_snapshot,
)
from utils.training_pipeline import (
    apply_mamba_runtime_flags,
    configure_unet_memory_features,
    enable_vae_memory_helpers,
    load_inpainting_pipeline,
    maybe_shard_unet,
)
from utils.training_precision import resolve_precision
from utils.logging_utils import TrainingProgressPrinter, ensure_logging_configured, log_vram_usage
from diffusers.schedulers import DDPMScheduler, EulerDiscreteScheduler
from utils.diffusers_mamba_time_patch import apply_mamba_time_patch

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None

_CACHED_DS_ACCELERATOR = None
_CACHED_DS_ACCELERATOR_CFG: dict[str, Any] = {}

apply_mamba_time_patch()


def _train_main(
    pre_trained_path: str,
    unet_path: str,
    train_glob: str,
    save_dir: str,
    *,
    stage_name: str,
    stage_h: int,
    stage_w: int,
    stage_epochs: int,
    stage_lr: float,
    stage_idx: int,
    frames_chunk: int = 14,
    overlap: int = 3,
    use_prev_target_overlap: bool = True,
    fps_condition: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.0,
    vae_encode_chunk_size: int = 5,
    target_avg_loss: Union[float, None] = 1e-4,
    mamba_learning_rate: float | None = None,
    weight_decay: float = 0.0,
    optimizer_foreach: bool = False,
    max_grad_norm: float = 1.0,
    precision: str = "fp16",
    enable_gradient_checkpointing: bool = True,
    checkpoint_use_reentrant: bool | None = None,
    attn: str = "auto",
    ff_chunk_size: int = 0,
    ff_chunk_dim: int = 1,
    keep_unet_fp32: bool = False,
    unet_shard_mode: str = "off",
    per_gpu_max_mem_gib: int | Sequence[int] = 11,
    unet_device_map: dict[str, int] | None = None,
    unet_shard_strategy: str = "two_stage_split",
    log_interval: int = 10,
    seed: int = 42,
    tensorboard_log_dir: Union[str, None] = None,
    save_interval_epochs: int = 1,
    scheduler_type: str = "cosine",
    scheduler_gamma: float = 0.95,
    scheduler_t_max: int = 1000,
    scheduler_eta_min: float = 1e-6,
    num_warmup_steps: int = 0,
    grad_accum_steps: int = 1,
    deepspeed: dict[str, Any] | None = None,
    deepspeed_plugin_key: str | None = None,
    deepspeed_plugin_configs: dict[str, dict[str, Any]] | None = None,
    dataset_split_ratios: Sequence[float] | None = None,
    dataset_split_group: str = "train",
    dataset_split_seed: int = 42,
    val_split: str | None = None,
    val_interval_epochs: int = 1,
    max_val_videos: int | None = None,
    resume_from: str | None = None,
    overlap_teacher_prob: float = 1.0,
    overlap_noise_std: float = 0.0,
    target_override_video_path: str | None = None,
    target_override_is_sbs: bool = True,
    random_crop_per_chunk: bool = False,
    vae_decode_device: str | None = None,
    mamba_use_fast_path: bool = True,
    mamba_autotune_warmup: bool = True,
    resume_mamba_runtime_flags: bool = True,
    mamba_auto_fallback: bool = True,
    mamba_fallback_mode: str = "inplace_or_reload",
    mamba_gate_schedule: str = "none",
    mamba_gate_start: float = 0.0,
    mamba_gate_end: float = 1.0,
    mamba_gate_log_interval: int = 10,
    debug_deepspeed_graph: bool = False,
    debug_deepspeed_param_scan: bool = False,
    mamba_diag_interval: int = 10,
    denoise_diag_interval: int = 0,
    image_diag_interval: int = 0,
    image_diag_max_frames: int = 2,
    image_diag_decode_chunk_size: int = 1,
    diffusion_scheduler_type: str = "ddpm",
    euler_train_num_steps: int = 20,
    euler_timestep_sampling: str = "uniform",
    euler_low_sigma_prob: float = 0.0,
    euler_low_sigma_fraction: float = 0.35,
    preflight_only: bool = False,
) -> bool:
    """Fine-tune the stereo inpainting pipeline.

    Args:
        pre_trained_path: 事前学習済み重み (image_encoder/vae を含む)。
        unet_path: 学習対象の UNet 重み (diffusers 形式)。
        train_glob: 学習動画のグロブパターン。
        save_dir: ログ/チェックポイント保存先。
        stage_name/stage_h/stage_w: 固定ステージ解像度と名称。
        stage_epochs/stage_lr: ステージごとの学習設定。
        use_prev_target_overlap: オーバーラップ領域を前チャンクGTで置換するか。
        unet_shard_mode: "off" | "on" | "auto"（OOM 時のみ 2GPU へ分割）。
        precision: "fp16" | "bf16" | "fp32"。
        dataset_split_ratios/dataset_split_group: 任意の分割設定。
        resume_from: 既存チェックポイントの再開。
        target_override_video_path: 指定時、学習targetを外部教師動画の右目フレームに差し替える。
    """
    ensure_logging_configured()
    logger.info("Starting training run. Saving artifacts to %s", save_dir)
    logger.info(
        "Tip: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (Pytorch_CUDA_ALLOC_CONF is ignored)."
    )
    os.makedirs(save_dir, exist_ok=True)
    if save_interval_epochs < 1:
        raise ValueError("save_interval_epochs must be >= 1")
    if val_split is not None and val_interval_epochs < 1:
        raise ValueError("val_interval_epochs must be >= 1 when val_split is set.")
    sched_key = (scheduler_type or "none").lower()
    if sched_key not in {"none", "cosine", "cosine_with_warmup", "exponential"}:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    if sched_key == "exponential" and scheduler_gamma <= 0:
        raise ValueError("scheduler_gamma must be > 0 for ExponentialLR")
    if sched_key == "cosine_with_warmup" and num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be >= 0 for cosine_with_warmup")
    # 乱数シードの固定 (再現性向上)
    set_global_seed(seed)
    # Ctrl+C や SIGTERM を受け取ったら「安全な地点」で停止するためのフラグ
    stop_event = setup_interrupt_handler()

    precision_key = (precision or "").lower()
    mixed_precision = "bf16" if precision_key == "bf16" else ("fp16" if precision_key == "fp16" else "no")
    local_rank = str(os.environ.get("LOCAL_RANK", "")).strip()
    is_local_main = local_rank in {"", "0"}

    def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True, ensure_ascii=True, default=str)
        os.replace(tmp_path, path)

    def _wait_for_file(path: str, timeout_s: float = 120.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return
            time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for DeepSpeed config: {path}")

    def _write_deepspeed_config(
        cfg: dict[str, Any],
        *,
        grad_steps: int,
        precision_mode: str,
        filename: str = "ds_config.json",
    ) -> str:
        zero_stage = int(cfg.get("zero_stage", 3))
        ignore_unused = bool(cfg.get("ignore_unused_parameters", zero_stage >= 2))
        offload_opt = str(cfg.get("offload_optimizer_device", "cpu")).strip().lower()
        offload_param = str(cfg.get("offload_param_device", "none")).strip().lower()
        train_batch = cfg.get("train_batch_size", None)
        train_micro = cfg.get("train_micro_batch_size_per_gpu", None)
        zero_opt: dict[str, Any] = {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
        }
        if cfg.get("reduce_bucket_size", None) is not None:
            zero_opt["reduce_bucket_size"] = int(cfg["reduce_bucket_size"])
        if cfg.get("allgather_bucket_size", None) is not None:
            zero_opt["allgather_bucket_size"] = int(cfg["allgather_bucket_size"])
        if zero_stage >= 2 and ignore_unused:
            zero_opt["ignore_unused_parameters"] = True
        if offload_opt not in {"none", "cpu"}:
            raise ValueError(f"offload_optimizer_device must be 'cpu' or 'none', got {offload_opt}")
        if offload_param not in {"none", "cpu"}:
            raise ValueError(f"offload_param_device must be 'cpu' or 'none', got {offload_param}")
        if offload_opt == "cpu":
            zero_opt["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        if offload_param == "cpu":
            zero_opt["offload_param"] = {"device": "cpu", "pin_memory": True}
        ds_cfg: dict[str, Any] = {
            "zero_optimization": zero_opt,
            "gradient_accumulation_steps": int(grad_steps),
        }
        if train_batch is not None:
            ds_cfg["train_batch_size"] = int(train_batch)
        elif train_micro is not None:
            ds_cfg["train_micro_batch_size_per_gpu"] = int(train_micro)
        else:
            ds_cfg["train_micro_batch_size_per_gpu"] = 1
        if precision_mode == "bf16":
            ds_cfg["bf16"] = {"enabled": True}
            ds_cfg["fp16"] = {"enabled": False}
        elif precision_mode == "fp16":
            ds_cfg["bf16"] = {"enabled": False}
            ds_cfg["fp16"] = {"enabled": True}
        else:
            ds_cfg["bf16"] = {"enabled": False}
            ds_cfg["fp16"] = {"enabled": False}
        path = os.path.join(save_dir, filename)
        _atomic_write_json(path, ds_cfg)
        return path

    def _ensure_deepspeed_batch_config(ds_path: str, cfg: dict[str, Any]) -> str:
        if not is_local_main:
            _wait_for_file(ds_path)
            patched_path = os.path.join(save_dir, "ds_config_autofix.json")
            if os.path.exists(patched_path) and os.path.getsize(patched_path) > 0:
                return patched_path
            return ds_path
        try:
            with open(ds_path, "r", encoding="utf-8") as fp:
                ds_loaded = json.load(fp)
        except Exception as err:
            logger.warning(
                "DeepSpeed config %s is not readable as JSON; skipping batch-size check: %s",
                ds_path,
                err,
            )
            return ds_path
        if "train_batch_size" in ds_loaded or "train_micro_batch_size_per_gpu" in ds_loaded:
            return ds_path
        train_batch = cfg.get("train_batch_size", None)
        train_micro = cfg.get("train_micro_batch_size_per_gpu", None)
        if train_batch is not None:
            ds_loaded["train_batch_size"] = int(train_batch)
            detail = f"train_batch_size={ds_loaded['train_batch_size']}"
        else:
            if train_micro is None:
                train_micro = 1
            ds_loaded["train_micro_batch_size_per_gpu"] = int(train_micro)
            detail = f"train_micro_batch_size_per_gpu={ds_loaded['train_micro_batch_size_per_gpu']}"
        patched_path = os.path.join(save_dir, "ds_config_autofix.json")
        _atomic_write_json(patched_path, ds_loaded)
        logger.warning(
            "DeepSpeed config missing train batch size; wrote %s (%s).",
            patched_path,
            detail,
        )
        return patched_path

    ds_cfg = deepspeed or {}
    ds_enabled = bool(ds_cfg.get("enabled", False))
    accelerator = None
    is_main_process = True
    ds_state_dir = None
    if ds_enabled:
        ds_grad_steps = int(ds_cfg.get("gradient_accumulation_steps", grad_accum_steps))
        if ds_grad_steps != grad_accum_steps:
            logger.info(
                "DeepSpeed gradient_accumulation_steps override: %d -> %d",
                grad_accum_steps,
                ds_grad_steps,
            )
            grad_accum_steps = ds_grad_steps
        ds_config_path = str(ds_cfg.get("deepspeed_config_path", "") or "").strip()
        if not ds_config_path:
            if is_local_main:
                ds_config_path = _write_deepspeed_config(
                    ds_cfg, grad_steps=grad_accum_steps, precision_mode=precision_key
                )
            else:
                ds_config_path = os.path.join(save_dir, "ds_config.json")
                _wait_for_file(ds_config_path)
        ds_config_path = _ensure_deepspeed_batch_config(ds_config_path, ds_cfg)
        try:
            from accelerate import Accelerator
            from accelerate.utils import DeepSpeedPlugin
        except Exception as err:  # pragma: no cover - optional dependency
            raise RuntimeError("DeepSpeed enabled but accelerate is not available.") from err
        global _CACHED_DS_ACCELERATOR, _CACHED_DS_ACCELERATOR_CFG
        plugin_key = str(deepspeed_plugin_key or "default")
        if _CACHED_DS_ACCELERATOR is None:
            plugin_configs = deepspeed_plugin_configs or {plugin_key: ds_cfg}
            ds_plugins: dict[str, Any] = {}
            for raw_key, raw_plugin_cfg in plugin_configs.items():
                key = str(raw_key)
                plugin_cfg = dict(raw_plugin_cfg or {})
                plugin_grad_steps = int(plugin_cfg.get("gradient_accumulation_steps", grad_accum_steps))
                plugin_path = str(plugin_cfg.get("deepspeed_config_path", "") or "").strip()
                if not plugin_path:
                    safe_key = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in key)
                    filename = "ds_config.json" if safe_key == "default" else f"ds_config_{safe_key}.json"
                    if is_local_main:
                        plugin_path = _write_deepspeed_config(
                            plugin_cfg,
                            grad_steps=plugin_grad_steps,
                            precision_mode=precision_key,
                            filename=filename,
                        )
                    else:
                        plugin_path = os.path.join(save_dir, filename)
                        _wait_for_file(plugin_path)
                plugin_path = _ensure_deepspeed_batch_config(plugin_path, plugin_cfg)
                ds_plugins[key] = DeepSpeedPlugin(
                    zero_stage=int(plugin_cfg.get("zero_stage", 3)),
                    offload_optimizer_device=str(plugin_cfg.get("offload_optimizer_device", "cpu")).strip().lower(),
                    offload_param_device=str(plugin_cfg.get("offload_param_device", "none")).strip().lower(),
                    gradient_accumulation_steps=plugin_grad_steps,
                    hf_ds_config=plugin_path or None,
                )
            accelerator = Accelerator(
                mixed_precision=mixed_precision,
                deepspeed_plugins=ds_plugins,
                gradient_accumulation_steps=int(grad_accum_steps),
            )
            if plugin_key in ds_plugins:
                accelerator.state.select_deepspeed_plugin(plugin_key)
            _CACHED_DS_ACCELERATOR = accelerator
            _CACHED_DS_ACCELERATOR_CFG = {
                "mixed_precision": mixed_precision,
                "grad_accum_steps": int(grad_accum_steps),
                "plugin_keys": sorted(ds_plugins.keys()),
            }
        else:
            accelerator = _CACHED_DS_ACCELERATOR
            cached = _CACHED_DS_ACCELERATOR_CFG
            if cached.get("mixed_precision") != mixed_precision or cached.get("grad_accum_steps") != int(
                grad_accum_steps
            ):
                logger.warning(
                    "Reusing cached DeepSpeed Accelerator with mixed_precision=%s grad_accum_steps=%s "
                    "(current: %s/%s).",
                    cached.get("mixed_precision"),
                    cached.get("grad_accum_steps"),
                    mixed_precision,
                    grad_accum_steps,
                )
            cached_keys = set(cached.get("plugin_keys", []))
            if plugin_key not in cached_keys:
                raise RuntimeError(
                    f"DeepSpeed plugin key {plugin_key!r} was not initialized. Known keys: {sorted(cached_keys)}"
                )
            accelerator.state.select_deepspeed_plugin(plugin_key)
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        ds_state_dir = os.path.join(save_dir, "deepspeed_state_latest")
        logger.info(
            "DeepSpeed enabled: zero_stage=%s offload_optimizer=%s offload_param=%s config=%s",
            ds_cfg.get("zero_stage", 3),
            ds_cfg.get("offload_optimizer_device", "cpu"),
            ds_cfg.get("offload_param_device", "none"),
            ds_config_path,
        )
    else:
        # 使用デバイスの決定 (CUDA 優先)
        device = get_compute_device()
    # 精度の解決: dtype / autocast の有無 / GradScaler をまとめて取得
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    torch_dtype, use_amp, scaler = resolve_precision(precision, device)
    logger.info("Using device %s with dtype %s (AMP enabled: %s)", device, torch_dtype, use_amp)
    def _use_scaler() -> bool:
        return (not ds_enabled) and scaler is not None and scaler.is_enabled()
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if stage_idx < 1:
        raise ValueError("stage_idx must be >= 1")
    if not str(stage_name).strip():
        raise ValueError("stage_name must be non-empty")
    stage_h = int(stage_h)
    stage_w = int(stage_w)
    if stage_h <= 0 or stage_w <= 0:
        raise ValueError("stage_h/stage_w must be positive integers.")
    stage_epochs = int(stage_epochs)
    if stage_epochs < 1:
        raise ValueError("stage_epochs must be >= 1")
    stage_lr = float(stage_lr)
    if stage_lr <= 0:
        raise ValueError("stage_lr must be > 0")

    shard_mode = (unet_shard_mode or "off").strip().lower()
    if shard_mode not in {"off", "on", "auto"}:
        raise ValueError("unet_shard_mode must be one of: off, on, auto")
    if ds_enabled:
        if shard_mode != "off" or unet_device_map is not None:
            logger.info("DeepSpeed enabled; disabling UNet sharding/device_map.")
        shard_mode = "off"
        unet_device_map = None
    shard_strategy = (unet_shard_strategy or "two_stage_split").strip().lower()
    if (
        stage_idx >= 2
        and unet_device_map is None
        and shard_strategy in {"two_stage_split", "manual_split", "fallback"}
    ):
        shard_strategy = "frontload_gpu1"
        logger.info(
            "Stage %d: overriding unet_shard_strategy to frontload_gpu1 to place down_blocks on cuda:1.",
            stage_idx,
        )
    if shard_mode == "off" and unet_device_map is not None:
        logger.warning("unet_device_map is ignored because unet_shard_mode=off.")

    frames_chunk = int(frames_chunk)
    if frames_chunk < 1:
        raise ValueError("frames_chunk must be >= 1")
    crop_multiple = 64
    crop_size = (stage_h, stage_w)
    crop_min_size = crop_size
    crop_max_size = crop_size
    if ff_chunk_size and ff_chunk_dim == 1 and frames_chunk % int(ff_chunk_size) != 0:
        logger.warning(
            "ff_chunk_size=%s is not divisible by frames_chunk=%s with ff_chunk_dim=1; overriding to 1.",
            ff_chunk_size,
            frames_chunk,
        )
        ff_chunk_size = 1
    mamba_fallback_mode = (mamba_fallback_mode or "inplace_or_reload").strip().lower()
    if mamba_fallback_mode not in {"inplace_only", "reload_only", "inplace_or_reload"}:
        raise ValueError("mamba_fallback_mode must be one of: inplace_only, reload_only, inplace_or_reload")
    mamba_gate_schedule_key = (mamba_gate_schedule or "none").strip().lower()
    if mamba_gate_schedule_key not in {"none", "linear"}:
        raise ValueError("mamba_gate_schedule must be one of: none, linear")
    mamba_gate_start = max(0.0, min(1.0, float(mamba_gate_start)))
    mamba_gate_end = max(0.0, min(1.0, float(mamba_gate_end)))
    mamba_gate_log_interval = int(max(0, mamba_gate_log_interval))
    effective_mamba_use_fast_path = bool(mamba_use_fast_path)
    effective_mamba_autotune_warmup = bool(mamba_autotune_warmup)

    def _set_mamba_env(use_fast_path: bool, autotune_warmup: bool) -> None:
        os.environ["MAMBA_MEM_EFF"] = "1" if use_fast_path else "0"
        os.environ["MAMBA_USE_MEM_EFF_PATH"] = "1" if use_fast_path else "0"
        os.environ["MAMBA_AUTOTUNE_WARMUP"] = "1" if autotune_warmup else "0"

    def _log_effective_mamba_flags(context: str) -> None:
        logger.info(
            "%s: effective_mamba_use_fast_path=%s effective_mamba_autotune_warmup=%s",
            context,
            effective_mamba_use_fast_path,
            effective_mamba_autotune_warmup,
        )

    _set_mamba_env(effective_mamba_use_fast_path, effective_mamba_autotune_warmup)

    # 事前学習済みの image_encoder/vae と、学習対象の UNet を組み込んだパイプラインを構築
    log_cuda_memory_stats(
        f"before_pipeline_load:stage{stage_idx}",
        device=device,
        summary=(device.type == "cuda" and stage_idx == 2),
    )
    pipeline = load_inpainting_pipeline(
        pre_trained_path=pre_trained_path,
        unet_path=unet_path,
        torch_dtype=torch_dtype,
        device=device,
        pipeline_device=device,
    )
    log_cuda_memory_stats(f"after_pipeline_load:stage{stage_idx}", device=device)
    log_vram_usage("After loading inpainting pipeline", device, level=logging.INFO)
    # AMP 安定化のため、必要に応じて UNet パラメータのみ FP32 で保持（計算は autocast で半精度）。
    # bf16 学習時は意図通り半精度になるよう FP32 へは強制変換しない。
    if keep_unet_fp32 and precision_key != "bf16":
        try:
            pipeline.unet.to(dtype=torch.float32)
        except Exception:
            pass
    elif keep_unet_fp32 and precision_key == "bf16":
        logger.info("keep_unet_fp32 requested but precision=bf16; keeping UNet in bf16 to honor precision setting.")

    def log_param_bytes_by_device(model: torch.nn.Module) -> None:
        bytes_by_dev = defaultdict(int)
        for _, p in model.named_parameters():
            if p.device.type == "cuda":
                bytes_by_dev[p.device.index] += p.numel() * p.element_size()
        gib = {k: v / (1024**3) for k, v in bytes_by_dev.items()}
        logger.info("UNet param bytes by device (GiB): %s", gib)

    if hasattr(pipeline.unet, "hf_device_map"):
        logger.info("UNet hf_device_map keys: %d", len(pipeline.unet.hf_device_map))
    log_param_bytes_by_device(pipeline.unet)
    _log_effective_mamba_flags("Initial mamba flags")
    updated = apply_mamba_runtime_flags(
        pipeline.unet,
        use_fast_path=effective_mamba_use_fast_path,
        autotune_warmup=effective_mamba_autotune_warmup,
    )
    if updated:
        logger.info("Applied mamba runtime flags to %d modules (initial).", updated)
    if enable_gradient_checkpointing:
        enable_force_reentrant_checkpoint(True)
        _patch_modules_holding_checkpoint_symbol()
        logger.info("Force reentrant checkpointing enabled (use_reentrant=True).")
    # 勾配チェックポイントや注意機構の省メモリ化を有効化
    configure_unet_memory_features(
        pipeline=pipeline,
        enable_gradient_checkpointing=enable_gradient_checkpointing,
        checkpoint_use_reentrant=checkpoint_use_reentrant,
        attn_mode=attn,
        ff_chunk_size=ff_chunk_size if ff_chunk_size > 0 else None,
        ff_chunk_dim=ff_chunk_dim,
    )
    # VAE 側のスライシング/タイル化 (対応していれば有効化)
    enable_vae_memory_helpers(pipeline)

    def _resolve_vae_decode_device(requested: str | None, fallback: torch.device) -> torch.device:
        if not requested:
            return fallback
        try:
            decode_device = torch.device(requested)
        except Exception as err:
            logger.warning("Invalid vae_decode_device=%s (%s); using %s", requested, err, fallback)
            return fallback
        if decode_device.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("vae_decode_device=%s but CUDA unavailable; using %s", decode_device, fallback)
                return fallback
            index = 0 if decode_device.index is None else decode_device.index
            if index >= torch.cuda.device_count():
                logger.warning(
                    "vae_decode_device=%s exceeds CUDA device count (%d); using %s",
                    decode_device,
                    torch.cuda.device_count(),
                    fallback,
                )
                return fallback
        return decode_device

    resolved_vae_decode_device = _resolve_vae_decode_device(vae_decode_device, device)

    def _resolve_unet_input_device() -> torch.device:
        """Resolve the device for UNet inputs (conv_in preferred, else first parameter)."""
        fallback = device
        conv_in = getattr(pipeline.unet, "conv_in", None)
        if conv_in is not None:
            try:
                ref_param = next(conv_in.parameters())
                ref_device = ref_param.device
                if ref_device.type == "cuda" and ref_device.index is None:
                    return torch.device("cuda:0")
                return ref_device
            except StopIteration:
                pass
        try:
            ref_param = next(pipeline.unet.parameters())
            ref_device = ref_param.device
            if ref_device.type == "cuda" and ref_device.index is None:
                return torch.device("cuda:0")
            return ref_device
        except StopIteration:
            pass
        if fallback.type == "cuda" and fallback.index is None:
            return torch.device("cuda:0")
        return fallback

    def _format_device(value: torch.device) -> str:
        if value.type != "cuda":
            return str(value)
        index = 0 if value.index is None else value.index
        return f"cuda:{index}"

    diffusion_scheduler_key = (diffusion_scheduler_type or "ddpm").strip().lower()
    if diffusion_scheduler_key not in {"ddpm", "euler"}:
        raise ValueError("diffusion_scheduler_type must be one of: ddpm, euler")
    if int(euler_train_num_steps) < 2:
        raise ValueError("euler_train_num_steps must be >= 2")
    euler_timestep_sampling_key = (euler_timestep_sampling or "uniform").strip().lower()
    if euler_timestep_sampling_key not in {"uniform", "low_sigma"}:
        raise ValueError("euler_timestep_sampling must be one of: uniform, low_sigma")
    euler_low_sigma_prob = min(max(float(euler_low_sigma_prob), 0.0), 1.0)
    euler_low_sigma_fraction = min(max(float(euler_low_sigma_fraction), 0.0), 1.0)

    # 学習用ノイズスケジューラ。DDPM は従来互換、Euler は origin 推論と同じ
    # continuous/Karras sigma 座標に合わせる。
    if diffusion_scheduler_key == "euler":
        noise_scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        noise_scheduler.set_timesteps(int(euler_train_num_steps), device=device)
    else:
        noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        pred_type = getattr(pipeline.scheduler.config, "prediction_type", None)
        if pred_type is not None and noise_scheduler.config.prediction_type != pred_type:
            noise_scheduler.register_to_config(prediction_type=pred_type)
    def _sched_cfg_dict(sched: Any) -> dict[str, Any]:
        cfg = getattr(sched, "config", None)
        return {
            "class": sched.__class__.__name__,
            "prediction_type": getattr(cfg, "prediction_type", None),
            "num_train_timesteps": getattr(cfg, "num_train_timesteps", None),
            "beta_schedule": getattr(cfg, "beta_schedule", None),
            "rescale_betas_zero_snr": getattr(cfg, "rescale_betas_zero_snr", None),
            "timestep_spacing": getattr(cfg, "timestep_spacing", None),
            "steps_offset": getattr(cfg, "steps_offset", None),
            "timestep_type": getattr(cfg, "timestep_type", None),
            "use_karras_sigmas": getattr(cfg, "use_karras_sigmas", None),
            "sigma_min": getattr(cfg, "sigma_min", None),
            "sigma_max": getattr(cfg, "sigma_max", None),
        }
    logger.info("[sched][train][pipeline] %s", _sched_cfg_dict(pipeline.scheduler))
    logger.info("[sched][train][noise_scheduler] %s", _sched_cfg_dict(noise_scheduler))
    if diffusion_scheduler_key == "euler":
        logger.info(
            "[sched][train][euler] num_steps=%d timestep_sampling=%s low_sigma_prob=%.3f "
            "low_sigma_fraction=%.3f timesteps_head=%s sigmas_head=%s init_noise_sigma=%s",
            int(euler_train_num_steps),
            euler_timestep_sampling_key,
            euler_low_sigma_prob,
            euler_low_sigma_fraction,
            noise_scheduler.timesteps[: min(8, len(noise_scheduler.timesteps))].detach().cpu().tolist(),
            noise_scheduler.sigmas[: min(8, len(noise_scheduler.sigmas))].detach().cpu().tolist(),
            getattr(noise_scheduler, "init_noise_sigma", None),
        )

    def _tensor_diag(metrics: dict[str, torch.Tensor], prefix: str, tensor: torch.Tensor) -> None:
        with torch.no_grad():
            data = tensor.detach().float()
            finite = torch.isfinite(data)
            if not bool(finite.all().item()):
                data = torch.where(finite, data, torch.zeros_like(data))
            metrics[f"{prefix}_mean"] = data.mean().detach()
            metrics[f"{prefix}_std"] = data.std(unbiased=False).detach()
            metrics[f"{prefix}_rms"] = data.pow(2).mean().sqrt().detach()
            metrics[f"{prefix}_abs_mean"] = data.abs().mean().detach()
            metrics[f"{prefix}_max_abs"] = data.abs().amax().detach()
            metrics[f"{prefix}_finite_frac"] = finite.float().mean().detach()

    def compute_batch_loss(
        batch: Any,
        *,
        collect_denoise_diag: bool = False,
        collect_image_diag: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward UNet once against a mini-batch and return loss + per-component metrics."""
        unet_in_device = _resolve_unet_input_device()

        def _to_unet_entry(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if isinstance(tensor, torch.Tensor) and tensor.device != unet_in_device:
                return tensor.to(unet_in_device, non_blocking=True)
            return tensor

        with torch.autocast(device_type=device.type, dtype=torch_dtype, enabled=use_amp):
            H, W = batch.cond.shape[2], batch.cond.shape[3]
            try:
                if next(pipeline.image_encoder.parameters()).device != device:
                    pipeline.image_encoder.to(device)
            except Exception:
                pass
            with torch.no_grad():
                image_embeddings = pipeline._encode_image(
                    batch.cond[0:1], device=device, num_videos_per_prompt=1, do_classifier_free_guidance=False
                )
            frames_cond = pipeline.image_processor.preprocess(batch.cond, height=H, width=W)
            if noise_aug_strength > 0.0:
                noise = torch.randn_like(frames_cond)
                frames_cond = frames_cond + noise_aug_strength * noise

            vae_device = resolved_vae_decode_device
            if next(pipeline.vae.parameters()).device != vae_device:
                pipeline.vae.to(vae_device)
            frames_cond_vae = frames_cond.to(vae_device)

            latent_list = []
            with torch.no_grad():
                for i_f in range(0, frames_cond.shape[0], max(1, vae_encode_chunk_size)):
                    latent_list.append(
                        pipeline.vae.encode(
                            frames_cond_vae[i_f : i_f + max(1, vae_encode_chunk_size)]
                        ).latent_dist.mode()
                    )
            frame_latents = torch.cat(latent_list, dim=0).unsqueeze(0)
            frame_latents *= pipeline.vae.config.scaling_factor
            frame_latents = frame_latents.to(device=unet_in_device, dtype=image_embeddings.dtype)
            frame_latents = _to_unet_entry(frame_latents)

            with torch.no_grad():
                frames_mask = pipeline.mask_processor.preprocess(batch.mask, height=H, width=W)
                frames_mask = torch.nn.functional.interpolate(
                    frames_mask, scale_factor=1 / pipeline.vae_scale_factor
                ).unsqueeze(0)
            mask_latents = frames_mask.to(device=unet_in_device, dtype=image_embeddings.dtype)
            mask_latents = _to_unet_entry(mask_latents)

            fps_ = fps_condition - 1
            add_time_ids = torch.tensor(
                [[float(fps_), float(motion_bucket_id), float(noise_aug_strength)]],
                dtype=image_embeddings.dtype,
                device=unet_in_device,
            )
            add_time_ids = _to_unet_entry(add_time_ids)

            frames_tgt = pipeline.image_processor.preprocess(batch.target, height=H, width=W)
            tgt_lat_list = []
            with torch.no_grad():
                for i_f in range(0, frames_tgt.shape[0], max(1, vae_encode_chunk_size)):
                    tgt_lat_list.append(
                        pipeline.vae.encode(
                            frames_tgt[i_f : i_f + max(1, vae_encode_chunk_size)].to(vae_device)
                        ).latent_dist.mode()
                    )
            x0 = torch.cat(tgt_lat_list, dim=0).unsqueeze(0).to(image_embeddings.dtype)
            x0 = x0.to(device=unet_in_device)
            x0 = _to_unet_entry(x0)
            x0 = x0 * pipeline.vae.config.scaling_factor

            eps = torch.randn_like(x0)
            sigma_value = None
            if diffusion_scheduler_key == "euler":
                timestep_count = len(noise_scheduler.timesteps)
                sample_low_sigma = (
                    euler_timestep_sampling_key == "low_sigma"
                    and euler_low_sigma_prob > 0.0
                    and float(torch.rand((), device=unet_in_device).item()) < euler_low_sigma_prob
                )
                low_sigma_start = 0
                if sample_low_sigma:
                    low_sigma_count = max(1, int(round(timestep_count * euler_low_sigma_fraction)))
                    low_sigma_start = max(0, timestep_count - low_sigma_count)
                step_idx = torch.randint(
                    low_sigma_start,
                    timestep_count,
                    (1,),
                    device=unet_in_device,
                    dtype=torch.long,
                )
                t = noise_scheduler.timesteps.to(device=unet_in_device, dtype=torch.float32)[step_idx]
                sigma_value = noise_scheduler.sigmas.to(device=unet_in_device, dtype=x0.dtype)[step_idx]
                sigma = sigma_value.flatten()
                while len(sigma.shape) < len(x0.shape):
                    sigma = sigma.unsqueeze(-1)
                sigma_denom = (sigma.pow(2) + 1.0).sqrt()
                x_t_raw = x0 + eps * sigma
                x_t = (x_t_raw / sigma_denom).to(dtype=frame_latents.dtype)
                x_t = _to_unet_entry(x_t)
                if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                    target = (eps - sigma * x0) / sigma_denom
                else:
                    target = eps
            else:
                t = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (1,),
                    device=unet_in_device,
                    dtype=torch.long,
                )
                t = _to_unet_entry(t)
                x_t = noise_scheduler.add_noise(x0, eps, t)
                x_t = x_t.to(dtype=frame_latents.dtype)
                x_t = _to_unet_entry(x_t)
                if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                    target = noise_scheduler.get_velocity(x0, eps, t)
                else:
                    target = eps

            image_embeddings = image_embeddings.to(device=unet_in_device, dtype=image_embeddings.dtype)
            image_embeddings = _to_unet_entry(image_embeddings)
            logger.debug(
                "latent devices: x_t=%s frame_latents=%s mask_latents=%s add_time_ids=%s unet_in=%s",
                x_t.device,
                frame_latents.device,
                mask_latents.device,
                add_time_ids.device,
                _format_device(unet_in_device),
            )
            logger.debug(
                "latent dtypes: x_t=%s frame_latents=%s mask_latents=%s add_time_ids=%s",
                x_t.dtype,
                frame_latents.dtype,
                mask_latents.dtype,
                add_time_ids.dtype,
            )
            latent_model_input = torch.cat([x_t, frame_latents, mask_latents], dim=2)
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=add_time_ids,
                return_dict=False,
            )[0]

            noise_loss = F.mse_loss(noise_pred, target)
            metrics = {"timestep": t.detach(), "loss_noise_mse": noise_loss.detach()}
            if sigma_value is not None:
                metrics["sigma"] = sigma_value.detach().float()
            if collect_image_diag:
                with torch.no_grad():
                    x0_pred = None
                    if diffusion_scheduler_key == "euler":
                        sigma_for_pred = sigma_value.flatten()
                        while len(sigma_for_pred.shape) < len(x_t.shape):
                            sigma_for_pred = sigma_for_pred.unsqueeze(-1)
                        sigma_denom_for_pred = (sigma_for_pred.pow(2) + 1.0).sqrt()
                        if getattr(noise_scheduler.config, "prediction_type", "epsilon") == "v_prediction":
                            x0_pred = (
                                x_t.float() - sigma_for_pred.float() * noise_pred.detach().float()
                            ) / sigma_denom_for_pred.float()
                        else:
                            x0_pred = (
                                x_t.float() * sigma_denom_for_pred.float()
                                - sigma_for_pred.float() * noise_pred.detach().float()
                            )
                    if x0_pred is not None:
                        diag_frames = min(int(x0_pred.shape[1]), max(1, int(image_diag_max_frames)))
                        decoded = pipeline.decode_latents(
                            x0_pred[:, :diag_frames].to(device=vae_device, dtype=torch_dtype),
                            num_frames=diag_frames,
                            decode_chunk_size=max(1, int(image_diag_decode_chunk_size)),
                        )
                        decoded = (decoded[0].permute(1, 0, 2, 3) / 2.0 + 0.5).clamp(0.0, 1.0)
                        target_img = batch.target[:diag_frames].detach().to(
                            device=decoded.device,
                            dtype=decoded.dtype,
                        )
                        mask_img = batch.mask[:diag_frames].detach().to(
                            device=decoded.device,
                            dtype=decoded.dtype,
                        )
                        err_img = decoded - target_img
                        image_mse = err_img.pow(2).mean()
                        image_l1 = err_img.abs().mean()
                        metrics["image_diag_mse"] = image_mse.detach()
                        metrics["image_diag_l1"] = image_l1.detach()
                        metrics["image_diag_psnr"] = (-10.0 * torch.log10(torch.clamp(image_mse, min=1e-8))).detach()
                        mask_sum = mask_img.sum() * float(decoded.shape[1])
                        if float(mask_sum.detach().cpu().item()) > 0.0:
                            mask_expanded = mask_img.expand_as(decoded)
                            mask_mse = (err_img.pow(2) * mask_expanded).sum() / mask_sum
                            mask_l1 = (err_img.abs() * mask_expanded).sum() / mask_sum
                            metrics["image_diag_mask_mse"] = mask_mse.detach()
                            metrics["image_diag_mask_l1"] = mask_l1.detach()
                            metrics["image_diag_mask_psnr"] = (
                                -10.0 * torch.log10(torch.clamp(mask_mse, min=1e-8))
                            ).detach()
            if collect_denoise_diag:
                _tensor_diag(metrics, "x0", x0)
                _tensor_diag(metrics, "eps", eps)
                _tensor_diag(metrics, "x_t", x_t)
                _tensor_diag(metrics, "target", target)
                _tensor_diag(metrics, "noise_pred", noise_pred)
                pred_err = noise_pred.detach() - target.detach()
                _tensor_diag(metrics, "pred_err", pred_err)
                with torch.no_grad():
                    metrics["pred_target_cos"] = F.cosine_similarity(
                        noise_pred.detach().float().flatten(),
                        target.detach().float().flatten(),
                        dim=0,
                    ).detach()
            return noise_loss, metrics
    preflight_batch: Any | None = None
    preflight_crop_hw: tuple[int, int] | None = None
    chunk_count_cache: dict[str, int] = {}

    def _make_preflight_batch() -> Any:
        nonlocal preflight_crop_hw
        sample_videos = sorted(glob.glob(train_glob))
        if not sample_videos:
            raise FileNotFoundError(f"No training videos found for pattern: {train_glob}")
        sample_video = sample_videos[0]
        logger.info(
            "Preflight: checking stage crop %dx%d on %s",
            stage_h,
            stage_w,
            os.path.basename(sample_video),
        )
        try:
            batches_pf = prepare_batches(
                sample_video,
                frames_chunk=frames_chunk,
                overlap=overlap,
                device=device,
                dtype=torch_dtype if precision_key != "fp32" else torch.float32,
                crop_multiple=crop_multiple,
                crop_min_size=(stage_h, stage_w),
                crop_max_size=(stage_h, stage_w),
                random_crop=random_crop_per_chunk,
                use_prev_target_overlap=use_prev_target_overlap,
                overlap_teacher_prob=overlap_teacher_prob,
                overlap_noise_std=overlap_noise_std,
                target_override_video_path=target_override_video_path,
                target_override_is_sbs=target_override_is_sbs,
            )
            batch_pf = next(iter(batches_pf))
            preflight_crop_hw = (int(batch_pf.cond.shape[2]), int(batch_pf.cond.shape[3]))
            return batch_pf
        except StopIteration:
            raise ValueError("Preflight failed: no frames yielded from the sample video.")
        except Exception as err:
            raise RuntimeError(f"Preflight failed while preparing batch: {err}") from err

    def _get_preflight_batch() -> Any:
        nonlocal preflight_batch
        if preflight_batch is None:
            preflight_batch = _make_preflight_batch()
        return preflight_batch

    def _reset_preflight_batch() -> None:
        nonlocal preflight_batch
        preflight_batch = None

    def _get_chunk_count(path: str) -> int:
        cached = chunk_count_cache.get(path)
        if cached is not None:
            return cached
        count = estimate_num_chunks(path, frames_chunk=frames_chunk, overlap=overlap)
        chunk_count_cache[path] = count
        return count

    def _assign_videos_balanced(paths: list[str], num_processes: int) -> tuple[list[list[str]], list[int]]:
        # If videos are fewer than processes, repeat to avoid empty ranks.
        if len(paths) < num_processes:
            repeat = math.ceil(num_processes / max(1, len(paths)))
            paths = (paths * repeat)[: num_processes]
        items = [(path, _get_chunk_count(path)) for path in paths]
        random.shuffle(items)
        items.sort(key=lambda x: x[1], reverse=True)
        buckets: list[list[str]] = [[] for _ in range(num_processes)]
        totals = [0 for _ in range(num_processes)]
        for path, count in items:
            idx = min(range(num_processes), key=lambda i: totals[i])
            buckets[idx].append(path)
            totals[idx] += count
        return buckets, totals

    def _is_oom_error(err: BaseException) -> bool:
        if isinstance(err, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(err, RuntimeError):
            msg = str(err)
            msg_lower = msg.lower()
            return "out of memory" in msg_lower or "tried to allocate" in msg_lower
        return False

    def _apply_mamba_flags_inplace(context: str) -> int:
        updated = apply_mamba_runtime_flags(
            pipeline.unet,
            use_fast_path=effective_mamba_use_fast_path,
            autotune_warmup=effective_mamba_autotune_warmup,
        )
        logger.info("Mamba runtime flags applied (%s): updated_modules=%d", context, updated)
        return updated

    def _reload_pipeline_for_mamba_fallback(context: str) -> None:
        nonlocal pipeline
        logger.info("Mamba fallback reload (%s): rebuilding pipeline with slow flags.", context)
        old_pipeline = pipeline
        unet_state = None
        try:
            unet_state = pipeline.unet.state_dict()
        except Exception as err:
            logger.warning("Failed to capture UNet state before reload: %s", err)
        try:
            old_pipeline.to("cpu")
        except Exception:
            pass
        pipeline = None
        cleanup_cuda(f"before_pipeline_rebuild:{context}", old_pipeline)
        _set_mamba_env(effective_mamba_use_fast_path, effective_mamba_autotune_warmup)
        try:
            pipeline = load_inpainting_pipeline(
                pre_trained_path=pre_trained_path,
                unet_path=unet_path,
                torch_dtype=torch_dtype,
                device=device,
                pipeline_device=torch.device("cpu"),
            )
        except Exception:
            pipeline = old_pipeline
            raise
        if unet_state is not None:
            try:
                materialize_mamba_time_embed_proj_from_state_dict(pipeline.unet, unet_state)
                missing, unexpected = pipeline.unet.load_state_dict(unet_state, strict=False)
                if missing or unexpected:
                    logger.warning(
                        "Reloaded UNet with state mismatch (missing=%d unexpected=%d).",
                        len(missing),
                        len(unexpected),
                    )
            except Exception as err:
                logger.warning("Failed to restore UNet state after reload: %s", err)
        updated = apply_mamba_runtime_flags(
            pipeline.unet,
            use_fast_path=effective_mamba_use_fast_path,
            autotune_warmup=effective_mamba_autotune_warmup,
        )
        if updated:
            logger.info("Applied mamba runtime flags to %d modules (reload).", updated)
        try:
            if next(pipeline.image_encoder.parameters()).device != device:
                pipeline.image_encoder.to(device)
        except Exception:
            pass
        if enable_gradient_checkpointing:
            enable_force_reentrant_checkpoint(True)
            _patch_modules_holding_checkpoint_symbol()
        configure_unet_memory_features(
            pipeline=pipeline,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
            attn_mode=attn,
            ff_chunk_size=ff_chunk_size if ff_chunk_size > 0 else None,
            ff_chunk_dim=ff_chunk_dim,
        )
        enable_vae_memory_helpers(pipeline)
        pipeline.unet.train()
        log_param_bytes_by_device(pipeline.unet)
        old_pipeline = None

    def _run_preflight_with_mamba_fallback(
        tag: str,
        *,
        post_reload: Callable[[], None] | None = None,
    ) -> dict[int, float]:
        nonlocal effective_mamba_use_fast_path, effective_mamba_autotune_warmup
        try:
            return _preflight_runner(tag)
        except Exception as err:
            if not _is_oom_error(err):
                raise
            if not mamba_auto_fallback:
                raise
            if not (effective_mamba_use_fast_path or effective_mamba_autotune_warmup):
                raise
            logger.warning(
                "Preflight OOM (%s). Triggering mamba fallback: fast_path/autotune -> false.",
                tag,
            )
            _reset_preflight_batch()
            effective_mamba_use_fast_path = False
            effective_mamba_autotune_warmup = False
            _set_mamba_env(False, False)
            cleanup_cuda(f"mamba_fallback:{tag}")
            _log_effective_mamba_flags("Mamba fallback activated")

            if mamba_fallback_mode in {"inplace_only", "inplace_or_reload"}:
                updated = _apply_mamba_flags_inplace(f"{tag}_inplace")
                if updated > 0:
                    try:
                        return _preflight_runner(f"{tag}_mamba_inplace")
                    except Exception as err2:
                        if not _is_oom_error(err2):
                            raise
                        logger.warning("Mamba inplace fallback still OOM (%s).", tag)
                        _reset_preflight_batch()
                        cleanup_cuda(f"mamba_fallback_retry:{tag}")
                else:
                    logger.info("Mamba inplace fallback skipped: no matching modules updated.")

            if mamba_fallback_mode in {"reload_only", "inplace_or_reload"}:
                _reload_pipeline_for_mamba_fallback(tag)
                if post_reload is not None:
                    post_reload()
                return _preflight_runner(f"{tag}_mamba_reload")
            raise

    def _zero_unet_grad() -> None:
        try:
            pipeline.unet.zero_grad(set_to_none=True)
        except TypeError:
            pipeline.unet.zero_grad()

    def _preflight_runner(tag: str) -> dict[int, float]:
        _ = tag
        batch = _get_preflight_batch()
        stats: dict[int, float] = {}
        if device.type == "cuda":
            torch.cuda.empty_cache()
            for idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(idx)
        _zero_unet_grad()
        loss_pf, _ = compute_batch_loss(batch)
        if _use_scaler():
            scaler.scale(loss_pf).backward()
        else:
            loss_pf.backward()
        _zero_unet_grad()
        if device.type == "cuda":
            for idx in range(torch.cuda.device_count()):
                stats[idx] = torch.cuda.max_memory_allocated(idx) / float(1024**2)
        return stats

    def _build_optimizer_for_dryrun() -> torch.optim.Optimizer:
        trainable_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
        mamba_param_ids: set[int] = set()
        for module in pipeline.unet.modules():
            if isinstance(module, (MambaSpatioTemporalAdapter, BiMambaSelfAttention)):
                for p in module.parameters(recurse=True):
                    if p.requires_grad:
                        mamba_param_ids.add(id(p))
        if mamba_learning_rate is not None and mamba_param_ids:
            base_params = [p for p in trainable_params if id(p) not in mamba_param_ids]
            mamba_params = [p for p in trainable_params if id(p) in mamba_param_ids]
            return torch.optim.AdamW(
                [
                    {"params": base_params, "lr": stage_lr, "group_name": "base"},
                    {"params": mamba_params, "lr": mamba_learning_rate, "group_name": "mamba"},
                ],
                lr=stage_lr,
                weight_decay=weight_decay,
                foreach=optimizer_foreach,
            )
        return torch.optim.AdamW(
            [{"params": trainable_params, "lr": stage_lr, "group_name": "base"}],
            lr=stage_lr,
            weight_decay=weight_decay,
            foreach=optimizer_foreach,
        )

    def _optimizer_dryrun_for_candidate(tag: str, device_map: dict[str, int]) -> bool:
        _ = device_map
        logger.info("Optimizer dry-run for contiguous candidate %s", tag)
        optimizer = None
        try:
            optimizer = _build_optimizer_for_dryrun()
            optimizer.zero_grad(set_to_none=True)
            _zero_unet_grad()
            batch = _get_preflight_batch()
            loss_pf, _ = compute_batch_loss(batch)
            if _use_scaler():
                scaler.scale(loss_pf).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_pf.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _zero_unet_grad()
            logger.info("Optimizer dry-run OK for candidate %s", tag)
            return True
        except Exception as err:
            if _is_oom_error(err):
                logger.warning("Optimizer dry-run OOM for candidate %s", tag)
                return False
            raise
        finally:
            if optimizer is not None:
                opt_ref = optimizer
                optimizer = None
                cleanup_cuda(f"optimizer_dryrun:{tag}", opt_ref)

    def _apply_sharding(strategy_override: str | None = None) -> None:
        effective_strategy = (strategy_override or shard_strategy).strip().lower()
        maybe_shard_unet(
            pipeline=pipeline,
            shard_unet_across_gpus=True,
            per_gpu_max_mem_gib=per_gpu_max_mem_gib,
            manual_device_map=unet_device_map,
            shard_strategy=effective_strategy,
            preflight_runner=_preflight_runner if effective_strategy in {"contiguous_search", "contiguous"} else None,
            candidate_validator=_optimizer_dryrun_for_candidate
            if effective_strategy in {"contiguous_search", "contiguous"}
            else None,
        )
        pipeline.unet.train()
        if hasattr(pipeline.unet, "hf_device_map"):
            logger.info("UNet device_map updated: %s", pipeline.unet.hf_device_map)
        log_param_bytes_by_device(pipeline.unet)

    def _apply_unsharded_unet() -> None:
        try:
            pipeline.unet.to(device)
        except Exception as err:
            logger.warning("Failed to move UNet to %s after reload: %s", device, err)
            raise
        pipeline.unet.train()
        log_param_bytes_by_device(pipeline.unet)

    def _run_stage_preflight() -> bool:
        effective_mode = shard_mode
        if shard_mode == "auto" and unet_device_map is not None:
            logger.info("unet_device_map provided; auto mode will start with sharding.")
            effective_mode = "on"

        def _estimate_optimizer_state_mib_by_device() -> dict[int, float]:
            bytes_by_dev: dict[int, int] = defaultdict(int)
            for p in pipeline.unet.parameters():
                if p.requires_grad and p.device.type == "cuda":
                    bytes_by_dev[p.device.index] += p.numel() * p.element_size()
            # AdamW keeps exp_avg and exp_avg_sq on device.
            return {idx: (2.0 * total) / float(1024**2) for idx, total in bytes_by_dev.items()}

        def _headroom_low(stats: dict[int, float], *, threshold: float = 0.92) -> tuple[bool, list[int]]:
            if device.type != "cuda" or not stats:
                return False, []
            opt_mib_by_dev = _estimate_optimizer_state_mib_by_device()
            low = []
            for dev_idx, peak_mib in stats.items():
                total_mib = torch.cuda.get_device_properties(dev_idx).total_memory / float(1024**2)
                est_opt_mib = opt_mib_by_dev.get(dev_idx, 0.0)
                projected = peak_mib + est_opt_mib
                if projected >= total_mib * threshold:
                    low.append(dev_idx)
            return bool(low), low

        def _try_contiguous_fallback(context: str) -> bool:
            if shard_strategy in {"contiguous_search", "contiguous"}:
                return False
            if device.type != "cuda" or torch.cuda.device_count() < 2:
                return False
            logger.info("Sharded preflight OOM (%s); trying contiguous_search.", context)
            _reset_preflight_batch()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _apply_sharding(strategy_override="contiguous_search")
            try:
                _run_preflight_with_mamba_fallback(
                    "sharded_contiguous",
                    post_reload=lambda: _apply_sharding(strategy_override="contiguous_search"),
                )
            except RuntimeError as err:
                if _is_oom_error(err):
                    logger.info("contiguous_search preflight OOM; no viable device_map.")
                    return False
                raise
            _reset_preflight_batch()
            return True

        if effective_mode == "off":
            return False

        if effective_mode == "on":
            _apply_sharding()
            try:
                stats = _run_preflight_with_mamba_fallback("sharded", post_reload=_apply_sharding)
            except RuntimeError as err:
                if _is_oom_error(err):
                    if _try_contiguous_fallback("sharded"):
                        return True
                    raise RuntimeError(
                        "Preflight OOM after sharding. Reduce resolution/frames/precision."
                    ) from err
                raise
            needs_rebalance, low_devices = _headroom_low(stats)
            if needs_rebalance and shard_strategy in {"two_stage_split", "manual_split", "fallback"}:
                logger.info(
                    "Sharded preflight headroom low on %s; switching sharding strategy to balance_params.",
                    ",".join(f"cuda:{idx}" for idx in low_devices),
                )
                _reset_preflight_batch()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _apply_sharding(strategy_override="balance_params")
                try:
                    _run_preflight_with_mamba_fallback(
                        "sharded_rebalanced",
                        post_reload=lambda: _apply_sharding(strategy_override="balance_params"),
                    )
                except RuntimeError as err2:
                    if _is_oom_error(err2):
                        if _try_contiguous_fallback("sharded_rebalanced"):
                            return True
                        raise RuntimeError(
                            "Preflight OOM even after sharding. Reduce resolution/frames/precision."
                        ) from err2
                    raise
            _reset_preflight_batch()
            return True

        try:
            stats = _run_preflight_with_mamba_fallback("single_gpu", post_reload=_apply_unsharded_unet)
            logger.info("preflight single gpu ok")
            _reset_preflight_batch()
            if device.type == "cuda" and torch.cuda.device_count() >= 2 and stats:
                device_index = 0 if device.index is None else device.index
                total_mib = torch.cuda.get_device_properties(device_index).total_memory / float(1024**2)
                peak_mib = max(stats.values())
                est_opt_mib = _estimate_optimizer_state_mib_by_device().get(device_index, 0.0)
                projected = peak_mib + est_opt_mib
                if projected >= total_mib * 0.92:
                    logger.info(
                        "Preflight headroom low (peak=%.1f MiB + est_opt=%.1f MiB >= %.0f%% of %.1f MiB); enabling sharding.",
                        peak_mib,
                        est_opt_mib,
                        92.0,
                        total_mib,
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    _apply_sharding()
                    try:
                        _run_preflight_with_mamba_fallback("sharded", post_reload=_apply_sharding)
                    except RuntimeError as err2:
                        if _is_oom_error(err2):
                            if _try_contiguous_fallback("sharded"):
                                return True
                            raise RuntimeError(
                                "Preflight OOM even after sharding. Reduce resolution/frames/precision."
                            ) from err2
                        raise
                    _reset_preflight_batch()
                    return True
            return False
        except RuntimeError as err:
            if not _is_oom_error(err):
                raise
            logger.info("OOM -> enable sharding")
            _reset_preflight_batch()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _apply_sharding()
            try:
                stats = _run_preflight_with_mamba_fallback("sharded", post_reload=_apply_sharding)
            except RuntimeError as err2:
                if _is_oom_error(err2):
                    if _try_contiguous_fallback("sharded"):
                        return True
                    raise RuntimeError(
                        "Preflight OOM even after sharding. Reduce resolution/frames/precision."
                    ) from err2
                raise
            needs_rebalance, low_devices = _headroom_low(stats)
            if needs_rebalance and shard_strategy in {"two_stage_split", "manual_split", "fallback"}:
                logger.info(
                    "Sharded preflight headroom low on %s; switching sharding strategy to balance_params.",
                    ",".join(f"cuda:{idx}" for idx in low_devices),
                )
                _reset_preflight_batch()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _apply_sharding(strategy_override="balance_params")
                try:
                    _run_preflight_with_mamba_fallback(
                        "sharded_rebalanced",
                        post_reload=lambda: _apply_sharding(strategy_override="balance_params"),
                    )
                except RuntimeError as err3:
                    if _is_oom_error(err3):
                        if _try_contiguous_fallback("sharded_rebalanced"):
                            return True
                        raise RuntimeError(
                            "Preflight OOM even after sharding. Reduce resolution/frames/precision."
                        ) from err3
                    raise
            _reset_preflight_batch()
            return True

    # チェックポイント読込 (UNetのみ)。preflight は resume 後の状態で実行する。
    ckpt_latest_path = os.path.join(save_dir, "train_state_latest.pt")
    start_epoch = 1
    global_step = 0
    pending_scheduler_state = None
    resume_candidate = resume_from
    did_resume = False
    resume_same_stage = False
    ckpt_sharded = False
    resume_ds_state_dir = None
    ckpt_optimizer_state = None
    ckpt_scheduler_state = None
    ckpt_scaler_state = None

    if resume_candidate is None and os.path.exists(ckpt_latest_path):
        resume_candidate = ckpt_latest_path
        logger.info("Auto-resuming from %s", resume_candidate)
    if ds_enabled and resume_candidate is None and ds_state_dir and os.path.isdir(ds_state_dir):
        resume_ds_state_dir = ds_state_dir
    if resume_candidate:
        ckpt = None
        try:
            if os.path.isdir(resume_candidate):
                if ds_enabled:
                    resume_ds_state_dir = resume_candidate
                else:
                    raise ValueError(f"resume_from expects a .pt file, got directory: {resume_candidate}")
            else:
                ckpt = torch.load(resume_candidate, map_location="cpu")
            if ckpt is not None:
                model_state = ckpt.get("model", None)
                if model_state:
                    try:
                        materialized = materialize_mamba_time_embed_proj_from_state_dict(pipeline.unet, model_state)
                        if materialized > 0:
                            logger.info(
                                "Materialized Mamba time_embed_proj modules before resume load: %d",
                                materialized,
                            )
                        pipeline.unet.load_state_dict(model_state, strict=False)
                    except Exception as err:
                        logger.warning("Failed to load UNet state from checkpoint: %s", err)
                global_step = int(ckpt.get("global_step", 0))
                ckpt_epoch = int(ckpt.get("epoch", 0))
                ckpt_stage_idx = ckpt.get("stage_idx", None)
                ckpt_stage_name = ckpt.get("stage_name", None)
                ckpt_sharded = bool(ckpt.get("unet_sharded", False))
                if ds_enabled:
                    resume_ds_state_dir = (
                        resume_ds_state_dir
                        or ckpt.get("deepspeed_state_dir", None)
                        or ds_state_dir
                    )
                ckpt_mamba_fast = ckpt.get("effective_mamba_use_fast_path", None)
                ckpt_mamba_autotune = ckpt.get("effective_mamba_autotune_warmup", None)
                if resume_mamba_runtime_flags and (ckpt_mamba_fast is not None or ckpt_mamba_autotune is not None):
                    effective_mamba_use_fast_path = bool(
                        ckpt_mamba_fast if ckpt_mamba_fast is not None else effective_mamba_use_fast_path
                    )
                    effective_mamba_autotune_warmup = bool(
                        ckpt_mamba_autotune
                        if ckpt_mamba_autotune is not None
                        else effective_mamba_autotune_warmup
                    )
                    _set_mamba_env(effective_mamba_use_fast_path, effective_mamba_autotune_warmup)
                    updated = _apply_mamba_flags_inplace("resume")
                    logger.info(
                        "Resumed mamba effective flags from checkpoint (updated_modules=%d).",
                        updated,
                    )
                elif ckpt_mamba_fast is not None or ckpt_mamba_autotune is not None:
                    logger.info(
                        "Ignoring checkpoint mamba runtime flags; using config values "
                        "(use_fast_path=%s autotune_warmup=%s).",
                        effective_mamba_use_fast_path,
                        effective_mamba_autotune_warmup,
                    )
                if ckpt_stage_idx is not None:
                    try:
                        ckpt_stage_idx = int(ckpt_stage_idx)
                    except Exception:
                        ckpt_stage_idx = None
                resume_same_stage = ckpt_stage_idx == stage_idx
                if resume_same_stage:
                    start_epoch = ckpt_epoch + 1
                    if not ds_enabled:
                        ckpt_optimizer_state = ckpt.get("optimizer", None)
                        ckpt_scheduler_state = ckpt.get("scheduler", None)
                        ckpt_scaler_state = ckpt.get("scaler", None)
                else:
                    start_epoch = 1
                    logger.info(
                        "Checkpoint stage differs (ckpt_stage=%s name=%s); loading UNet only for stage %d.",
                        ckpt_stage_idx,
                        ckpt_stage_name,
                        stage_idx,
                    )
                did_resume = True
                logger.info(
                    "Resumed from %s (epoch=%d, global_step=%d, stage=%s)",
                    resume_candidate,
                    ckpt_epoch,
                    global_step,
                    ckpt_stage_idx,
                )
            elif resume_ds_state_dir:
                did_resume = True
        except Exception as err:
            logger.warning("Failed to resume from %s: %s. Starting fresh.", resume_candidate, err)
            did_resume = False
        finally:
            if ckpt is not None:
                ckpt_ref = ckpt
                ckpt = None
                cleanup_cuda("after_ckpt_extract", ckpt_ref)
    if ds_enabled and resume_ds_state_dir and not os.path.isdir(resume_ds_state_dir):
        logger.warning("DeepSpeed resume state dir not found: %s", resume_ds_state_dir)
        resume_ds_state_dir = None

    shard_applied = _run_stage_preflight()
    if preflight_crop_hw is not None:
        if crop_min_size != preflight_crop_hw or crop_max_size != preflight_crop_hw:
            logger.info(
                "Aligning training crop size to preflight crop size: %s -> %s",
                crop_min_size,
                preflight_crop_hw,
            )
        crop_min_size = preflight_crop_hw
        crop_max_size = preflight_crop_hw

    def _materialize_lazy_time_embed_proj_with_preflight() -> int:
        def _count_materialized_modules() -> tuple[int, int]:
            total = 0
            materialized = 0
            for module in pipeline.unet.modules():
                if isinstance(module, BiMambaSelfAttention):
                    total += 1
                    if isinstance(getattr(module, "time_embed_proj", None), torch.nn.Linear):
                        materialized += 1
            return materialized, total

        before = 0
        after = 0
        before, total = _count_materialized_modules()
        if total > 0 and before == total:
            return 0
        try:
            batch = _get_preflight_batch()
            with torch.no_grad():
                _ = compute_batch_loss(batch)
        except Exception as err:
            logger.warning("Failed to pre-materialize time_embed_proj via preflight: %s", err)
            return 0
        after, _ = _count_materialized_modules()
        created = max(after - before, 0)
        if created > 0:
            logger.info("Pre-materialized lazy time_embed_proj modules before optimizer init: %d", created)
        return created

    _materialize_lazy_time_embed_proj_with_preflight()

    # 学習対象パラメータのみ最適化
    trainable_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
    mamba_param_ids: set[int] = set()
    for module in pipeline.unet.modules():
        if isinstance(module, (MambaSpatioTemporalAdapter, BiMambaSelfAttention)):
            for p in module.parameters(recurse=True):
                if p.requires_grad:
                    mamba_param_ids.add(id(p))
    if mamba_learning_rate is not None and mamba_param_ids:
        base_params = [p for p in trainable_params if id(p) not in mamba_param_ids]
        mamba_params = [p for p in trainable_params if id(p) in mamba_param_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": stage_lr, "group_name": "base"},
                {"params": mamba_params, "lr": float(mamba_learning_rate), "group_name": "mamba"},
            ],
            lr=stage_lr,
            weight_decay=weight_decay,
            foreach=optimizer_foreach,
        )
        logger.info(
            "Optimizer param groups: base=%d (lr=%.2e), mamba=%d (lr=%.2e)",
            len(base_params),
            stage_lr,
            len(mamba_params),
            float(mamba_learning_rate),
        )
    else:
        if mamba_learning_rate is not None and not mamba_param_ids:
            logger.warning(
                "mamba_learning_rate set but no Mamba blocks found in UNet; using single learning_rate=%.2e.",
                stage_lr,
            )
        optimizer = torch.optim.AdamW(
            [{"params": trainable_params, "lr": stage_lr, "group_name": "base"}],
            lr=stage_lr,
            weight_decay=weight_decay,
            foreach=optimizer_foreach,
        )
    optimizer.zero_grad(set_to_none=True)
    lr_scheduler = None
    if sched_key == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)

    if ds_enabled and accelerator is not None:
        id2name = {id(p): n for n, p in pipeline.unet.named_parameters()}
        opt_param_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group.get("params", [])
        }
        bad = []
        for gi, group in enumerate(optimizer.param_groups):
            for param in group["params"]:
                if not param.requires_grad:
                    bad.append(
                        (
                            gi,
                            id2name.get(id(param), "<not_in_unet>"),
                            tuple(param.shape),
                            str(param.dtype),
                            str(param.device),
                        )
                    )
        print(f"[OPT CHECK] frozen_in_optimizer={len(bad)}", flush=True)
        for row in bad[:50]:
            print("[FROZEN OPT]", row, flush=True)
        if lr_scheduler is not None:
            pipeline.unet, optimizer, lr_scheduler = accelerator.prepare(
                pipeline.unet, optimizer, lr_scheduler
            )
        else:
            pipeline.unet, optimizer = accelerator.prepare(pipeline.unet, optimizer)
        pipeline.unet.train()
        trainable_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
        if debug_deepspeed_param_scan:
            if hasattr(torch.autograd.graph, "_get_grad_fn_or_grad_acc"):
                bad_params: list[tuple[str, str]] = []
                for name, param in pipeline.unet.named_parameters():
                    if not param.requires_grad:
                        continue
                    try:
                        node = torch.autograd.graph._get_grad_fn_or_grad_acc(param)
                        if node is None:
                            bad_params.append((name, "grad_fn_or_acc=None"))
                    except Exception as err:
                        is_inf = (
                            str(param.is_inference())
                            if hasattr(param, "is_inference")
                            else "no is_inference()"
                        )
                        bad_params.append((name, f"err={err} inference={is_inf}"))
                if bad_params:
                    logger.warning(
                        "DeepSpeed debug: %d params failed _get_grad_fn_or_grad_acc (first 10): %s",
                        len(bad_params),
                        bad_params[:10],
                    )
                else:
                    logger.info("DeepSpeed debug: all trainable params passed _get_grad_fn_or_grad_acc.")
        if hasattr(torch.autograd.graph, "_get_grad_fn_or_grad_acc"):
            if not getattr(torch.autograd.graph, "_stereocraft_grad_fn_wrapped", False):
                orig = torch.autograd.graph._get_grad_fn_or_grad_acc

                def wrapped(param: torch.Tensor):
                    try:
                        return orig(param)
                    except Exception as err:
                        name = id2name.get(id(param), "<unknown>")
                        is_inference = (
                            str(param.is_inference())
                            if hasattr(param, "is_inference")
                            else "no is_inference()"
                        )
                        print(
                            "[GRAD_FN FAIL]",
                            name,
                            "requires_grad=",
                            param.requires_grad,
                            "shape=",
                            tuple(param.shape),
                            "dtype=",
                            param.dtype,
                            "device=",
                            param.device,
                            "in_optimizer=",
                            id(param) in opt_param_ids,
                            "inference=",
                            is_inference,
                            "grad_enabled=",
                            torch.is_grad_enabled(),
                            "err=",
                            repr(err),
                            flush=True,
                        )
                        raise

                torch.autograd.graph._get_grad_fn_or_grad_acc = wrapped
                torch.autograd.graph._stereocraft_grad_fn_wrapped = True
            else:
                logger.warning("DeepSpeed debug: torch.autograd.graph._get_grad_fn_or_grad_acc unavailable.")
    if not ds_enabled:
        sharding_changed = bool(ckpt_sharded) != bool(shard_applied)
        should_restore_state = did_resume and resume_same_stage and not sharding_changed
        if should_restore_state:
            if ckpt_optimizer_state is not None:
                optimizer.load_state_dict(ckpt_optimizer_state)
            pending_scheduler_state = ckpt_scheduler_state
            if _use_scaler() and ckpt_scaler_state is not None:
                scaler.load_state_dict(ckpt_scaler_state)
        else:
            pending_scheduler_state = None
            if did_resume and resume_same_stage and sharding_changed:
                logger.info(
                    "Sharding state changed (ckpt_sharded=%s -> stage_sharded=%s); resetting optimizer/scheduler/scaler.",
                    ckpt_sharded,
                    shard_applied,
                )
    else:
        pending_scheduler_state = None
        if resume_same_stage and resume_ds_state_dir:
            try:
                accelerator.load_state(resume_ds_state_dir)
                logger.info("Restored DeepSpeed state from %s", resume_ds_state_dir)
            except Exception as err:
                logger.warning("Failed to restore DeepSpeed state: %s", err)

    def _apply_stage_lrs() -> None:
        base_lr = float(stage_lr)
        mamba_lr = float(mamba_learning_rate) if mamba_learning_rate is not None else None
        for group in optimizer.param_groups:
            group_name = str(group.get("group_name", "base"))
            if group_name == "mamba" and mamba_lr is not None:
                group["lr"] = mamba_lr
            else:
                group["lr"] = base_lr

    def _sync_scheduler_base_lrs() -> None:
        if lr_scheduler is None:
            return
        try:
            lr_scheduler.base_lrs = [group["lr"] for group in optimizer.param_groups]
        except Exception:
            pass

    # チェックポイントの保存
    def _save_full_checkpoint(tag: str, epoch_value: int, *, update_latest: bool = True) -> None:
        if ds_enabled and accelerator is not None:
            ds_dir = ds_state_dir if update_latest else os.path.join(save_dir, f"deepspeed_state_{tag}")
            model_state = None
            if is_main_process:
                try:
                    model_state = accelerator.get_state_dict(pipeline.unet)
                except Exception as err:
                    logger.warning("Failed to gather UNet state for checkpoint: %s", err)
            state = {
                "epoch": int(epoch_value),
                "global_step": int(global_step),
                "stage_idx": int(stage_idx),
                "stage_name": str(stage_name),
                "unet_sharded": bool(shard_applied),
                "effective_mamba_use_fast_path": bool(effective_mamba_use_fast_path),
                "effective_mamba_autotune_warmup": bool(effective_mamba_autotune_warmup),
                "mamba_gate_schedule": mamba_gate_schedule_key,
                "mamba_gate_start": float(mamba_gate_start),
                "mamba_gate_end": float(mamba_gate_end),
                "deepspeed_state_dir": ds_dir,
                "model": model_state,
            }
            path = os.path.join(save_dir, f"train_state_{tag}.pt")
            try:
                if is_main_process:
                    torch.save(state, path)
                    if update_latest:
                        torch.save(state, ckpt_latest_path)
                accelerator.wait_for_everyone()
                accelerator.save_state(ds_dir)
                accelerator.wait_for_everyone()
                logger.info("Saved DeepSpeed checkpoint (%s)", ds_dir)
            except Exception as err:
                logger.warning("Failed to save DeepSpeed checkpoint %s: %s", ds_dir, err)
            return
        state = {
            "epoch": int(epoch_value),
            "global_step": int(global_step),
            "stage_idx": int(stage_idx),
            "stage_name": str(stage_name),
            "unet_sharded": bool(shard_applied),
            "effective_mamba_use_fast_path": bool(effective_mamba_use_fast_path),
            "effective_mamba_autotune_warmup": bool(effective_mamba_autotune_warmup),
            "mamba_gate_schedule": mamba_gate_schedule_key,
            "mamba_gate_start": float(mamba_gate_start),
            "mamba_gate_end": float(mamba_gate_end),
            "model": pipeline.unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler": scaler.state_dict() if _use_scaler() else None,
        }
        path = os.path.join(save_dir, f"train_state_{tag}.pt")
        try:
            torch.save(state, path)
            if update_latest:
                torch.save(state, ckpt_latest_path)
            logger.info("Saved full checkpoint (%s)", path)
        except Exception as err:
            logger.warning("Failed to save checkpoint %s: %s", path, err)

    _apply_stage_lrs()
    _sync_scheduler_base_lrs()

    train_log_header = ["step", "epoch", "stage", "video", "timestep", "loss_noise_mse"]
    val_log_header = ["step", "epoch", "stage", "video", "timestep", "loss_noise_mse"]
    train_metric_keys = ["timestep", "loss_noise_mse"]
    val_metric_keys = ["timestep", "loss_noise_mse"]

    # Keep val header aligned with train for downstream tooling.
    val_log_header = list(train_log_header)
    val_metric_keys = list(train_metric_keys)

    def _format_metric(metrics: dict[str, torch.Tensor], key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return ""
        if key == "timestep":
            detached = value.detach().cpu()
            if torch.is_floating_point(detached):
                return f"{float(detached.float().item()):.8g}"
            return str(int(detached.long().item()))
        return f"{float(value.detach().float().cpu().item()):.6f}"

    # 簡易 CSV ログ (ステップごとの損失を記録)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    rank_log_suffix = ""
    if ds_enabled and accelerator is not None:
        rank_log_suffix = f"_rank{int(accelerator.process_index)}"
    csv_path, train_log_exists = select_log_path(
        save_dir,
        f"train_log{rank_log_suffix}",
        run_tag,
        reuse_existing=did_resume,
    )
    val_csv_path, val_log_exists = select_log_path(
        save_dir,
        f"val_log{rank_log_suffix}",
        run_tag,
        reuse_existing=did_resume,
    )
    trimmed_train = 0
    trimmed_val = 0
    manage_local_logs = bool(rank_log_suffix) or is_main_process
    if manage_local_logs:
        if train_log_exists and did_resume and resume_same_stage:
            trimmed_train = trim_train_log(csv_path, start_epoch, train_log_header)
        else:
            init_csv_log(csv_path, train_log_header)
        if val_log_exists and did_resume and resume_same_stage:
            trimmed_val = trim_val_log(val_csv_path, start_epoch, val_log_header)
        else:
            init_csv_log(val_csv_path, val_log_header)
        if did_resume and resume_same_stage and (trimmed_train or trimmed_val):
            logger.info(
                "Trimmed log rows for epochs >= %d (train=%d, val=%d) for local rank log.",
                start_epoch,
                trimmed_train,
                trimmed_val,
            )
    logger.info("Log files for this run: train=%s val=%s", csv_path, val_csv_path)
    diag_interval = int(max(0, mamba_diag_interval))
    mamba_diag_header = [
        "step",
        "epoch",
        "stage",
        "video",
        "batch",
        "module",
        "did_step",
        "module_grad_norm",
        "module_param_norm",
        "time_grad_norm",
        "time_param_norm_pre",
        "time_param_norm_post",
        "time_param_norm_delta_ratio_step",
        "time_param_norm_delta_ratio_interlog",
        "time_param_norm_delta_ratio",
        "time_weight_abs_mean",
        "time_weight_nonzero_frac",
        "time_bias_abs_mean",
    ]
    mamba_diag_csv_path = ""
    if diag_interval > 0:
        mamba_diag_csv_path, mamba_diag_exists = select_log_path(
            save_dir,
            f"mamba_diag{rank_log_suffix}",
            run_tag,
            reuse_existing=did_resume,
        )
        if not mamba_diag_exists:
            init_csv_log(mamba_diag_csv_path, mamba_diag_header)
        logger.info("Mamba diagnostics enabled: interval=%d, file=%s", diag_interval, mamba_diag_csv_path)

    denoise_diag_every = int(max(0, denoise_diag_interval))
    denoise_diag_header = [
        "step",
        "epoch",
        "stage",
        "video",
        "batch",
        "timestep",
        "sigma",
        "loss_noise_mse",
        "pred_target_cos",
    ]
    for _prefix in ("x0", "eps", "x_t", "target", "noise_pred", "pred_err"):
        for _stat in ("mean", "std", "rms", "abs_mean", "max_abs", "finite_frac"):
            denoise_diag_header.append(f"{_prefix}_{_stat}")
    denoise_diag_csv_path = ""
    if denoise_diag_every > 0:
        denoise_diag_csv_path, denoise_diag_exists = select_log_path(
            save_dir,
            f"denoise_diag{rank_log_suffix}",
            run_tag,
            reuse_existing=did_resume,
        )
        if not denoise_diag_exists:
            init_csv_log(denoise_diag_csv_path, denoise_diag_header)
        logger.info(
            "Denoise diagnostics enabled: interval=%d, file=%s",
            denoise_diag_every,
            denoise_diag_csv_path,
        )

    image_diag_every = int(max(0, image_diag_interval))
    image_diag_header = [
        "step",
        "epoch",
        "stage",
        "video",
        "batch",
        "timestep",
        "sigma",
        "loss_noise_mse",
        "image_diag_mse",
        "image_diag_l1",
        "image_diag_psnr",
        "image_diag_mask_mse",
        "image_diag_mask_l1",
        "image_diag_mask_psnr",
    ]
    image_diag_csv_path = ""
    if image_diag_every > 0:
        if diffusion_scheduler_key != "euler":
            logger.warning("image_diag_interval is currently supported for Euler training only; no image rows may be emitted.")
        image_diag_csv_path, image_diag_exists = select_log_path(
            save_dir,
            f"image_diag{rank_log_suffix}",
            run_tag,
            reuse_existing=did_resume,
        )
        if not image_diag_exists:
            init_csv_log(image_diag_csv_path, image_diag_header)
        logger.info(
            "Image diagnostics enabled: interval=%d, max_frames=%d, decode_chunk_size=%d, file=%s",
            image_diag_every,
            int(image_diag_max_frames),
            int(image_diag_decode_chunk_size),
            image_diag_csv_path,
        )

    def _get_unwrapped_unet() -> torch.nn.Module:
        if accelerator is not None:
            try:
                return accelerator.unwrap_model(pipeline.unet)
            except Exception:
                pass
        return pipeline.unet

    def _iter_mamba_modules() -> list[tuple[str, BiMambaSelfAttention]]:
        model = _get_unwrapped_unet()
        out: list[tuple[str, BiMambaSelfAttention]] = []
        for name, module in model.named_modules():
            if isinstance(module, BiMambaSelfAttention):
                out.append((name, module))
        return out

    def _iter_gated_mamba_modules() -> list[tuple[str, GatedResidualMambaSelfAttention]]:
        model = _get_unwrapped_unet()
        out: list[tuple[str, GatedResidualMambaSelfAttention]] = []
        for name, module in model.named_modules():
            if isinstance(module, GatedResidualMambaSelfAttention):
                out.append((name, module))
        return out

    gated_module_count = len(_iter_gated_mamba_modules())
    if gated_module_count > 0:
        logger.info(
            "Gated residual Mamba enabled: modules=%d schedule=%s start=%.3f end=%.3f",
            gated_module_count,
            mamba_gate_schedule_key,
            mamba_gate_start,
            mamba_gate_end,
        )

    def _scheduled_mamba_gate(epoch_value: int, batch_value: int, batches_total: int) -> float:
        if mamba_gate_schedule_key == "none":
            return mamba_gate_start
        total_epochs = max(int(planned_epochs_total - start_epoch + 1), 1)
        epoch_offset = max(int(epoch_value - start_epoch), 0)
        batch_frac = 0.0
        if batches_total > 1:
            batch_frac = max(0.0, min(1.0, float(batch_value - 1) / float(batches_total - 1)))
        elif batches_total == 1:
            batch_frac = 1.0
        progress = (float(epoch_offset) + batch_frac) / float(total_epochs)
        progress = max(0.0, min(1.0, progress))
        return mamba_gate_start + (mamba_gate_end - mamba_gate_start) * progress

    def _apply_scheduled_mamba_gate(epoch_value: int, batch_value: int, batches_total: int) -> float:
        if gated_module_count <= 0:
            return 1.0
        gate = _scheduled_mamba_gate(epoch_value, batch_value, batches_total)
        disable_reference = gate >= 0.999
        updated = set_gated_mamba_gate(_get_unwrapped_unet(), gate, disable_reference=disable_reference)
        if (
            mamba_gate_log_interval > 0
            and (global_step + 1) % mamba_gate_log_interval == 0
            and is_local_main
        ):
            logger.info(
                "Mamba gate schedule: step=%d epoch=%d batch=%d/%d gate=%.4f disable_reference=%s updated=%d",
                global_step + 1,
                epoch_value,
                batch_value,
                batches_total,
                gate,
                disable_reference,
                updated,
            )
        if writer_tb:
            writer_tb.add_scalar("mamba_gate/value", gate, global_step + 1)
        return gate

    def _norm2(value: float) -> str:
        return f"{value:.8e}"

    def _compute_param_norm(module: torch.nn.Module) -> float:
        sq = 0.0
        for param in module.parameters(recurse=True):
            data = param.detach()
            sq += float(data.float().pow(2).sum().item())
        return math.sqrt(max(sq, 0.0))

    def _compute_grad_norm(module: torch.nn.Module) -> float:
        sq = 0.0
        for param in module.parameters(recurse=True):
            grad = param.grad
            if grad is None:
                continue
            sq += float(grad.detach().float().pow(2).sum().item())
        return math.sqrt(max(sq, 0.0))

    def _safe_row_write(path: str, row: list[str]) -> None:
        if not path:
            return
        with open(path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(row)

    def _metric_float(metrics: dict[str, torch.Tensor], key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return ""
        return f"{float(value.detach().float().cpu().item()):.8e}"

    def _log_denoise_diag(
        *,
        step_value: int,
        epoch_value: int,
        video_name_value: str,
        batch_value: int,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        if denoise_diag_every <= 0 or not denoise_diag_csv_path:
            return
        if step_value % denoise_diag_every != 0:
            return
        row = [
            str(step_value),
            str(epoch_value),
            stage_name,
            video_name_value,
            str(batch_value),
            _metric_float(metrics, "timestep") if "timestep" in metrics else "",
            _metric_float(metrics, "sigma"),
            _metric_float(metrics, "loss_noise_mse"),
            _metric_float(metrics, "pred_target_cos"),
        ]
        for prefix in ("x0", "eps", "x_t", "target", "noise_pred", "pred_err"):
            for stat in ("mean", "std", "rms", "abs_mean", "max_abs", "finite_frac"):
                row.append(_metric_float(metrics, f"{prefix}_{stat}"))
        _safe_row_write(denoise_diag_csv_path, row)

    def _log_image_diag(
        *,
        step_value: int,
        epoch_value: int,
        video_name_value: str,
        batch_value: int,
        metrics: dict[str, torch.Tensor],
    ) -> None:
        if image_diag_every <= 0 or not image_diag_csv_path:
            return
        if step_value % image_diag_every != 0:
            return
        row = [
            str(step_value),
            str(epoch_value),
            stage_name,
            video_name_value,
            str(batch_value),
            _metric_float(metrics, "timestep") if "timestep" in metrics else "",
            _metric_float(metrics, "sigma"),
            _metric_float(metrics, "loss_noise_mse"),
            _metric_float(metrics, "image_diag_mse"),
            _metric_float(metrics, "image_diag_l1"),
            _metric_float(metrics, "image_diag_psnr"),
            _metric_float(metrics, "image_diag_mask_mse"),
            _metric_float(metrics, "image_diag_mask_l1"),
            _metric_float(metrics, "image_diag_mask_psnr"),
        ]
        _safe_row_write(image_diag_csv_path, row)

    # Keep previous logged weights to detect delayed updates (e.g. wrapped optimizers).
    _last_logged_time_weight_cpu: dict[str, torch.Tensor] = {}
    _audit_add_group_unsupported_logged = False
    _audit_add_group_disabled = False

    def _collect_mamba_diag_snapshot() -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for name, module in _iter_mamba_modules():
            entry: dict[str, Any] = {
                "module_grad_norm": _compute_grad_norm(module),
                "module_param_norm": _compute_param_norm(module),
                "time_grad_norm": 0.0,
                "time_param_norm_pre": 0.0,
                "time_weight_pre_cpu": None,
                "time_weight_abs_mean": 0.0,
                "time_weight_nonzero_frac": 0.0,
                "time_bias_abs_mean": 0.0,
            }
            proj = getattr(module, "time_embed_proj", None)
            if isinstance(proj, torch.nn.Linear):
                if proj.weight.grad is not None:
                    entry["time_grad_norm"] = float(proj.weight.grad.detach().float().norm().item())
                weight = proj.weight.detach().float()
                entry["time_param_norm_pre"] = float(weight.norm().item())
                entry["time_weight_pre_cpu"] = weight.cpu().clone()
                entry["time_weight_abs_mean"] = float(weight.abs().mean().item())
                entry["time_weight_nonzero_frac"] = float((weight != 0).float().mean().item())
                if proj.bias is not None:
                    entry["time_bias_abs_mean"] = float(proj.bias.detach().float().abs().mean().item())
            snapshot[name] = entry
        return snapshot

    def _unwrap_optimizer_for_groups(opt: Any) -> Any:
        cur = opt
        seen: set[int] = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if hasattr(cur, "param_groups"):
                return cur
            next_opt = None
            for attr in ("optimizer", "optim", "_optimizer"):
                if hasattr(cur, attr):
                    next_opt = getattr(cur, attr)
                    break
            cur = next_opt
        return None

    def _audit_and_fix_time_proj_optimizer_registration(tag: str) -> tuple[int, int]:
        nonlocal _audit_add_group_unsupported_logged, _audit_add_group_disabled
        if not diag_interval:
            return 0, 0
        if _audit_add_group_disabled:
            return 0, 0
        try:
            opt_ref = _unwrap_optimizer_for_groups(optimizer)
        except Exception:
            opt_ref = None
        if opt_ref is None:
            return 0, 0

        param_ids = {
            id(param)
            for group in opt_ref.param_groups
            for param in group.get("params", [])
            if isinstance(param, torch.nn.Parameter)
        }
        mamba_group = None
        fallback_group = opt_ref.param_groups[0] if opt_ref.param_groups else None
        for group in opt_ref.param_groups:
            if str(group.get("group_name", "")).strip().lower() == "mamba":
                mamba_group = group
                break
            if str(group.get("group_name", "")).strip().lower() == "base" and fallback_group is None:
                fallback_group = group
        template_group = mamba_group or fallback_group
        if template_group is None:
            return 0, 0
        add_param_group_fn = getattr(opt_ref, "add_param_group", None)
        can_add_param_group = callable(add_param_group_fn)
        if not can_add_param_group:
            if not _audit_add_group_unsupported_logged:
                logger.info(
                    "Skipping dynamic optimizer param-group patch (%s): optimizer type %s does not support add_param_group.",
                    tag,
                    type(opt_ref).__name__,
                )
                _audit_add_group_unsupported_logged = True
            return 0, 0

        total_missing = 0
        total_added = 0
        copied_keys = [
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "foreach",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
        ]
        for module_name, module in _iter_mamba_modules():
            proj = getattr(module, "time_embed_proj", None)
            if not isinstance(proj, torch.nn.Linear):
                continue
            missing_params: list[torch.nn.Parameter] = []
            for param in (proj.weight, proj.bias):
                if param is None or not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
                    continue
                if id(param) not in param_ids:
                    missing_params.append(param)
            if not missing_params:
                continue
            total_missing += len(missing_params)
            group_payload: dict[str, Any] = {"params": missing_params, "group_name": "mamba_late_time_proj"}
            for key in copied_keys:
                if key in template_group:
                    group_payload[key] = template_group[key]
            try:
                add_param_group_fn(group_payload)
                total_added += len(missing_params)
                for param in missing_params:
                    param_ids.add(id(param))
                logger.warning(
                    "Added %d missing time_embed_proj params to optimizer (%s): module=%s",
                    len(missing_params),
                    tag,
                    module_name,
                )
            except Exception as err:
                err_msg = str(err)
                # DeepSpeed wrappers can expose callable add_param_group but fail internally.
                if isinstance(err, AttributeError) or "add_param_group" in err_msg:
                    _audit_add_group_disabled = True
                    if not _audit_add_group_unsupported_logged:
                        logger.info(
                            "Disabling dynamic optimizer param-group patch (%s): optimizer type %s is not add_param_group-compatible (%s).",
                            tag,
                            type(opt_ref).__name__,
                            err_msg,
                        )
                        _audit_add_group_unsupported_logged = True
                    return total_missing, total_added
                logger.warning(
                    "Failed to add %d missing time_embed_proj params (%s): module=%s err=%s",
                    len(missing_params),
                    tag,
                    module_name,
                    err,
                )
        return total_missing, total_added

    def _log_mamba_diag(
        *,
        step_value: int,
        epoch_value: int,
        video_name_value: str,
        batch_value: int,
        did_step: bool,
        pre_snapshot: dict[str, dict[str, float]],
    ) -> None:
        if diag_interval <= 0 or not mamba_diag_csv_path:
            return
        if step_value % diag_interval != 0:
            return
        if not pre_snapshot:
            return
        _audit_and_fix_time_proj_optimizer_registration(f"diag_step_{step_value}")
        post_time_norms: dict[str, float] = {}
        post_time_weights: dict[str, torch.Tensor] = {}
        if did_step:
            for name, module in _iter_mamba_modules():
                proj = getattr(module, "time_embed_proj", None)
                if isinstance(proj, torch.nn.Linear):
                    post_weight = proj.weight.detach().float().cpu()
                    post_time_weights[name] = post_weight
                    post_time_norms[name] = float(post_weight.norm().item())
        for name, entry in pre_snapshot.items():
            pre_norm = float(entry.get("time_param_norm_pre", 0.0))
            post_norm = float(post_time_norms.get(name, pre_norm))
            delta_ratio_step = 0.0
            if did_step:
                pre_weight_cpu = entry.get("time_weight_pre_cpu")
                post_weight_cpu = post_time_weights.get(name)
                if isinstance(pre_weight_cpu, torch.Tensor) and isinstance(post_weight_cpu, torch.Tensor):
                    if pre_weight_cpu.shape == post_weight_cpu.shape:
                        diff_norm = float((post_weight_cpu - pre_weight_cpu).norm().item())
                        delta_ratio_step = diff_norm / max(pre_norm, 1e-12)

            delta_ratio_interlog = 0.0
            prev_weight_cpu = _last_logged_time_weight_cpu.get(name)
            cur_weight_cpu = post_time_weights.get(name)
            if cur_weight_cpu is None:
                cur_weight_cpu = entry.get("time_weight_pre_cpu")
            if did_step and isinstance(prev_weight_cpu, torch.Tensor) and isinstance(cur_weight_cpu, torch.Tensor):
                if prev_weight_cpu.shape == cur_weight_cpu.shape:
                    prev_norm = float(prev_weight_cpu.norm().item())
                    inter_diff_norm = float((cur_weight_cpu - prev_weight_cpu).norm().item())
                    delta_ratio_interlog = inter_diff_norm / max(prev_norm, 1e-12)

            delta_ratio = max(delta_ratio_step, delta_ratio_interlog)
            row = [
                str(step_value),
                str(epoch_value),
                stage_name,
                video_name_value,
                str(batch_value),
                name,
                "1" if did_step else "0",
                _norm2(float(entry.get("module_grad_norm", 0.0))),
                _norm2(float(entry.get("module_param_norm", 0.0))),
                _norm2(float(entry.get("time_grad_norm", 0.0))),
                _norm2(pre_norm),
                _norm2(post_norm),
                _norm2(delta_ratio_step),
                _norm2(delta_ratio_interlog),
                _norm2(delta_ratio),
                _norm2(float(entry.get("time_weight_abs_mean", 0.0))),
                _norm2(float(entry.get("time_weight_nonzero_frac", 0.0))),
                _norm2(float(entry.get("time_bias_abs_mean", 0.0))),
            ]
            _safe_row_write(mamba_diag_csv_path, row)
            if isinstance(cur_weight_cpu, torch.Tensor):
                _last_logged_time_weight_cpu[name] = cur_weight_cpu.clone()

    missing_optimizer_params, added_optimizer_params = _audit_and_fix_time_proj_optimizer_registration(
        "post_optimizer_init"
    )
    if missing_optimizer_params > 0:
        logger.info(
            "time_embed_proj optimizer audit: missing=%d added=%d",
            missing_optimizer_params,
            added_optimizer_params,
        )

    # (オプション) TensorBoard ログ
    writer_tb = None
    if tensorboard_log_dir and is_main_process:
        if SummaryWriter is None:
            logger.warning(
                "tensorboard package not available. Install it with `pip install tensorboard` to enable TensorBoard logging."
            )
        else:
            log_dir = tensorboard_log_dir if os.path.isabs(tensorboard_log_dir) else os.path.join(save_dir, tensorboard_log_dir)
            os.makedirs(log_dir, exist_ok=True)
            writer_tb = SummaryWriter(log_dir=log_dir)

    # 入力動画パスをグロブから列挙し、必要なら分割
    all_video_paths = sorted(glob.glob(train_glob))
    if not all_video_paths:
        raise FileNotFoundError(f"No training videos found for pattern: {train_glob}")
    valid_keys = ("train", "val", "test")
    split_key = (dataset_split_group or "train").strip().lower()
    if split_key not in valid_keys:
        raise ValueError(f"dataset_split_group must be one of {valid_keys}, got '{dataset_split_group}'.")
    split_map: dict[str, list[str]] = {key: [] for key in valid_keys}
    split_map["train"] = list(all_video_paths)
    if dataset_split_ratios:
        ratios = [float(value) for value in dataset_split_ratios]
        if len(ratios) != 3:
            raise ValueError("dataset_split_ratios must contain exactly three values: [train, val, test].")
        if any(value < 0 for value in ratios):
            raise ValueError("dataset_split_ratios cannot contain negative values.")
        ratio_sum = sum(ratios)
        if ratio_sum <= 0:
            raise ValueError("dataset_split_ratios must sum to a positive value.")
        shuffled = list(all_video_paths)
        random.Random(dataset_split_seed).shuffle(shuffled)
        total_videos = len(shuffled)
        normalized = [ratio / ratio_sum for ratio in ratios]
        raw_counts = [norm * total_videos for norm in normalized]
        counts = [math.floor(value) for value in raw_counts]
        remainder = total_videos - sum(counts)
        if remainder > 0:
            fractional_order = sorted(
                range(len(raw_counts)),
                key=lambda idx: (raw_counts[idx] - counts[idx]),
                reverse=True,
            )
            for idx in fractional_order[:remainder]:
                counts[idx] += 1
        split_map: dict[str, list[str]] = {name: [] for name in valid_keys}
        cursor = 0
        for key, count in zip(valid_keys, counts):
            if count > 0:
                split_map[key] = shuffled[cursor : cursor + count]
            cursor += count

        selected = split_map[split_key]
        if not selected:
            raise ValueError(
                f"No videos assigned to split '{dataset_split_group}'. "
                f"Ratios={ratios}, total_videos={total_videos}"
            )
        logger.info(
            "Dataset split ratios %s (seed=%d) -> counts train=%d val=%d test=%d",
            ratios,
            dataset_split_seed,
            len(split_map["train"]),
            len(split_map["val"]),
            len(split_map["test"]),
        )
        preview = ", ".join(os.path.basename(path) for path in selected[:3])
        if preview:
            extra = "..." if len(selected) > 3 else ""
            logger.info("Using '%s' split with %d videos (e.g., %s%s)", split_key, len(selected), preview, extra)
    else:
        if split_key != "train":
            raise ValueError(
                "dataset_split_group other than 'train' requires dataset_split_ratios to define the split sizes."
            )
        video_sample = ", ".join(os.path.basename(path) for path in split_map["train"][:3])
        if video_sample:
            extra = "..." if len(split_map["train"]) > 3 else ""
            logger.info(
                "Using %d videos matched by %s (e.g., %s%s)",
                len(split_map["train"]),
                train_glob,
                video_sample,
                extra,
            )
    video_paths = split_map[split_key]
    train_batches = None
    last_batch = None

    cleanup_tag = (
        f"stage{stage_idx}->stage{stage_idx + 1}" if stage_idx < 3 else f"stage{stage_idx}->end"
    )

    def _cleanup_stage_resources(tag: str) -> None:
        nonlocal pipeline
        nonlocal optimizer
        nonlocal lr_scheduler
        nonlocal writer_tb
        nonlocal trainable_params
        nonlocal video_paths
        nonlocal train_batches
        nonlocal last_batch
        nonlocal split_map
        nonlocal all_video_paths
        nonlocal preflight_batch
        nonlocal pending_scheduler_state
        nonlocal ckpt_optimizer_state
        nonlocal ckpt_scheduler_state
        nonlocal ckpt_scaler_state
        unet_ref = None
        try:
            unet_ref = pipeline.unet
        except Exception:
            pass
        pipeline_ref = pipeline
        optimizer_ref = optimizer
        lr_scheduler_ref = lr_scheduler
        writer_ref = writer_tb
        trainable_ref = trainable_params
        video_paths_ref = video_paths
        train_batches_ref = train_batches
        last_batch_ref = last_batch
        split_map_ref = split_map
        all_video_paths_ref = all_video_paths
        preflight_ref = preflight_batch
        pending_sched_ref = pending_scheduler_state
        ckpt_opt_ref = ckpt_optimizer_state
        ckpt_sched_ref = ckpt_scheduler_state
        ckpt_scaler_ref = ckpt_scaler_state

        pipeline = None
        optimizer = None
        lr_scheduler = None
        writer_tb = None
        trainable_params = None
        video_paths = None
        train_batches = None
        last_batch = None
        split_map = None
        all_video_paths = None
        preflight_batch = None
        pending_scheduler_state = None
        ckpt_optimizer_state = None
        ckpt_scheduler_state = None
        ckpt_scaler_state = None

        cleanup_cuda(
            tag,
            pipeline_ref,
            unet_ref,
            optimizer_ref,
            lr_scheduler_ref,
            writer_ref,
            trainable_ref,
            video_paths_ref,
            train_batches_ref,
            last_batch_ref,
            split_map_ref,
            all_video_paths_ref,
            preflight_ref,
            pending_sched_ref,
            ckpt_opt_ref,
            ckpt_sched_ref,
            ckpt_scaler_ref,
        )
        if ds_enabled and accelerator is not None:
            try:
                accelerator.free_memory(pipeline_ref, optimizer_ref, lr_scheduler_ref)
            except Exception:
                pass

    if preflight_only:
        logger.info(
            "Preflight-only completed for Stage %d/3 (%s); skipping epoch training.",
            stage_idx,
            stage_name,
        )
        if ds_enabled and accelerator is not None:
            accelerator.wait_for_everyone()
        _cleanup_stage_resources(f"preflight_stage{stage_idx}")
        return shard_applied

    def run_validation(split_name: str, epoch_idx: int) -> float | None:
        """Run a full forward pass over the requested split and log average loss."""
        val_key = (split_name or "").strip().lower()
        if not val_key:
            return None
        if val_key not in split_map:
            logger.warning("Unknown val split '%s'; skipping validation.", split_name)
            return None
        val_video_paths_all = split_map[val_key]
        if not val_video_paths_all:
            logger.warning("No videos available for val split '%s'; skipping.", val_key)
            return None
        max_videos = max_val_videos if (max_val_videos or 0) > 0 else None
        total_assigned = len(val_video_paths_all)
        if max_videos is not None and max_videos < total_assigned:
            val_video_paths_all = val_video_paths_all[:max_videos]

        val_video_paths_local = val_video_paths_all
        dist_enabled = (
            ds_enabled
            and accelerator is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if dist_enabled:
            buckets, totals = _assign_videos_balanced(val_video_paths_all, accelerator.num_processes)
            val_video_paths_local = buckets[accelerator.process_index]
            if is_main_process:
                totals_str = ", ".join(str(total) for total in totals)
                if max_videos is not None and max_videos < total_assigned:
                    logger.info(
                        "Validating split '%s' on %d/%d videos (limited by max_val_videos).",
                        val_key,
                        len(val_video_paths_all),
                        total_assigned,
                    )
                else:
                    logger.info("Validating split '%s' on %d videos.", val_key, total_assigned)
                logger.info("Validation chunk assignment across ranks: %s", totals_str)
            logger.info(
                "Rank %s validating split '%s' on %d videos.",
                accelerator.process_index,
                val_key,
                len(val_video_paths_local),
            )
        else:
            if max_videos is not None and max_videos < total_assigned:
                logger.info(
                    "Validating split '%s' on %d/%d videos (limited by max_val_videos).",
                    val_key,
                    len(val_video_paths_all),
                    total_assigned,
                )
            else:
                logger.info("Validating split '%s' on %d videos.", val_key, total_assigned)
        was_training = pipeline.unet.training
        pipeline.unet.eval()
        total_loss = 0.0
        total_batches = 0
        val_local = 0
        try:
            with torch.no_grad():
                writer = None
                val_f = None
                if val_csv_path:
                    val_f = open(val_csv_path, "a", encoding="utf-8", newline="")
                    writer = csv.writer(val_f)
                try:
                    for video_idx, video_path in enumerate(val_video_paths_local, start=1):
                        if stop_event.is_set():
                            raise KeyboardInterrupt
                        batches = prepare_batches(
                            video_path,
                            frames_chunk=frames_chunk,
                            overlap=overlap,
                            device=device,
                            dtype=torch_dtype if precision_key != "fp32" else torch.float32,
                            crop_multiple=crop_multiple,
                            crop_min_size=crop_min_size,
                            crop_max_size=crop_max_size,
                            random_crop=random_crop_per_chunk,
                            use_prev_target_overlap=use_prev_target_overlap,
                            overlap_teacher_prob=overlap_teacher_prob,
                            overlap_noise_std=overlap_noise_std,
                        )
                        for batch_i, batch in enumerate(batches, start=1):
                            if stop_event.is_set():
                                raise KeyboardInterrupt
                            loss_raw, metrics = compute_batch_loss(batch)
                            batch_loss_val = float(loss_raw.detach().item())
                            total_loss += batch_loss_val
                            total_batches += 1
                            val_local += 1
                            if writer is not None:
                                writer.writerow(
                                    [
                                        global_step + val_local,
                                        epoch_idx,
                                        stage_name,
                                        f"{val_key}/{os.path.basename(video_path)}",
                                    ]
                                    + [_format_metric(metrics, key) for key in val_metric_keys]
                                )
                finally:
                    if val_f is not None:
                        val_f.close()
        finally:
            if was_training:
                pipeline.unet.train()
        if dist_enabled:
            stats = torch.tensor([total_loss, float(total_batches)], device=device, dtype=torch.float64)
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            total_loss = float(stats[0].item())
            total_batches = int(round(float(stats[1].item())))
        if total_batches == 0:
            if (not dist_enabled) or is_main_process:
                logger.warning("Validation split '%s' produced zero batches.", val_key)
            return None
        avg_loss = total_loss / total_batches
        if (not dist_enabled) or is_main_process:
            logger.info(
                "Val split '%s' epoch %d: avg_loss=%.6f over %d batches.",
                val_key,
                epoch_idx,
                avg_loss,
                total_batches,
            )
            if writer_tb:
                writer_tb.add_scalar(f"loss/{val_key}_avg", avg_loss, epoch_idx)
        return avg_loss

    printer = TrainingProgressPrinter(
        device=device,
        log_interval=log_interval,
        enable_mem=True,
        logger_obj=logger,
    )
    def _dist_min(value: int) -> int:
        if not (ds_enabled and accelerator is not None):
            return value
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value
        tensor = torch.tensor([value], device=device, dtype=torch.int64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return int(tensor.item())
    def _dist_max(value: int) -> int:
        if not (ds_enabled and accelerator is not None):
            return value
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value
        tensor = torch.tensor([value], device=device, dtype=torch.int64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return int(tensor.item())

    try:
        accum_counter = 0
        # stage_epochs を上限にし、target_avg_loss は早期終了のみに使用
        planned_epochs_total = stage_epochs
        if sched_key in {"cosine", "cosine_with_warmup"} and lr_scheduler is None:
            t_max = scheduler_t_max if scheduler_t_max > 0 else planned_epochs_total
            t_max = max(int(t_max), 1)
            eta_min = max(scheduler_eta_min, 0.0)
            if sched_key == "cosine":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=t_max, eta_min=eta_min
                )
            else:
                warmup_steps = max(int(num_warmup_steps), 0)
                warmup_steps = min(warmup_steps, t_max)
                if warmup_steps == 0:
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=t_max, eta_min=eta_min
                    )
                else:
                    base_lrs = [group["lr"] for group in optimizer.param_groups]
                    base_lr = max(base_lrs[0], 0.0) if base_lrs else 0.0
                    eta_ratio = (eta_min / base_lr) if base_lr > 0 else 0.0
                    eta_ratio = min(max(eta_ratio, 0.0), 1.0)
                    cosine_steps = max(1, t_max - warmup_steps)

                    def lr_lambda(step_idx: int) -> float:
                        if step_idx < warmup_steps - 1:
                            factor = (step_idx + 2) / float(warmup_steps)
                            return max(factor, eta_ratio)
                        t = step_idx - (warmup_steps - 1)
                        cosine = (1.0 + math.cos(math.pi * t / cosine_steps)) / 2.0
                        return eta_ratio + (1.0 - eta_ratio) * cosine

                    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=lr_lambda
                    )
                    if start_epoch == 1 and pending_scheduler_state is None:
                        initial_factor = max(1.0 / float(warmup_steps), eta_ratio)
                        for group, lr in zip(optimizer.param_groups, base_lrs):
                            group["lr"] = lr * initial_factor
        if lr_scheduler is not None and pending_scheduler_state is not None and resume_same_stage:
            try:
                lr_scheduler.load_state_dict(pending_scheduler_state)
                logger.info("Restored scheduler state from checkpoint.")
            except Exception as err:
                logger.warning("Failed to restore scheduler state: %s", err)
            pending_scheduler_state = None
            _apply_stage_lrs()
            _sync_scheduler_base_lrs()
        if start_epoch > planned_epochs_total:
            logger.info(
                "Checkpoint epoch (%d) >= planned total epochs (%d). Nothing to do.",
                start_epoch - 1,
                planned_epochs_total,
            )
            _cleanup_stage_resources(cleanup_tag)
            return shard_applied
        current_epoch = start_epoch - 1
        logged_loss_meta = False
        for epoch in range(start_epoch, planned_epochs_total + 1):
            current_epoch = epoch
            # エポックごとに動画順をシャッフル
            random.shuffle(video_paths)
            epoch_video_paths = video_paths
            if ds_enabled and accelerator is not None:
                buckets, totals = _assign_videos_balanced(video_paths, accelerator.num_processes)
                epoch_video_paths = buckets[accelerator.process_index]
                if accelerator.is_main_process:
                    totals_str = ", ".join(str(total) for total in totals)
                    logger.info("Balanced chunk assignment across ranks: %s", totals_str)
                shared_video_count = _dist_max(len(epoch_video_paths))
                if epoch_video_paths and len(epoch_video_paths) < shared_video_count:
                    pad = shared_video_count - len(epoch_video_paths)
                    epoch_video_paths = epoch_video_paths + [epoch_video_paths[-1]] * pad
            epoch_loss = 0.0
            epoch_batches = 0
            printer.start_epoch(epoch_idx=epoch, epochs_total=planned_epochs_total, videos_total=len(epoch_video_paths))

            for video_idx, video_path in enumerate(epoch_video_paths, start=1):
                if stop_event.is_set():
                    raise KeyboardInterrupt
                rank_label = os.environ.get("LOCAL_RANK", "0")
                logger.info(
                    "Rank %s video %d/%d: %s",
                    rank_label,
                    video_idx,
                    len(epoch_video_paths),
                    os.path.basename(video_path),
                )
                # 動画を時間方向にチャンク分割し、GPU 上に順次ロード
                train_batches = prepare_batches(
                    video_path,
                    frames_chunk=frames_chunk,
                    overlap=overlap,
                    device=device,
                    dtype=torch_dtype if precision_key != "fp32" else torch.float32,
                    crop_multiple=crop_multiple,
                    crop_min_size=crop_min_size,
                    crop_max_size=crop_max_size,
                    random_crop=random_crop_per_chunk,
                    use_prev_target_overlap=use_prev_target_overlap,
                    overlap_teacher_prob=overlap_teacher_prob,
                    overlap_noise_std=overlap_noise_std,
                    target_override_video_path=target_override_video_path,
                    target_override_is_sbs=target_override_is_sbs,
                )
                local_batch_count = len(train_batches)
                shared_batch_limit = _dist_max(local_batch_count)
                printer.start_video(video_idx=video_idx, batches_total=shared_batch_limit)
                batch_iter = iter(train_batches)

                for batch_i in range(1, shared_batch_limit + 1):
                    if stop_event.is_set():
                        raise KeyboardInterrupt
                    try:
                        batch = next(batch_iter)
                    except StopIteration:
                        if last_batch is None:
                            last_batch = _get_preflight_batch()
                        batch = last_batch
                    else:
                        last_batch = batch

                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                        log_vram_usage(
                            f"VRAM before processing video {video_idx} batch {batch_i}",
                            device,
                            level=logging.DEBUG,
                        )

                    is_last_batch_epoch = (video_idx == len(epoch_video_paths)) and (batch_i == len(train_batches))
                    # ===== ランダムtの通常学習: 1回のUNet前向きでノイズ予測MSE =====
                    current_mamba_gate = _apply_scheduled_mamba_gate(epoch, batch_i, shared_batch_limit)
                    if ds_enabled and accelerator is not None:
                        skip_update = False
                        if is_last_batch_epoch:
                            try:
                                accelerator.gradient_state.end_of_dataloader = True
                            except AttributeError:
                                grad_state = accelerator.gradient_state
                                active_dl = getattr(grad_state, "active_dataloader", None)
                                if active_dl is not None and hasattr(active_dl, "end_of_dataloader"):
                                    active_dl.end_of_dataloader = True
                        with accelerator.accumulate(pipeline.unet):
                            step_value_for_diag = global_step + 1
                            loss_raw, metrics = compute_batch_loss(
                                batch,
                                collect_denoise_diag=(
                                    denoise_diag_every > 0
                                    and step_value_for_diag % denoise_diag_every == 0
                                ),
                                collect_image_diag=(
                                    image_diag_every > 0
                                    and step_value_for_diag % image_diag_every == 0
                                ),
                            )
                            non_finite = (~torch.isfinite(loss_raw)).to(torch.int32)
                            if torch.distributed.is_available() and torch.distributed.is_initialized():
                                torch.distributed.all_reduce(
                                    non_finite, op=torch.distributed.ReduceOp.MAX
                                )
                            if non_finite.item() == 1:
                                bad_value = loss_raw.detach().float().item()
                                logger.warning(
                                    "Non-finite loss (%.4f) detected at epoch %d video %s batch %d. Skipping update.",
                                    bad_value,
                                    epoch,
                                    os.path.basename(video_path),
                                    batch_i,
                                )
                                loss = loss_raw * 0.0
                                accelerator.backward(loss)
                                optimizer.zero_grad(set_to_none=True)
                                accum_counter = 0
                                skip_update = True
                            else:
                                accum_counter += 1
                                loss = loss_raw
                                if debug_deepspeed_graph and not logged_loss_meta:
                                    logger.warning(
                                        "DeepSpeed debug: grad_enabled=%s inference_mode=%s loss.requires_grad=%s loss.grad_fn=%s",
                                        torch.is_grad_enabled(),
                                        torch.is_inference_mode_enabled(),
                                        loss.requires_grad,
                                        type(loss.grad_fn).__name__ if loss.grad_fn is not None else None,
                                    )
                                    logged_loss_meta = True
                                accelerator.backward(loss)
                                step_value = global_step + 1
                                pre_diag_snapshot = (
                                    _collect_mamba_diag_snapshot()
                                    if (diag_interval > 0 and step_value % diag_interval == 0)
                                    else {}
                                )
                                if accelerator.sync_gradients:
                                    if max_grad_norm > 0:
                                        accelerator.clip_grad_norm_(
                                            pipeline.unet.parameters(), max_grad_norm
                                        )
                                    optimizer.step()
                                    _log_mamba_diag(
                                        step_value=step_value,
                                        epoch_value=epoch,
                                        video_name_value=os.path.basename(video_path),
                                        batch_value=batch_i,
                                        did_step=True,
                                        pre_snapshot=pre_diag_snapshot,
                                    )
                                    optimizer.zero_grad(set_to_none=True)
                                    accum_counter = 0
                                else:
                                    _log_mamba_diag(
                                        step_value=step_value,
                                        epoch_value=epoch,
                                        video_name_value=os.path.basename(video_path),
                                        batch_value=batch_i,
                                        did_step=False,
                                        pre_snapshot=pre_diag_snapshot,
                                    )
                        if is_last_batch_epoch:
                            try:
                                accelerator.gradient_state.end_of_dataloader = False
                            except AttributeError:
                                grad_state = accelerator.gradient_state
                                active_dl = getattr(grad_state, "active_dataloader", None)
                                if active_dl is not None and hasattr(active_dl, "end_of_dataloader"):
                                    active_dl.end_of_dataloader = False
                        if skip_update:
                            continue
                    else:
                        step_value_for_diag = global_step + 1
                        loss_raw, metrics = compute_batch_loss(
                            batch,
                            collect_denoise_diag=(
                                denoise_diag_every > 0
                                and step_value_for_diag % denoise_diag_every == 0
                            ),
                            collect_image_diag=(
                                image_diag_every > 0
                                and step_value_for_diag % image_diag_every == 0
                            ),
                        )
                        if not torch.isfinite(loss_raw):
                            bad_value = loss_raw.detach().float().item()
                            logger.warning(
                                "Non-finite loss (%.4f) detected at epoch %d video %s batch %d. Skipping update.",
                                bad_value,
                                epoch,
                                os.path.basename(video_path),
                                batch_i,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            if _use_scaler():
                                scaler.update()
                            accum_counter = 0
                            continue
                        accum_counter += 1
                        accum_scale = grad_accum_steps
                        if is_last_batch_epoch and accum_counter < grad_accum_steps:
                            accum_scale = accum_counter
                        loss = loss_raw / float(accum_scale)

                        # 6) backward + optimizer step
                        if _use_scaler():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        step_value = global_step + 1
                        pre_diag_snapshot = (
                            _collect_mamba_diag_snapshot()
                            if (diag_interval > 0 and step_value % diag_interval == 0)
                            else {}
                        )

                        should_step = (accum_counter >= grad_accum_steps) or is_last_batch_epoch
                        if should_step:
                            if _use_scaler():
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                                optimizer.step()
                            _log_mamba_diag(
                                step_value=step_value,
                                epoch_value=epoch,
                                video_name_value=os.path.basename(video_path),
                                batch_value=batch_i,
                                did_step=True,
                                pre_snapshot=pre_diag_snapshot,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            accum_counter = 0
                        else:
                            _log_mamba_diag(
                                step_value=step_value,
                                epoch_value=epoch,
                                video_name_value=os.path.basename(video_path),
                                batch_value=batch_i,
                                did_step=False,
                                pre_snapshot=pre_diag_snapshot,
                            )

                    # ログ・記録 (勾配蓄積の有無に関わらずバッチ単位で記録)
                    global_step += 1
                    epoch_batches += 1
                    batch_loss_val = loss_raw.detach().item()
                    epoch_loss += batch_loss_val

                    printer.step(global_step=global_step, batch_idx=batch_i, loss_value=batch_loss_val)

                    if writer_tb:
                        writer_tb.add_scalar("mamba_gate/current", current_mamba_gate, global_step)
                        noise_val = metrics.get("loss_noise_mse", loss_raw)
                        writer_tb.add_scalar("loss/noise_mse", noise_val.detach().item(), global_step)
                        for image_key in (
                            "image_diag_mse",
                            "image_diag_l1",
                            "image_diag_psnr",
                            "image_diag_mask_mse",
                            "image_diag_mask_l1",
                            "image_diag_mask_psnr",
                        ):
                            value = metrics.get(image_key)
                            if value is not None:
                                writer_tb.add_scalar(f"image_diag/{image_key}", value.detach().item(), global_step)

                    _log_denoise_diag(
                        step_value=global_step,
                        epoch_value=epoch,
                        video_name_value=os.path.basename(video_path),
                        batch_value=batch_i,
                        metrics=metrics,
                    )
                    _log_image_diag(
                        step_value=global_step,
                        epoch_value=epoch,
                        video_name_value=os.path.basename(video_path),
                        batch_value=batch_i,
                        metrics=metrics,
                    )

                    if csv_path:
                        with open(csv_path, "a", encoding="utf-8", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [global_step, epoch, stage_name, os.path.basename(video_path)]
                                + [_format_metric(metrics, key) for key in train_metric_keys]
                            )
                    if device.type == "cuda":
                        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / float(1024**2)
                        logger.debug(
                            "VRAM peak allocated during video %d batch %d: %.1f MiB",
                            video_idx,
                            batch_i,
                            peak_allocated_mib,
                        )

            # エポック平均を表示して毎エポック後にチェックポイントを保存
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            printer.finish_epoch(avg_loss=avg_epoch_loss, epoch_batches=epoch_batches)
            if val_split and (epoch % val_interval_epochs == 0):
                run_validation(val_split, epoch)
            if ds_enabled and accelerator is not None:
                accelerator.wait_for_everyone()

            _save_full_checkpoint("latest", epoch)
            if epoch % save_interval_epochs == 0:
                _save_full_checkpoint(f"epoch{epoch:06d}", epoch, update_latest=False)
            if lr_scheduler is not None:
                lr_scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info("Scheduler step completed. Current learning rate: %.6e", current_lr)
            # 目標avg_loss に到達したら早期終了
            stop_early = (
                (target_avg_loss is not None) and (avg_epoch_loss <= float(target_avg_loss))
            )
            if ds_enabled and accelerator is not None and torch.distributed.is_initialized():
                flag = torch.tensor([1 if stop_early else 0], device=device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                stop_early = flag.item() == 1
            if stop_early:
                logger.info(
                    "Target avg_loss %.6f reached at epoch %d. Stopping early.",
                    target_avg_loss,
                    epoch,
                )
                break

            if stop_event.is_set():
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        # 割り込み時も最後に到達した重みを保存して終了
        try:
            _save_full_checkpoint("interrupted", current_epoch, update_latest=False)
        except Exception as e:
            logger.error("Interrupted. Failed to save checkpoint: %s", e)
        if writer_tb:
            writer_tb.close()
        _cleanup_stage_resources(cleanup_tag)
        return shard_applied

    final_path = os.path.join(save_dir, "unet_final.pt")
    if ds_enabled and accelerator is not None:
        if is_main_process:
            try:
                torch.save(accelerator.get_state_dict(pipeline.unet), final_path)
            except Exception as err:
                logger.warning("Failed to save final UNet weights: %s", err)
        accelerator.wait_for_everyone()
    else:
        torch.save(pipeline.unet.state_dict(), final_path)
    _save_full_checkpoint("final", current_epoch)
    if is_main_process:
        logger.info("Training complete. Final UNet weights stored at %s", final_path)

    if writer_tb:
        writer_tb.close()
    _cleanup_stage_resources(cleanup_tag)
    return shard_applied


def main(config: str | None = None, config_dir: str = "train_config", **overrides: Any) -> None:
    """Entry point for Fire CLI with optional JSON config loading."""
    ensure_logging_configured()
    config_identifier = config or os.environ.get("STEREOCRAFT_TRAIN_CONFIG")
    config_values: dict[str, Any] = {}
    if config_identifier:
        config_path, payload = load_json_config(config_identifier, config_dir)
        config_values.update(payload)
        logger.info("Loaded training config from %s", config_path)
    config_values.update(overrides)
    ds_cfg = config_values.get("deepspeed") or {}
    ds_enabled = bool(ds_cfg.get("enabled", False))
    local_rank = str(os.environ.get("LOCAL_RANK", "")).strip()
    is_main_process = local_rank in {"", "0"}

    banned_keys = {
        "min_h",
        "min_w",
        "max_h",
        "max_w",
        "crop_multiple",
        "shard_unet_across_gpus",
    }
    banned_present = sorted(banned_keys.intersection(config_values.keys()))
    if banned_present:
        raise ValueError(
            "SVD staged training is always-on; remove these keys: " + ", ".join(banned_present)
        )

    stage_epochs_raw = config_values.get("stage_epochs")
    stage_lrs_raw = config_values.get("stage_learning_rates")
    if stage_epochs_raw is None or stage_lrs_raw is None:
        raise ValueError("stage_epochs and stage_learning_rates are required and must have length 3.")
    if not isinstance(stage_epochs_raw, (list, tuple)) or not isinstance(stage_lrs_raw, (list, tuple)):
        raise ValueError("stage_epochs and stage_learning_rates must be list/tuple values of length 3.")
    if len(stage_epochs_raw) != 3 or len(stage_lrs_raw) != 3:
        raise ValueError("stage_epochs and stage_learning_rates must contain exactly 3 values.")
    stage_epochs = [int(value) for value in stage_epochs_raw]
    stage_lrs = [float(value) for value in stage_lrs_raw]
    if any(value < 1 for value in stage_epochs):
        raise ValueError("stage_epochs entries must be >= 1.")
    if any(value <= 0 for value in stage_lrs):
        raise ValueError("stage_learning_rates entries must be > 0.")

    fixed_stage_keys = {"stage_name", "stage_h", "stage_w", "stage_lr", "stage_idx"}
    fixed_stage_present = sorted(fixed_stage_keys.intersection(config_values.keys()))
    if fixed_stage_present:
        raise ValueError(
            "SVD staged training is always-on; remove these keys: " + ", ".join(fixed_stage_present)
        )

    max_stage_idx = int(config_values.get("max_stage_idx", 3))
    if max_stage_idx < 1 or max_stage_idx > 3:
        raise ValueError("max_stage_idx must be between 1 and 3.")

    base_config = dict(config_values)
    base_config.pop("stage_epochs", None)
    base_config.pop("stage_learning_rates", None)
    base_config.pop("max_stage_idx", None)
    stage_overrides = _normalize_stage_overrides(base_config.pop("stage_overrides", None))
    preflight_all_stages_before_train = bool(base_config.pop("preflight_all_stages_before_train", False))
    preflight_exit_after_all_stages = bool(base_config.pop("preflight_exit_after_all_stages", False))
    resume_into_source_dir = bool(base_config.pop("resume_into_source_dir", True))
    if ds_enabled:
        if str(base_config.get("unet_shard_mode", "off")).strip().lower() != "off":
            logger.info("DeepSpeed enabled; forcing unet_shard_mode=off.")
        base_config["unet_shard_mode"] = "off"
        base_config.pop("unet_device_map", None)

    signature = inspect.signature(_train_main)
    allowed_params = set(signature.parameters.keys())
    # ignore deepspeed/torchrun local rank injection
    base_config.pop("local_rank", None)
    unexpected_keys = set(base_config) - allowed_params
    if unexpected_keys:
        unexpected_list = ", ".join(sorted(unexpected_keys))
        raise ValueError(f"Unknown training parameters: {unexpected_list}")
    protected_stage_override_keys = {
        "stage_name",
        "stage_h",
        "stage_w",
        "stage_epochs",
        "stage_lr",
        "stage_idx",
        "resume_from",
        "save_dir",
        "preflight_only",
        "deepspeed_plugin_key",
        "deepspeed_plugin_configs",
        "local_rank",
    }
    for override_stage_idx, override_values in stage_overrides.items():
        override_unexpected = set(override_values) - allowed_params
        if override_unexpected:
            unexpected_list = ", ".join(sorted(override_unexpected))
            raise ValueError(f"Unknown training parameters in stage_overrides[{override_stage_idx}]: {unexpected_list}")
        override_protected = protected_stage_override_keys.intersection(override_values)
        if override_protected:
            protected_list = ", ".join(sorted(override_protected))
            raise ValueError(f"Do not set internal parameters in stage_overrides[{override_stage_idx}]: {protected_list}")

    stage_fields = {"stage_name", "stage_h", "stage_w", "stage_epochs", "stage_lr", "stage_idx"}
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect._empty and name not in base_config and name not in stage_fields
    ]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required training parameters: {missing_list}")

    save_dir = str(base_config.get("save_dir", "")).strip()
    resume_candidate = base_config.get("resume_from")
    resume_path_str = str(resume_candidate).strip() if resume_candidate is not None else ""
    resume_path_exists = bool(resume_path_str) and os.path.exists(resume_path_str)
    if resume_path_exists and resume_into_source_dir:
        resume_source_dir = (
            resume_path_str if os.path.isdir(resume_path_str) else os.path.dirname(resume_path_str)
        )
        if resume_source_dir:
            save_dir = resume_source_dir
            config_values["save_dir"] = save_dir
            base_config["save_dir"] = save_dir
            logger.info("Resume detected; using existing save_dir: %s", save_dir)
    else:
        resolved_save_dir = resolve_run_save_dir(save_dir)
        if resolved_save_dir:
            config_values["save_dir"] = resolved_save_dir
            base_config["save_dir"] = resolved_save_dir
            if resolved_save_dir != save_dir:
                logger.info("Resolved save_dir to per-run folder: %s", resolved_save_dir)
            save_dir = resolved_save_dir
    if resume_candidate is None and save_dir:
        ckpt_latest_path = os.path.join(save_dir, "train_state_latest.pt")
        if os.path.exists(ckpt_latest_path):
            resume_candidate = ckpt_latest_path
    if is_main_process:
        write_run_config_snapshot(save_dir, config_values, resume_candidate, logger=logger)

    resume_stage_idx = None
    resume_epoch = None
    resume_sharded = False
    resume_meta_path = None
    if resume_candidate and os.path.exists(resume_candidate):
        if os.path.isdir(resume_candidate):
            meta_guess = os.path.join(save_dir, "train_state_latest.pt")
            if os.path.exists(meta_guess):
                resume_meta_path = meta_guess
        else:
            resume_meta_path = resume_candidate
    if resume_meta_path:
        try:
            ckpt = torch.load(resume_meta_path, map_location="cpu")
            resume_stage_idx = ckpt.get("stage_idx", None)
            resume_epoch = ckpt.get("epoch", None)
            resume_sharded = bool(ckpt.get("unet_sharded", False))
            if resume_stage_idx is not None:
                resume_stage_idx = int(resume_stage_idx)
            if resume_epoch is not None:
                resume_epoch = int(resume_epoch)
        except Exception as err:
            logger.warning("Failed to read stage info from %s: %s", resume_meta_path, err)

    start_stage_idx = 1
    if resume_stage_idx is not None and 1 <= resume_stage_idx <= 3:
        stage_epoch_limit = stage_epochs[resume_stage_idx - 1]
        if resume_epoch is not None and resume_epoch >= stage_epoch_limit:
            start_stage_idx = resume_stage_idx + 1
        else:
            start_stage_idx = resume_stage_idx
        logger.info(
            "Resume checkpoint stage=%d epoch=%s -> starting at stage %d/3.",
            resume_stage_idx,
            resume_epoch,
            start_stage_idx,
        )

    if start_stage_idx > 3:
        logger.info("All stages already completed; nothing to do.")
        return

    base_shard_mode = str(base_config.get("unet_shard_mode", "off")).strip().lower()
    force_sharding = base_shard_mode == "on" or resume_sharded

    stages = [
        ("256x384", 256, 384),
        ("320x576", 320, 576),
        ("576x1024", 576, 1024),
    ]
    deepspeed_plugin_configs: dict[str, dict[str, Any]] = {}
    if ds_enabled:
        for ds_stage_idx in range(1, max_stage_idx + 1):
            ds_stage_kwargs = _merge_stage_config(base_config, stage_overrides.get(ds_stage_idx, {}))
            ds_stage_cfg = ds_stage_kwargs.get("deepspeed") or {}
            if bool(ds_stage_cfg.get("enabled", False)):
                deepspeed_plugin_configs[f"stage{ds_stage_idx}"] = dict(ds_stage_cfg)
    if preflight_all_stages_before_train and resume_candidate is None:
        logger.info(
            "Running preflight-only pass for stages 1..%d before starting a new training run.",
            max_stage_idx,
        )
        preflight_force_sharding = force_sharding
        for stage_idx, (stage_name, stage_h, stage_w) in enumerate(stages, start=1):
            if stage_idx > max_stage_idx:
                break
            stage_epochs_value = stage_epochs[stage_idx - 1]
            stage_lr_value = stage_lrs[stage_idx - 1]
            logger.info(
                "Preflight Stage %d/3: %s %dx%d lr=%.2e epochs=%d",
                stage_idx,
                stage_name,
                stage_h,
                stage_w,
                stage_lr_value,
                stage_epochs_value,
            )
            stage_kwargs = _merge_stage_config(base_config, stage_overrides.get(stage_idx, {}))
            stage_shard_mode = base_shard_mode
            if base_shard_mode == "auto" and preflight_force_sharding:
                stage_shard_mode = "on"
            stage_kwargs.update(
                {
                    "stage_name": stage_name,
                    "stage_h": stage_h,
                    "stage_w": stage_w,
                    "stage_epochs": stage_epochs_value,
                    "stage_lr": stage_lr_value,
                    "stage_idx": stage_idx,
                    "unet_shard_mode": stage_shard_mode,
                    "resume_from": None,
                    "preflight_only": True,
                    "deepspeed_plugin_key": f"stage{stage_idx}" if deepspeed_plugin_configs else None,
                    "deepspeed_plugin_configs": deepspeed_plugin_configs or None,
                }
            )
            used_sharding = _train_main(**stage_kwargs)
            if base_shard_mode == "auto" and used_sharding:
                preflight_force_sharding = True
        logger.info("Preflight-only pass completed; starting training run.")
        if preflight_exit_after_all_stages:
            logger.info("preflight_exit_after_all_stages=true; exiting before training.")
            return
    elif preflight_all_stages_before_train:
        logger.info(
            "Skipping preflight_all_stages_before_train because resume_candidate is set: %s",
            resume_candidate,
        )

    for stage_idx, (stage_name, stage_h, stage_w) in enumerate(stages, start=1):
        if stage_idx > max_stage_idx:
            logger.info("Stopping after stage %d/3 due to max_stage_idx=%d.", stage_idx - 1, max_stage_idx)
            break
        if stage_idx < start_stage_idx:
            logger.info("Skipping Stage %d/3 (%s): already completed.", stage_idx, stage_name)
            continue
        stage_epochs_value = stage_epochs[stage_idx - 1]
        stage_lr_value = stage_lrs[stage_idx - 1]
        logger.info(
            "Stage %d/3: %s %dx%d lr=%.2e epochs=%d",
            stage_idx,
            stage_name,
            stage_h,
            stage_w,
            stage_lr_value,
            stage_epochs_value,
        )
        stage_kwargs = _merge_stage_config(base_config, stage_overrides.get(stage_idx, {}))
        stage_shard_mode = base_shard_mode
        if base_shard_mode == "auto" and force_sharding:
            stage_shard_mode = "on"
        stage_kwargs.update(
            {
                "stage_name": stage_name,
                "stage_h": stage_h,
                "stage_w": stage_w,
                "stage_epochs": stage_epochs_value,
                "stage_lr": stage_lr_value,
                "stage_idx": stage_idx,
                "unet_shard_mode": stage_shard_mode,
                "deepspeed_plugin_key": f"stage{stage_idx}" if deepspeed_plugin_configs else None,
                "deepspeed_plugin_configs": deepspeed_plugin_configs or None,
            }
        )
        if stage_idx == start_stage_idx:
            stage_kwargs["resume_from"] = resume_candidate
        else:
            stage_kwargs["resume_from"] = None
        used_sharding = _train_main(**stage_kwargs)
        if base_shard_mode == "auto" and used_sharding:
            force_sharding = True


if __name__ == "__main__":
    Fire(main)
