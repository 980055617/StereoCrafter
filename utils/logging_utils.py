"""Lightweight logging and memory/time utilities for experiments.

含まれるもの:
- ensure_dir: 出力ディレクトリの作成
- get_gpu_memory_mb/tensor_mem_mb: CUDA/テンソルのメモリ量を MB 単位で取得
- format_seconds: 秒を H:MM:SS 表記に整形
- TrainCSVLogger/EventCSVLogger: CSV ベースの簡易ロガー
- StepTimer/MemoryTracer: ステップ時間とメモリの測定ユーティリティ

学習ループに最小限のロギングを足したいときに便利なユーティリティ群です。
"""

import csv
import logging
import os
import time
from typing import Dict, Iterable, Optional, Tuple, Union

import torch


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist (like `mkdir -p`)."""
    os.makedirs(path, exist_ok=True)


def _bytes_to_mb(x: int) -> float:
    return round(x / (1024**2), 2)


logger = logging.getLogger(__name__)


def ensure_logging_configured(level: int = logging.DEBUG) -> None:
    """Configure root logger once with a simple console formatter."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _normalize_cuda_device(device: Optional[Union[torch.device, int, str]]) -> Optional[int]:
    """Return a CUDA device index for various device representations."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        return device.index if device.index is not None else torch.cuda.current_device()
    if isinstance(device, str):
        device = device.strip().lower()
        if device == "cuda":
            return torch.cuda.current_device()
        if device.startswith("cuda:"):
            _, _, idx = device.partition(":")
            if idx.isdigit():
                return int(idx)
            return torch.cuda.current_device()
        if device.startswith("gpu"):
            _, _, idx = device.partition(":")
            if idx.isdigit():
                return int(idx)
    return torch.cuda.current_device()


def _try_get_vram_usage_mib(device: Optional[torch.device]) -> Optional[Tuple[float, float]]:
    """Return (used_mib, total_mib) for the given CUDA device."""
    if not torch.cuda.is_available():
        return None
    try:
        dev_index = _normalize_cuda_device(device)
        if dev_index is None:
            return None
        torch.cuda.synchronize(dev_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev_index)
    except Exception as exc:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "torch.cuda.mem_get_info unavailable for device %s: %s; falling back to memory_allocated()",
                device,
                exc,
            )
        # Fallback: use allocated memory and device properties
        try:
            dev_index = _normalize_cuda_device(device)
            if dev_index is None:
                return None
            torch.cuda.synchronize(dev_index)
            allocated = torch.cuda.memory_allocated(dev_index)
            total_bytes = torch.cuda.get_device_properties(dev_index).total_memory
        except Exception:
            return None
        used_bytes = allocated
    else:
        used_bytes = total_bytes - free_bytes
    mib = 1024**2
    return used_bytes / mib, total_bytes / mib


def log_vram_usage(message: str, device: Optional[torch.device], *, level: int = logging.INFO) -> None:
    """Log VRAM usage via torch.cuda.mem_get_info when available."""
    ensure_logging_configured()
    if device is None:
        logger.log(level, "%s (device unavailable)", message)
        return
    if not logger.isEnabledFor(level):
        return
    usage = _try_get_vram_usage_mib(device)
    if usage is None:
        logger.log(level, "%s (VRAM usage unavailable for device %s)", message, device)
        return
    used_mib, total_mib = usage
    mem_breakdown = get_gpu_memory_mb(device)
    allocated_mb = mem_breakdown.get("allocated_mb", 0.0)
    reserved_mb = mem_breakdown.get("reserved_mb", 0.0)
    logger.log(
        level,
        "%s (VRAM used: %.1f MiB / %.1f MiB | allocated: %.1f MB | reserved: %.1f MB)",
        message,
        used_mib,
        total_mib,
        allocated_mb,
        reserved_mb,
    )


def get_gpu_memory_mb(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Return allocated and reserved GPU memory (in MB). 0 if CUDA unavailable."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    dev_index = device.index if isinstance(device, torch.device) and device.type == "cuda" else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev_index)
    reserved = torch.cuda.memory_reserved(dev_index)
    return {"allocated_mb": _bytes_to_mb(allocated), "reserved_mb": _bytes_to_mb(reserved)}


def tensor_mem_mb(x: Optional[torch.Tensor]) -> float:
    if x is None:
        return 0.0
    if not isinstance(x, torch.Tensor):
        return 0.0
    try:
        elem_size = x.element_size()
        return round(x.numel() * elem_size / (1024**2), 2)
    except Exception:
        return 0.0


def format_seconds(seconds: float) -> str:
    """Format seconds to H:MM:SS."""
    seconds = int(max(seconds, 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class TrainCSVLogger:
    """CSV logger that overwrites on initialization and appends thereafter.

    反復ごとの損失など、定型のスカラ値を簡単に CSV へ追記する用途を想定しています。
    """

    def __init__(self, csv_path: str, fieldnames: Iterable[str]):
        self.csv_path = csv_path
        self.fieldnames = list(fieldnames)
        ensure_dir(os.path.dirname(csv_path) or ".")
        # Overwrite at the start of a run to avoid accumulating from previous runs
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict):
        # Only keep known fields; missing ones default to ""
        filtered = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(filtered)


class EventCSVLogger:
    """Event logger that overwrites on initialization and appends thereafter.

    任意のイベントを時刻・フェーズ名・詳細の 3 カラムで残します。
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        ensure_dir(os.path.dirname(csv_path) or ".")
        self.fieldnames = [
            "time_s",
            "phase",
            "detail",
        ]
        # Overwrite at the start of a run to avoid accumulating from previous runs
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, phase: str, detail: str = "", device: Optional[torch.device] = None, t0: Optional[float] = None):
        row = {
            "time_s": round(time.time() - (t0 or 0.0), 3),
            "phase": phase,
            "detail": detail,
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class StepTimer:
    """Utility to track elapsed and step durations."""

    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def step(self) -> Dict[str, float]:
        now = time.time()
        step_time = now - self.last_time
        elapsed = now - self.start_time
        self.last_time = now
        return {"step_time": step_time, "elapsed": elapsed}


class MemoryTracer:
    """Context manager to measure CUDA memory usage for a code block.

    Records allocated/reserved at entry and exit, plus peak allocated during the block.
    """

    def __init__(self, device: Optional[torch.device] = None, label: str = ""):
        self.device = device if isinstance(device, torch.device) else torch.device("cuda") if torch.cuda.is_available() else None
        self.label = label
        self.start_alloc = 0
        self.start_reserved = 0
        self.end_alloc = 0
        self.end_reserved = 0
        self.peak_alloc = 0

    def __enter__(self):
        if self.device is None or not torch.cuda.is_available():
            return self
        dev_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(dev_index)
        torch.cuda.reset_peak_memory_stats(dev_index)
        self.start_alloc = torch.cuda.memory_allocated(dev_index)
        self.start_reserved = torch.cuda.memory_reserved(dev_index)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device is None or not torch.cuda.is_available():
            return False
        dev_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        torch.cuda.synchronize(dev_index)
        self.end_alloc = torch.cuda.memory_allocated(dev_index)
        self.end_reserved = torch.cuda.memory_reserved(dev_index)
        self.peak_alloc = torch.cuda.max_memory_allocated(dev_index)
        return False

    def to_row(self, prefix: str = "") -> Dict[str, float]:
        if self.device is None or not torch.cuda.is_available():
            return {
                f"{prefix}alloc_start_mb": 0.0,
                f"{prefix}alloc_end_mb": 0.0,
                f"{prefix}reserved_start_mb": 0.0,
                f"{prefix}reserved_end_mb": 0.0,
                f"{prefix}peak_alloc_mb": 0.0,
            }
        return {
            f"{prefix}alloc_start_mb": _bytes_to_mb(self.start_alloc),
            f"{prefix}alloc_end_mb": _bytes_to_mb(self.end_alloc),
            f"{prefix}reserved_start_mb": _bytes_to_mb(self.start_reserved),
            f"{prefix}reserved_end_mb": _bytes_to_mb(self.end_reserved),
            f"{prefix}peak_alloc_mb": _bytes_to_mb(self.peak_alloc),
        }


class TrainingProgressPrinter:
    """Simple terminal progress printer for training loops.

    Prints compact lines with epoch/video/batch indices, loss, step time, elapsed, and GPU memory.
    Control verbosity via log_interval.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        log_interval: int = 10,
        enable_mem: bool = True,
        *,
        log_level: int = logging.INFO,
        logger_obj: Optional[logging.Logger] = None,
    ):
        ensure_logging_configured()
        self.logger = logger_obj or logging.getLogger(__name__)
        self.log_level = log_level
        self.device = device
        self.log_interval = max(1, int(log_interval))
        self.enable_mem = enable_mem
        self.timer = StepTimer()
        self._epochs_total = 0
        self._epoch_idx = 0
        self._videos_total = 0
        self._video_idx = 0
        self._batches_total = 0

    def start_epoch(self, epoch_idx: int, epochs_total: int, videos_total: int) -> None:
        self._epoch_idx = epoch_idx
        self._epochs_total = epochs_total
        self._videos_total = videos_total
        self.timer = StepTimer()

    def start_video(self, video_idx: int, batches_total: int) -> None:
        self._video_idx = video_idx
        self._batches_total = max(1, int(batches_total))

    def step(self, *, global_step: int, batch_idx: int, loss_value: float, extra: Optional[Dict[str, float]] = None) -> None:
        if global_step % self.log_interval != 0:
            return
        if not self.logger.isEnabledFor(self.log_level):
            return
        t = self.timer.step()
        mem = get_gpu_memory_mb(self.device) if self.enable_mem else {"allocated_mb": 0.0, "reserved_mb": 0.0}
        parts = [
            f"[ep {self._epoch_idx}/{self._epochs_total}]",
            f"[vid {self._video_idx}/{self._videos_total}]",
            f"[b {batch_idx}/{self._batches_total}]",
            f"step={global_step}",
            f"loss={loss_value:.4f}",
            f"time={t['step_time']:.3f}s",
            f"elapsed={format_seconds(t['elapsed'])}",
        ]
        if self.enable_mem and torch.cuda.is_available():
            parts.append(f"mem={mem['allocated_mb']:.0f}/{mem['reserved_mb']:.0f}MB")
        if extra:
            for k, v in extra.items():
                try:
                    parts.append(f"{k}={float(v):.4f}")
                except Exception:
                    parts.append(f"{k}={v}")
        self.logger.log(self.log_level, " ".join(parts))

    def finish_epoch(self, avg_loss: float, epoch_batches: int) -> None:
        mem = get_gpu_memory_mb(self.device) if self.enable_mem else {"allocated_mb": 0.0, "reserved_mb": 0.0}
        if not self.logger.isEnabledFor(self.log_level):
            return
        message = (
            f"Epoch {self._epoch_idx}/{self._epochs_total} done | avg_loss={avg_loss:.4f} | "
            f"batches={epoch_batches} | mem={mem.get('allocated_mb',0):.0f}/{mem.get('reserved_mb',0):.0f}MB"
        )
        self.logger.log(self.log_level, message)
