"""Training environment helpers.

主に「学習スクリプトの冒頭に毎回書く」環境まわりの初期化を関数化しています。

- set_global_seed: 乱数シードの一括設定 (Python/Torch/CUDA)
- setup_interrupt_handler: Ctrl+C/SIGTERM を受けたら安全に停止するためのフラグを返す
- get_compute_device: CUDA があれば CUDA、なければ CPU を選択

Usage:
    set_global_seed(42)
    stop_event = setup_interrupt_handler()  # 学習ループで stop_event.is_set() を監視
    device = get_compute_device()
"""

import random
import signal
import threading
from typing import Optional, Sequence

import torch


def set_global_seed(seed: int) -> None:
    """Seed Python and torch RNGs for reproducibility.

    Args:
        seed: 任意の整数。再現性向上のため固定します。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_interrupt_handler(signals: Optional[Sequence[int]] = None) -> threading.Event:
    """Return an Event that flips when SIGINT/SIGTERM (or provided signals) fire.

    学習ループ中に `stop_event.is_set()` をチェックし、割り込みを安全なポイントで反映します。

    Args:
        signals: 監視するシグナルのリスト。未指定時は [SIGINT, SIGTERM]。

    Returns:
        threading.Event: 割り込み受信時に set() されるイベント。
    """
    stop_event = threading.Event()

    watched = list(signals) if signals is not None else [signal.SIGINT, signal.SIGTERM]

    def _handle_sig(signum, _frame):
        if not stop_event.is_set():
            print(f"Signal {signum} received. Will stop after the current step...")
        stop_event.set()

    for sig in watched:
        try:
            signal.signal(sig, _handle_sig)
        except Exception:
            continue

    return stop_event


def get_compute_device() -> torch.device:
    """Select CUDA if available; fall back to CPU with a warning.

    Returns:
        torch.device: `cuda` もしくは `cpu`。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("Warning: CUDA device not detected. Training will run on CPU.")
    return torch.device("cpu")
