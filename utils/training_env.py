# =============================================
# File: /workspace/stereocraft/utils/training_env.py
# ---------------------------------------------
# 目的: 学習環境の初期化とシード
# =============================================

"""Training environment helpers (seed, interrupts, device selection)."""

import logging
import random
import signal
import threading
from typing import Optional, Sequence

import torch

from utils.logging_utils import ensure_logging_configured


logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Seed Python/torch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_interrupt_handler(signals: Optional[Sequence[int]] = None) -> threading.Event:
    """Return an Event that flips on SIGINT/SIGTERM."""
    ensure_logging_configured()
    stop_event = threading.Event()

    watched = list(signals) if signals is not None else [signal.SIGINT, signal.SIGTERM]

    def _handle_sig(signum, _frame):
        if not stop_event.is_set():
            logger.warning("Signal %s received. Will stop after the current step...", signum)
        stop_event.set()

    for sig in watched:
        try:
            signal.signal(sig, _handle_sig)
        except Exception:
            continue

    return stop_event


def get_compute_device() -> torch.device:
    """Select CUDA if available; otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    ensure_logging_configured()
    logger.warning("CUDA device not detected. Training will run on CPU.")
    return torch.device("cpu")
