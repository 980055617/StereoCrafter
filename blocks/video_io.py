# =============================================
# File: /workspace/stereocraft/blocks/video_io.py
# ---------------------------------------------
# 目的: 2x2タイル動画の分割
# =============================================

from __future__ import annotations

import numpy as np


def split_quadrants_frames(frames_thwc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split 2x2 tiled [T,H,W,C] into (warped, left, mask) in THWC."""
    _, H, W, _ = frames_thwc.shape
    h, w = H // 2, W // 2
    left = frames_thwc[:, :h, :w, :]
    mask = frames_thwc[:, h:, :w, :]
    warped = frames_thwc[:, h:, w:, :]
    return warped, left, mask
