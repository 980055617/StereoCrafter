from __future__ import annotations
import numpy as np

def split_quadrants_frames(frames_thwc: np.ndarray):
    """
    Split a 2x2 tiled video frame sequence [T,H,W,C] into (warped, left, mask) in THWC order.
    Assumes layout:
      top-left  = left (GT)
      bottom-left = mask
      bottom-right = warped
    Returns: (warped_np, left_np, mask_np)
    """
    T, H, W, C = frames_thwc.shape
    h, w = H // 2, W // 2
    left = frames_thwc[:, :h, :w, :]
    mask = frames_thwc[:, h:, :w, :]
    warped = frames_thwc[:, h:, w:, :]
    return warped, left, mask

