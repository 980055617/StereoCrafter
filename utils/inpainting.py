# =============================================
# File: /workspace/stereocraft/utils/inpainting.py
# ---------------------------------------------
# 目的: インペイント用タイル処理と動画I/O
# =============================================

import inspect
from contextlib import nullcontext
from typing import Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """Blend two tiles horizontally over an overlap region in latent space."""
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(b.device)
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[:, :, :, -overlap_size:] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    """Blend two tiles vertically over an overlap region in latent space."""
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(b.device)
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[:, :, -overlap_size:, :] + weight_b * b[:, :, :overlap_size, :]
    return b


def spatial_tiled_process(
    cond_frames: torch.Tensor,
    mask_frames: torch.Tensor,
    process_func,
    tile_num: int,
    spatial_n_compress: int = 8,
    enable_autograd: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Run a diffusion pipeline on spatial tiles and stitch the latent tiles.

    Returns a latent tensor of shape [F, C, H/8, W/8].
    """
    context = nullcontext() if enable_autograd else torch.no_grad()
    supports_grad_flag = "grad_enabled" in inspect.signature(process_func.__call__).parameters

    with context:
        height = cond_frames.shape[2]
        width = cond_frames.shape[3]

        tile_overlap = (128, 128)
        tile_size = (
            int((height + tile_overlap[0] * (tile_num - 1)) / tile_num),
            int((width + tile_overlap[1] * (tile_num - 1)) / tile_num),
        )
        tile_stride = (tile_size[0] - tile_overlap[0], tile_size[1] - tile_overlap[1])

        cols = []
        for i in range(0, tile_num):
            rows = []
            for j in range(0, tile_num):
                cond_tile = cond_frames[
                    :,
                    :,
                    i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                    j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
                ]
                mask_tile = mask_frames[
                    :,
                    :,
                    i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                    j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
                ]

                call_kwargs = dict(
                    frames=cond_tile,
                    frames_mask=mask_tile,
                    height=cond_tile.shape[2],
                    width=cond_tile.shape[3],
                    num_frames=len(cond_tile),
                    output_type="latent",
                    **kwargs,
                )
                if supports_grad_flag:
                    call_kwargs["grad_enabled"] = enable_autograd

                tile = process_func(**call_kwargs).frames[0]

                rows.append(tile)
            cols.append(rows)

        latent_stride = (tile_stride[0] // spatial_n_compress, tile_stride[1] // spatial_n_compress)
        latent_overlap = (tile_overlap[0] // spatial_n_compress, tile_overlap[1] // spatial_n_compress)

        results_cols = []
        for i, rows in enumerate(cols):
            results_rows = []
            for j, tile in enumerate(rows):
                if i > 0:
                    tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
                if j > 0:
                    tile = blend_h(rows[j - 1], tile, latent_overlap[1])
                results_rows.append(tile)
            results_cols.append(results_rows)

        pixels = []
        for i, rows in enumerate(results_cols):
            for j, tile in enumerate(rows):
                if i < len(results_cols) - 1:
                    tile = tile[:, :, : latent_stride[0], :]
                if j < len(rows) - 1:
                    tile = tile[:, :, :, : latent_stride[1]]
                rows[j] = tile
            pixels.append(torch.cat(rows, dim=3))
        x = torch.cat(pixels, dim=2)
    return x


def write_video_opencv(input_frames: np.ndarray, fps: float, output_video_path: str) -> None:
    """Write frames [T,H,W,C] in RGB float[0,255] or uint8 to an mp4 file."""
    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(num_frames):
        out.write(input_frames[i, :, :, ::-1])
    out.release()


@torch.no_grad()
def read_and_prepare_video(
    input_video_path: str,
    return_right: bool = False,
) -> Tuple:
    """Read a 2x2 tiled input video and split to left/warped/mask tensors.

    Returns:
        - `return_right=False`: fps, frames_left, frames_warped, frames_mask (all float tensors in [0,1])
        - `return_right=True`: fps, frames_left, frames_warped, frames_mask, frames_right (right-eye tensor in [0,1])
        - frames_* shapes: [T, C, H, W] with mask as 1-channel (C=1)
    """
    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = float(video_reader.get_avg_fps())
    frame_indices = list(range(len(video_reader)))
    frames = video_reader.get_batch(frame_indices)
    frames = torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float()  # [T,C,H,W]

    height, width = frames.shape[2] // 2, frames.shape[3] // 2
    frames_left = frames[:, :, :height, :width]
    frames_right = frames[:, :, :height, width:]
    frames_mask = frames[:, :, height:, :width]
    frames_warped = frames[:, :, height:, width:]

    # Crop to multiples of 128
    height = height // 128 * 128
    width = width // 128 * 128
    frames_left = frames_left[:, :, :height, :width]
    frames_right = frames_right[:, :, :height, :width]
    frames_mask = frames_mask[:, :, :height, :width]
    frames_warped = frames_warped[:, :, :height, :width]

    # Normalize to [0,1], mask to gray 1ch
    frames_left = frames_left / 255.0
    frames_right = frames_right / 255.0
    frames_warped = frames_warped / 255.0
    frames_mask = (frames_mask / 255.0).mean(dim=1, keepdim=True)

    if return_right:
        return fps, frames_left, frames_warped, frames_mask, frames_right

    return fps, frames_left, frames_warped, frames_mask
