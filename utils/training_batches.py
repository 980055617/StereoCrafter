"""Batch preparation helpers for training.

動画を「時間方向のチャンク」に分割し、GPU メモリに優しい形のバッチに詰め替えます。

形状の取り決め:
- すべてのフレームテンソルは [F, C, H, W]
- mask は 1ch (C=1)。cond/target は 3ch (C=3)。値域は [0,1]
"""

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import torch

from utils.inpainting import read_and_prepare_video


@dataclass
class TrainBatch:
    """Per-chunk tensors used by training.

    Attributes:
        cond: 条件入力 (左+ワープなどから作られた条件) [f, 3, H, W]
        mask: インペインティング領域のマスク [f, 1, H, W]
        target: 右画像(教師) [f, 3, H, W]
    """

    cond: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor


def chunk_frame_ranges(num_frames: int, chunk_size: int, overlap: int) -> Iterator[Tuple[int, int]]:
    """Yield [start, end) frame ranges using the inference-like stride.

    - 入力が chunk_size より短い場合は全フレームを 1 チャンクで返す
    - 最終チャンクは末尾に合わせて調整 (推論と同一ポリシー)
    """
    if num_frames <= chunk_size or chunk_size <= 0:
        yield 0, num_frames
        return

    step = max(chunk_size - overlap, 1)
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        if end - start < chunk_size:
            start = max(num_frames - chunk_size, 0)
            end = start + chunk_size
        yield start, end
        if end >= num_frames:
            break
        start += step


class _BatchIterable(Iterable[TrainBatch]):
    """Lazy iterable that moves chunked frames to GPU on-the-fly.

    Avoids holding all GPU chunks for a video simultaneously.
    """

    def __init__(
        self,
        frames_warped_cpu: torch.Tensor,
        frames_mask_cpu: torch.Tensor,
        frames_right_cpu: torch.Tensor,
        frames_chunk: int,
        overlap: int,
        device: torch.device,
        dtype: torch.dtype,
        crop_size: Optional[Tuple[int, int]] = None,
        crop_multiple: int = 128,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._frames_warped_cpu = frames_warped_cpu
        self._frames_mask_cpu = frames_mask_cpu
        self._frames_right_cpu = frames_right_cpu
        self._source_hw = (int(frames_warped_cpu.shape[2]), int(frames_warped_cpu.shape[3]))
        self._ranges: List[Tuple[int, int]] = list(
            chunk_frame_ranges(frames_warped_cpu.shape[0], frames_chunk, overlap)
        )
        self._device = device
        self._dtype = dtype
        self._crop_size = crop_size
        self._crop_region = crop_region
        self._crop_multiple = max(1, crop_multiple)

    def __len__(self) -> int:  # for progress bars
        return len(self._ranges)

    def __iter__(self) -> Iterator[TrainBatch]:
        for start, end in self._ranges:
            cond_cpu = self._frames_warped_cpu[start:end]
            mask_cpu = self._frames_mask_cpu[start:end]
            target_cpu = self._frames_right_cpu[start:end]

            if self._crop_region is not None and self._crop_size is not None:
                cond_cpu, mask_cpu, target_cpu = self._apply_fixed_crop(cond_cpu, mask_cpu, target_cpu)
            elif self._crop_size is not None:
                cond_cpu, mask_cpu, target_cpu = self._apply_random_crop(cond_cpu, mask_cpu, target_cpu)

            cond = cond_cpu.to(device=self._device, dtype=self._dtype, non_blocking=True)
            mask = mask_cpu.to(device=self._device, dtype=self._dtype, non_blocking=True)
            target = target_cpu.to(device=self._device, dtype=self._dtype, non_blocking=True)
            yield TrainBatch(cond=cond, mask=mask, target=target)

    def _apply_random_crop(
        self,
        cond: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crop_h, crop_w = self._crop_size  # type: ignore[misc]
        height = cond.shape[2]
        width = cond.shape[3]

        crop_h = self._align_dim(crop_h, height)
        crop_w = self._align_dim(crop_w, width)

        if crop_h >= height and crop_w >= width:
            return cond, mask, target

        max_top = height - crop_h
        max_left = width - crop_w
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0

        slice_h = slice(top, top + crop_h)
        slice_w = slice(left, left + crop_w)
        cond = cond[:, :, slice_h, slice_w]
        mask = mask[:, :, slice_h, slice_w]
        target = target[:, :, slice_h, slice_w]
        return cond, mask, target

    @property
    def crop_region_info(self) -> Optional[dict]:
        if self._crop_region is None:
            return None
        top, left, crop_h, crop_w = self._crop_region
        return {
            "top": int(top),
            "left": int(left),
            "height": int(crop_h),
            "width": int(crop_w),
            "source_height": self._source_hw[0],
            "source_width": self._source_hw[1],
        }

    def _align_dim(self, desired: int, max_dim: int) -> int:
        """Clamp desired crop to fit and respect the configured multiple."""
        desired = max(1, min(desired, max_dim))
        multiple = self._crop_multiple
        if multiple <= 1:
            return desired
        aligned = (desired // multiple) * multiple
        if aligned == 0:
            aligned = multiple if max_dim >= multiple else max_dim
        if aligned > max_dim:
            aligned = max_dim
        return aligned

    def _apply_fixed_crop(
        self,
        cond: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top, left, crop_h, crop_w = self._crop_region  # type: ignore[misc]
        height = cond.shape[2]
        width = cond.shape[3]

        crop_h = self._align_dim(crop_h, height)
        crop_w = self._align_dim(crop_w, width)

        max_top = max(height - crop_h, 0)
        max_left = max(width - crop_w, 0)
        top = max(0, min(top, max_top))
        left = max(0, min(left, max_left))

        slice_h = slice(top, top + crop_h)
        slice_w = slice(left, left + crop_w)
        cond = cond[:, :, slice_h, slice_w]
        mask = mask[:, :, slice_h, slice_w]
        target = target[:, :, slice_h, slice_w]
        return cond, mask, target


def prepare_batches(
    video_path: str,
    frames_chunk: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
    random_crop_size: Optional[Tuple[int, int]] = None,
    crop_multiple: int = 128,
    crop_min_size: Optional[Tuple[int, int]] = None,
    crop_max_size: Optional[Tuple[int, int]] = None,
) -> Iterable[TrainBatch]:
    """Load a stereo tiled video and yield `TrainBatch` lazily per chunk.

    Args:
        video_path: 入力動画のパス (2x2 タイル前提)。
        frames_chunk: 1 チャンクに含めるフレーム数。
        overlap: チャンク間のオーバーラップ数。
        device: テンソルを配置するデバイス。
        dtype: テンソル化時の dtype。
        random_crop_size: (height, width)。指定時は cond/mask/target を同じ位置で各チャンクごとにランダムクロップ。
        crop_multiple: クロップ縦横を揃える倍数。VAE のスケールに合わせて 128 などを推奨。
        crop_min_size/crop_max_size: (min_h/min_w), (max_h/max_w)。動画ごとに固定サイズ・開始位置をランダム決定する場合に使用。

    Returns:
        Iterable[TrainBatch]: イテラブル（len() は利用可能）。各反復で GPU にコピーされたチャンクを返す。
    """
    _, _, frames_warped_cpu, frames_mask_cpu, frames_right_cpu = read_and_prepare_video(video_path, return_right=True)

    crop_size = None
    crop_region: Optional[Tuple[int, int, int, int]] = None
    if random_crop_size is not None:
        crop_h, crop_w = random_crop_size
        if crop_h > 0 and crop_w > 0:
            crop_size = (crop_h, crop_w)
    elif crop_min_size is not None or crop_max_size is not None:
        min_h = crop_min_size[0] if crop_min_size is not None else 1
        min_w = crop_min_size[1] if crop_min_size is not None else 1
        max_h = crop_max_size[0] if crop_max_size is not None else frames_warped_cpu.shape[2]
        max_w = crop_max_size[1] if crop_max_size is not None else frames_warped_cpu.shape[3]

        height = frames_warped_cpu.shape[2]
        width = frames_warped_cpu.shape[3]

        min_h = max(1, min(min_h, height))
        max_h = max(min_h, min(max_h, height))
        min_w = max(1, min(min_w, width))
        max_w = max(min_w, min(max_w, width))

        crop_h = random.randint(min_h, max_h)
        crop_w = random.randint(min_w, max_w)
        crop_size = (crop_h, crop_w)

        max_top = max(height - crop_h, 0)
        max_left = max(width - crop_w, 0)
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        crop_region = (top, left, crop_h, crop_w)

    return _BatchIterable(
        frames_warped_cpu=frames_warped_cpu,
        frames_mask_cpu=frames_mask_cpu,
        frames_right_cpu=frames_right_cpu,
        frames_chunk=frames_chunk,
        overlap=overlap,
        device=device,
        dtype=dtype,
        crop_size=crop_size,
        crop_multiple=crop_multiple,
        crop_region=crop_region,
    )
