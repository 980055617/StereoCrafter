# =============================================
# File: /workspace/stereocraft/utils/training_batches.py
# ---------------------------------------------
# 目的: 学習用バッチ生成（動画チャンク）
# =============================================

"""Batch prep for 2x2 tiled videos (frames: [F,C,H,W], mask: 1ch)."""

import logging
import os
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
from decord import VideoReader, cpu

logger = logging.getLogger(__name__)


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


class _StreamingVideo:
    """Wrapper around VideoReader that loads tiled frames on demand."""

    def __init__(self, video_path: str) -> None:
        self._reader = VideoReader(video_path, ctx=cpu(0))
        self._video_path = video_path
        if len(self._reader) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        first = self._reader[0].asnumpy()
        raw_h, raw_w = first.shape[0], first.shape[1]
        tile_h = (raw_h // 2) // 128 * 128
        tile_w = (raw_w // 2) // 128 * 128
        if tile_h == 0:
            tile_h = max(raw_h // 2, 1)
        if tile_w == 0:
            tile_w = max(raw_w // 2, 1)
        self._tile_h = tile_h
        self._tile_w = tile_w

    @property
    def frame_count(self) -> int:
        return len(self._reader)

    @property
    def spatial_hw(self) -> Tuple[int, int]:
        return self._tile_h, self._tile_w

    def load_chunk(self, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if start < 0 or end <= start:
            raise ValueError(f"Invalid chunk range: {start}:{end}")
        end = min(end, self.frame_count)
        indices = list(range(start, end))
        batch = self._reader.get_batch(indices).asnumpy()  # [F, H_total, W_total, 3]
        frames = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        height = self._tile_h
        width = self._tile_w
        frames = frames[:, :, : height * 2, : width * 2]
        frames_right = frames[:, :, :height, width:]
        frames_mask = frames[:, :, height:, :width]
        frames_warped = frames[:, :, height:, width:]
        frames_mask = frames_mask.mean(dim=1, keepdim=True)
        return frames_warped, frames_mask, frames_right


class _BatchIterable(Iterable[TrainBatch]):
    """Lazy iterable that moves chunked frames to GPU on-the-fly.

    Avoids holding all GPU chunks for a video simultaneously.
    """

    def __init__(
        self,
        video_stream: _StreamingVideo,
        frames_chunk: int,
        overlap: int,
        device: torch.device,
        dtype: torch.dtype,
        crop_multiple: int = 128,
        crop_region: Optional[Tuple[int, int, int, int]] = None,
        use_prev_target_overlap: bool = False,
        overlap_teacher_prob: float = 1.0,
        overlap_noise_std: float = 0.0,
    ) -> None:
        self._video_stream = video_stream
        self._source_hw = video_stream.spatial_hw
        self._ranges: List[Tuple[int, int]] = list(
            chunk_frame_ranges(video_stream.frame_count, frames_chunk, overlap)
        )
        self._overlap = max(0, overlap)
        self._device = device
        self._dtype = dtype
        self._crop_region = crop_region
        self._crop_multiple = max(1, crop_multiple)
        self._use_prev_target_overlap = use_prev_target_overlap
        self._overlap_teacher_prob = max(0.0, min(1.0, overlap_teacher_prob))
        self._overlap_noise_std = max(0.0, float(overlap_noise_std))

    def __len__(self) -> int:  # for progress bars
        return len(self._ranges)

    def __iter__(self) -> Iterator[TrainBatch]:
        prev_target_cpu: Optional[torch.Tensor] = None
        prev_end: Optional[int] = None
        debug_overlap = os.getenv("TRAIN_DEBUG_OVERLAP") == "1"
        debug_logged = False
        for start, end in self._ranges:
            cond_cpu, mask_cpu, target_cpu = self._video_stream.load_chunk(start, end)

            if self._crop_region is not None:
                cond_cpu, mask_cpu, target_cpu = self._apply_fixed_crop(cond_cpu, mask_cpu, target_cpu)

            # 直前チャンクの出力（教師フレーム）でオーバーラップ領域を置き換え、推論時の条件付けに近づける
            actual_overlap = max(0, (prev_end or 0) - start) if prev_end is not None else 0
            if debug_overlap and not debug_logged and actual_overlap != self._overlap and start > 0:
                logger.debug(
                    "Actual overlap differs from configured overlap (actual=%d, configured=%d) at chunk start=%d",
                    actual_overlap,
                    self._overlap,
                    start,
                )
                debug_logged = True
            if (
                self._use_prev_target_overlap
                and prev_target_cpu is not None
                and actual_overlap > 0
                and start > 0
            ):
                ov = min(actual_overlap, cond_cpu.shape[0], prev_target_cpu.shape[0])
                if ov > 0:
                    # scheduled sampling: 一定確率で教師(前チャンクGT)を使い、残りは元のcondを保持
                    if random.random() < self._overlap_teacher_prob:
                        overlap_val = prev_target_cpu[-ov:].clone()
                        if self._overlap_noise_std > 0:
                            noise = torch.randn_like(overlap_val) * self._overlap_noise_std
                            overlap_val = torch.clamp(overlap_val + noise, 0.0, 1.0)
                        cond_cpu[:ov] = overlap_val

            cond = cond_cpu.to(device=self._device, dtype=self._dtype, non_blocking=True)
            mask = mask_cpu.to(device=self._device, dtype=self._dtype, non_blocking=True)
            target = target_cpu.to(device=self._device, dtype=self._dtype, non_blocking=True)
            prev_target_cpu = target_cpu.detach().clone() if self._use_prev_target_overlap else None
            prev_end = end
            yield TrainBatch(cond=cond, mask=mask, target=target)

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
    crop_multiple: int = 128,
    crop_min_size: Optional[Tuple[int, int]] = None,
    crop_max_size: Optional[Tuple[int, int]] = None,
    use_prev_target_overlap: bool = True,
    overlap_teacher_prob: float = 1.0,
    overlap_noise_std: float = 0.0,
) -> Iterable[TrainBatch]:
    """Load a stereo tiled video and yield `TrainBatch` lazily per chunk.

    Args:
        video_path: 入力動画のパス (2x2 タイル前提)。
        frames_chunk: 1 チャンクに含めるフレーム数。
        overlap: チャンク間のオーバーラップ数。
        device: テンソルを配置するデバイス。
        dtype: テンソル化時の dtype。
        crop_multiple: クロップ縦横を揃える倍数。VAE のスケールに合わせて 128 などを推奨。
        crop_min_size/crop_max_size: 固定クロップサイズ (H, W)。両方指定し、同一サイズにすること。
        use_prev_target_overlap: True のとき、オーバーラップ領域の条件フレームを前チャンクのターゲットで置換し、推論時の条件付けを模倣。

    Returns:
        Iterable[TrainBatch]: イテラブル（len() は利用可能）。各反復で GPU にコピーされたチャンクを返す。
    """
    video_stream = _StreamingVideo(video_path)

    def _align_dim(desired: int, max_dim: int) -> int:
        desired = max(1, min(desired, max_dim))
        if crop_multiple <= 1:
            return desired
        aligned = (desired // crop_multiple) * crop_multiple
        if aligned == 0:
            aligned = crop_multiple if max_dim >= crop_multiple else max_dim
        if aligned > max_dim:
            aligned = max_dim
        return aligned

    crop_region: Optional[Tuple[int, int, int, int]] = None
    if crop_min_size is not None or crop_max_size is not None:
        if crop_min_size is None or crop_max_size is None:
            raise ValueError("crop_min_size and crop_max_size must both be set for fixed crops.")
        if tuple(crop_min_size) != tuple(crop_max_size):
            raise ValueError(
                "Variable crop sizes are no longer supported; set crop_min_size == crop_max_size."
            )
        height, width = video_stream.spatial_hw
        crop_h = _align_dim(int(crop_min_size[0]), height)
        crop_w = _align_dim(int(crop_min_size[1]), width)
        max_top = max(height - crop_h, 0)
        max_left = max(width - crop_w, 0)
        top = max_top // 2
        left = max_left // 2
        crop_region = (top, left, crop_h, crop_w)

    return _BatchIterable(
        video_stream=video_stream,
        frames_chunk=frames_chunk,
        overlap=overlap,
        device=device,
        dtype=dtype,
        crop_multiple=crop_multiple,
        crop_region=crop_region,
        use_prev_target_overlap=use_prev_target_overlap,
        overlap_teacher_prob=overlap_teacher_prob,
        overlap_noise_std=overlap_noise_std,
    )
