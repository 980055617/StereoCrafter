"""Batch preparation helpers for training.

動画を「時間方向のチャンク」に分割し、GPU メモリに優しい形のバッチに詰め替えます。

形状の取り決め:
- すべてのフレームテンソルは [F, C, H, W]
- mask は 1ch (C=1)。cond/target は 3ch (C=3)。値域は [0,1]
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Iterable

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
    ) -> None:
        self._frames_warped_cpu = frames_warped_cpu
        self._frames_mask_cpu = frames_mask_cpu
        self._frames_right_cpu = frames_right_cpu
        self._ranges: List[Tuple[int, int]] = list(
            chunk_frame_ranges(frames_warped_cpu.shape[0], frames_chunk, overlap)
        )
        self._device = device
        self._dtype = dtype

    def __len__(self) -> int:  # for progress bars
        return len(self._ranges)

    def __iter__(self) -> Iterator[TrainBatch]:
        for start, end in self._ranges:
            cond = self._frames_warped_cpu[start:end].to(device=self._device, dtype=self._dtype, non_blocking=True)
            mask = self._frames_mask_cpu[start:end].to(device=self._device, dtype=self._dtype, non_blocking=True)
            target = self._frames_right_cpu[start:end].to(device=self._device, dtype=self._dtype, non_blocking=True)
            yield TrainBatch(cond=cond, mask=mask, target=target)


def prepare_batches(
    video_path: str,
    frames_chunk: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterable[TrainBatch]:
    """Load a stereo tiled video and yield `TrainBatch` lazily per chunk.

    Args:
        video_path: 入力動画のパス (2x2 タイル前提)。
        frames_chunk: 1 チャンクに含めるフレーム数。
        overlap: チャンク間のオーバーラップ数。
        device: テンソルを配置するデバイス。
        dtype: テンソル化時の dtype。

    Returns:
        Iterable[TrainBatch]: イテラブル（len() は利用可能）。各反復で GPU にコピーされたチャンクを返す。
    """
    _, _, frames_warped_cpu, frames_mask_cpu, frames_right_cpu = read_and_prepare_video(video_path, return_right=True)
    return _BatchIterable(
        frames_warped_cpu=frames_warped_cpu,
        frames_mask_cpu=frames_mask_cpu,
        frames_right_cpu=frames_right_cpu,
        frames_chunk=frames_chunk,
        overlap=overlap,
        device=device,
        dtype=dtype,
    )
