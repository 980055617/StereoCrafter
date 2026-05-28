"""Evaluate inpainting SBS output against a 2x2 StereoCrafter train tile.

The train tile layout is:
top-left: left eye, top-right: GT right eye,
bottom-left: mask, bottom-right: warped right eye.
Generated videos are expected to be SBS: left | generated right.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any

import torch
from decord import VideoReader, cpu
from fire import Fire


def _load_video(path: str) -> torch.Tensor:
    reader = VideoReader(path, ctx=cpu(0))
    frames = reader.get_batch(list(range(len(reader)))).asnumpy()
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


def _center_crop(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    src_h = int(frames.shape[2])
    src_w = int(frames.shape[3])
    if height > src_h or width > src_w:
        raise ValueError(f"Crop {height}x{width} exceeds source {src_h}x{src_w}")
    top = (src_h - height) // 2
    left = (src_w - width) // 2
    return frames[:, :, top : top + height, left : left + width]


def _psnr(mse: float) -> float:
    return 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-8)))


def _metric_row(
    *,
    label: str,
    region: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    pred, target = _common_crop(pred, target)
    diff = pred - target
    if mask is not None:
        mask = _common_crop(mask, pred)[0]
        mask_rgb = mask.expand_as(diff)
        denom = float(mask_rgb.sum().item())
        if denom <= 0.0:
            return {
                "label": label,
                "region": region,
                "frames": int(pred.shape[0]),
                "mse": "",
                "mae": "",
                "psnr": "",
            }
        mse = float((diff.pow(2) * mask_rgb).sum().item() / denom)
        mae = float((diff.abs() * mask_rgb).sum().item() / denom)
    else:
        mse = float(diff.pow(2).mean().item())
        mae = float(diff.abs().mean().item())
    return {
        "label": label,
        "region": region,
        "frames": int(pred.shape[0]),
        "mse": f"{mse:.10g}",
        "mae": f"{mae:.10g}",
        "psnr": f"{_psnr(mse):.10g}",
    }


def _common_crop(*frames: torch.Tensor) -> list[torch.Tensor]:
    frame_count = min(int(x.shape[0]) for x in frames)
    height = min(int(x.shape[2]) for x in frames)
    width = min(int(x.shape[3]) for x in frames)
    return [x[:frame_count, :, :height, :width] for x in frames]


def main(
    generated_sbs: str,
    train_tile: str = "video_data/train/0160_train.mp4",
    output_csv: str | None = None,
    target_height: int | None = None,
    target_width: int | None = None,
    mask_threshold: float = 0.5,
) -> None:
    generated = _load_video(generated_sbs)
    train = _load_video(train_tile)

    train_h = int(train.shape[2]) // 2
    train_w = int(train.shape[3]) // 2
    gt_right = train[:, :, :train_h, train_w : train_w * 2]
    mask = train[:, :, train_h : train_h * 2, :train_w].mean(dim=1, keepdim=True)
    warped = train[:, :, train_h : train_h * 2, train_w : train_w * 2]

    gen_right = generated[:, :, :, generated.shape[3] // 2 :]
    if target_height is not None and target_width is not None:
        gt_right = _center_crop(gt_right, int(target_height), int(target_width))
        mask = _center_crop(mask, int(target_height), int(target_width))
        warped = _center_crop(warped, int(target_height), int(target_width))
    mask_bin = (mask > float(mask_threshold)).float()
    inv_mask = 1.0 - mask_bin

    label = Path(generated_sbs).parent.name or Path(generated_sbs).stem
    rows = [
        _metric_row(label=label, region="all_generated", pred=gen_right, target=gt_right),
        _metric_row(label=label, region="all_warped", pred=warped, target=gt_right),
        _metric_row(label=label, region="mask_generated", pred=gen_right, target=gt_right, mask=mask_bin),
        _metric_row(label=label, region="mask_warped", pred=warped, target=gt_right, mask=mask_bin),
        _metric_row(label=label, region="inv_mask_generated", pred=gen_right, target=gt_right, mask=inv_mask),
        _metric_row(label=label, region="inv_mask_warped", pred=warped, target=gt_right, mask=inv_mask),
    ]

    if output_csv is None:
        output_csv = os.path.join(str(Path(generated_sbs).parent), "metrics_vs_train_tile.csv")
    os.makedirs(str(Path(output_csv).parent), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["label", "region", "frames", "mse", "mae", "psnr"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['label']} {row['region']}: "
            f"PSNR={row['psnr']} MSE={row['mse']} MAE={row['mae']}"
        )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    Fire(main)
