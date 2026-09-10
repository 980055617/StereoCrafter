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

_LPIPS_MODEL = None


def _lpips_model() -> Any:
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips

        _LPIPS_MODEL = lpips.LPIPS(net="alex")
        if torch.cuda.is_available():
            _LPIPS_MODEL = _LPIPS_MODEL.cuda()
    return _LPIPS_MODEL


def _lpips_distance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> float:
    """Mean LPIPS distance over frames. pred/target are (N,3,H,W) in [0,1]."""
    model = _lpips_model()
    device = next(model.parameters()).device
    pred01 = pred.clamp(0.0, 1.0) * 2.0 - 1.0
    target01 = target.clamp(0.0, 1.0) * 2.0 - 1.0
    if mask is not None:
        # Zero out non-mask regions in both images so LPIPS focuses on the
        # inpainted area; still a full-image forward pass since LPIPS is a
        # deep, spatially-pooled network, not a pixelwise metric.
        mask_rgb = mask.expand_as(pred01)
        pred01 = pred01 * mask_rgb
        target01 = target01 * mask_rgb
    total = 0.0
    count = int(pred01.shape[0])
    batch = 8
    with torch.no_grad():
        for start in range(0, count, batch):
            p = pred01[start : start + batch].to(device)
            t = target01[start : start + batch].to(device)
            d = model(p, t)
            total += float(d.sum().item())
    return total / max(count, 1)


def _lpips_distance_bbox_crop(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pad: int = 8,
    min_size: int = 32,
) -> float:
    """Per-frame bounding-box-crop LPIPS: crop pred/target to the mask's
    bounding box (plus padding) instead of zeroing pixels, so LPIPS sees real
    local image content rather than an image with a black hole punched in it.
    Frames with no mask pixels are skipped. Slower than the zeroed full-frame
    approach (one forward pass per frame, since crop sizes vary), but a much
    more standard way to compute a "local region" perceptual distance."""
    model = _lpips_model()
    device = next(model.parameters()).device
    pred01 = pred.clamp(0.0, 1.0) * 2.0 - 1.0
    target01 = target.clamp(0.0, 1.0) * 2.0 - 1.0
    n, _, h, w = pred01.shape
    total = 0.0
    used = 0
    with torch.no_grad():
        for i in range(n):
            m = mask[i, 0]
            rows = torch.where(m.any(dim=1))[0]
            cols = torch.where(m.any(dim=0))[0]
            if rows.numel() == 0 or cols.numel() == 0:
                continue
            top = max(int(rows.min().item()) - pad, 0)
            bottom = min(int(rows.max().item()) + 1 + pad, h)
            left = max(int(cols.min().item()) - pad, 0)
            right = min(int(cols.max().item()) + 1 + pad, w)
            if (bottom - top) < min_size:
                extra = min_size - (bottom - top)
                top = max(top - extra // 2, 0)
                bottom = min(top + min_size, h)
            if (right - left) < min_size:
                extra = min_size - (right - left)
                left = max(left - extra // 2, 0)
                right = min(left + min_size, w)
            p = pred01[i : i + 1, :, top:bottom, left:right].to(device)
            t = target01[i : i + 1, :, top:bottom, left:right].to(device)
            d = model(p, t)
            total += float(d.sum().item())
            used += 1
    return total / max(used, 1)


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
    compute_lpips: bool = False,
) -> dict[str, Any]:
    pred, target = _common_crop(pred, target)
    diff = pred - target
    mask_cropped = _common_crop(mask, pred)[0] if mask is not None else None
    if mask_cropped is not None:
        mask_rgb = mask_cropped.expand_as(diff)
        denom = float(mask_rgb.sum().item())
        if denom <= 0.0:
            return {
                "label": label,
                "region": region,
                "frames": int(pred.shape[0]),
                "mse": "",
                "mae": "",
                "psnr": "",
                "lpips": "",
            }
        mse = float((diff.pow(2) * mask_rgb).sum().item() / denom)
        mae = float((diff.abs() * mask_rgb).sum().item() / denom)
    else:
        mse = float(diff.pow(2).mean().item())
        mae = float(diff.abs().mean().item())
    lpips_value = None
    if compute_lpips:
        if mask_cropped is not None:
            lpips_value = _lpips_distance_bbox_crop(pred, target, mask_cropped)
        else:
            lpips_value = _lpips_distance(pred, target, None)
    return {
        "label": label,
        "region": region,
        "frames": int(pred.shape[0]),
        "mse": f"{mse:.10g}",
        "mae": f"{mae:.10g}",
        "psnr": f"{_psnr(mse):.10g}",
        "lpips": f"{lpips_value:.10g}" if lpips_value is not None else "",
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
    compute_lpips: bool = False,
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
        _metric_row(label=label, region="all_generated", pred=gen_right, target=gt_right, compute_lpips=compute_lpips),
        _metric_row(label=label, region="all_warped", pred=warped, target=gt_right, compute_lpips=compute_lpips),
        _metric_row(label=label, region="mask_generated", pred=gen_right, target=gt_right, mask=mask_bin, compute_lpips=compute_lpips),
        _metric_row(label=label, region="mask_warped", pred=warped, target=gt_right, mask=mask_bin, compute_lpips=compute_lpips),
        _metric_row(label=label, region="inv_mask_generated", pred=gen_right, target=gt_right, mask=inv_mask, compute_lpips=compute_lpips),
        _metric_row(label=label, region="inv_mask_warped", pred=warped, target=gt_right, mask=inv_mask, compute_lpips=compute_lpips),
    ]

    if output_csv is None:
        output_csv = os.path.join(str(Path(generated_sbs).parent), "metrics_vs_train_tile.csv")
    os.makedirs(str(Path(output_csv).parent), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["label", "region", "frames", "mse", "mae", "psnr", "lpips"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['label']} {row['region']}: "
            f"PSNR={row['psnr']} MSE={row['mse']} MAE={row['mae']} LPIPS={row['lpips']}"
        )
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    Fire(main)
