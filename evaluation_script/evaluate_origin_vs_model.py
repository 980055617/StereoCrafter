"""Evaluate origin and custom StereoCrafter outputs against GT right-eye videos.

This script intentionally does not run inference. Generate videos first, then use
this evaluator to compare both output folders against the same GT folder.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from fire import Fire


HIGHER_IS_BETTER = {"psnr", "ssim", "masked_psnr"}
LOWER_IS_BETTER = {"lpips", "tof", "masked_mae"}


@dataclass
class EvalConfig:
    frames_chunk: int = 16
    generated_is_sbs: bool = True
    enable_lpips: bool = False
    enable_tof: bool = False
    mask_dir: str | None = None
    mask_threshold: float = 0.5


def chunk_frame_ranges(num_frames: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    if chunk_size <= 0 or num_frames <= chunk_size:
        yield 0, num_frames
        return
    for start in range(0, num_frames, chunk_size):
        yield start, min(start + chunk_size, num_frames)


def compute_psnr_from_sums(sum_sq: float, count: float) -> float:
    if count <= 0:
        return float("nan")
    mse = max(sum_sq / count, 1e-8)
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim_map(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Return SSIM map for [T,C,H,W] tensors in [0,1]."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu1 = F.avg_pool2d(pred, 3, 1, 1)
    mu2 = F.avg_pool2d(tgt, 3, 1, 1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(tgt * tgt, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(pred * tgt, 3, 1, 1) - mu1_mu2
    return ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )


def load_video_chunk(path: str, start: int, end: int) -> torch.Tensor:
    vr = VideoReader(path, ctx=cpu(0))
    batch = vr.get_batch(list(range(start, end))).asnumpy()
    return torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0


def load_generated_right(path: str, start: int, end: int, generated_is_sbs: bool) -> torch.Tensor:
    frames = load_video_chunk(path, start, end)
    if generated_is_sbs:
        return frames[:, :, :, frames.shape[3] // 2 :]
    return frames


def load_mask_chunk(mask_path: str, start: int, end: int, target_h: int, target_w: int) -> torch.Tensor:
    """Load mask as [T,1,H,W].

    If the mask video is a 2x2 StereoCrafter training tile, the bottom-left tile is
    used. Otherwise the full frame is converted to grayscale.
    """
    frames = load_video_chunk(mask_path, start, end)
    _, _, h, w = frames.shape
    if h >= target_h * 2 and w >= target_w * 2:
        frames = frames[:, :, target_h : target_h * 2, :target_w]
    gray = (0.2989 * frames[:, 0] + 0.5870 * frames[:, 1] + 0.1140 * frames[:, 2]).unsqueeze(1)
    return gray[:, :, :target_h, :target_w]


def crop_common(*frames: torch.Tensor) -> list[torch.Tensor]:
    h_min = min(int(x.shape[2]) for x in frames)
    w_min = min(int(x.shape[3]) for x in frames)
    return [x[:, :, :h_min, :w_min] for x in frames]


def find_mp4_by_stem(directory: str, stem: str) -> str | None:
    exact = Path(directory) / f"{stem}.mp4"
    if exact.is_file():
        return str(exact)
    candidates = sorted(Path(directory).glob(f"{stem}*.mp4"))
    return str(candidates[0]) if candidates else None


def metric_mean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def evaluate_one_folder(
    label: str,
    generated_dir: str,
    gt_dir: str,
    output_csv: str,
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    gen_files = sorted(Path(generated_dir).glob("*.mp4"))
    if not gen_files:
        raise FileNotFoundError(f"No .mp4 files found in {generated_dir}")

    lpips_model = None
    if cfg.enable_lpips:
        try:
            import lpips  # type: ignore
        except Exception as err:
            print(f"[WARN] LPIPS unavailable ({err}). Skipping LPIPS for {label}.")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lpips_model = lpips.LPIPS(net="vgg").eval().to(device)

    rows: list[dict[str, Any]] = []
    for index, gen_path in enumerate(gen_files, start=1):
        stem = gen_path.stem
        gt_path = find_mp4_by_stem(gt_dir, stem)
        if gt_path is None:
            print(f"[{label} {index}/{len(gen_files)}] missing GT for {stem}, skip")
            continue

        vr_gen = VideoReader(str(gen_path), ctx=cpu(0))
        vr_gt = VideoReader(gt_path, ctx=cpu(0))
        n_frames = min(len(vr_gen), len(vr_gt))
        if n_frames <= 0:
            print(f"[{label} {index}/{len(gen_files)}] zero frames for {stem}, skip")
            continue

        mask_path = find_mp4_by_stem(cfg.mask_dir, stem) if cfg.mask_dir else None
        sum_sq = count_sq = 0.0
        ssim_values: list[float] = []
        lpips_values: list[float] = []
        tof_values: list[float] = []
        masked_sum_sq = masked_count = 0.0
        masked_abs_sum = masked_abs_count = 0.0

        for start, end in chunk_frame_ranges(n_frames, cfg.frames_chunk):
            gen_right = load_generated_right(str(gen_path), start, end, cfg.generated_is_sbs)
            gt = load_video_chunk(gt_path, start, end)
            gen_right, gt = crop_common(gen_right, gt)

            diff = gen_right - gt
            sum_sq += float((diff * diff).sum().item())
            count_sq += float(diff.numel())

            ssim_values.append(float(compute_ssim_map(gen_right, gt).mean().clamp(-1.0, 1.0).item()))

            if lpips_model is not None:
                dev = next(lpips_model.parameters()).device
                lp = lpips_model(gen_right.to(dev) * 2.0 - 1.0, gt.to(dev) * 2.0 - 1.0)
                lpips_values.append(float(lp.mean().item()))

            if cfg.enable_tof and gen_right.shape[0] > 1:
                tof_values.append(compute_tof(gen_right, gt))

            if mask_path is not None:
                mask = load_mask_chunk(mask_path, start, end, gen_right.shape[2], gen_right.shape[3])
                gen_right, gt, mask = crop_common(gen_right, gt, mask)
                mask = (mask > cfg.mask_threshold).float()
                mask_rgb = mask.expand_as(gen_right)
                masked_count += float(mask_rgb.sum().item())
                masked_sum_sq += float(((gen_right - gt).pow(2) * mask_rgb).sum().item())
                masked_abs_count += float(mask_rgb.sum().item())
                masked_abs_sum += float(((gen_right - gt).abs() * mask_rgb).sum().item())

        row = {
            "video": stem,
            "label": label,
            "frames": n_frames,
            "psnr": compute_psnr_from_sums(sum_sq, count_sq),
            "ssim": metric_mean(ssim_values),
            "lpips": metric_mean(lpips_values) if lpips_model is not None else float("nan"),
            "tof": metric_mean(tof_values) if cfg.enable_tof else float("nan"),
            "masked_psnr": compute_psnr_from_sums(masked_sum_sq, masked_count) if mask_path else float("nan"),
            "masked_mae": masked_abs_sum / masked_abs_count if masked_abs_count > 0 else float("nan"),
        }
        rows.append(row)
        print(
            f"[{label} {index}/{len(gen_files)}] {stem}: "
            f"PSNR={row['psnr']:.3f}, SSIM={row['ssim']:.4f}"
        )

    write_csv(output_csv, rows, metric_fieldnames())
    return rows


def compute_tof(gen: torch.Tensor, gt: torch.Tensor) -> float:
    def to_gray(x: torch.Tensor) -> torch.Tensor:
        return (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).cpu().numpy()

    gen_gray = to_gray(gen)
    gt_gray = to_gray(gt)
    values: list[float] = []
    for idx in range(gen.shape[0] - 1):
        flow_gen = cv2.calcOpticalFlowFarneback(
            gen_gray[idx], gen_gray[idx + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flow_gt = cv2.calcOpticalFlowFarneback(
            gt_gray[idx], gt_gray[idx + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        values.append(float(abs(flow_gen - flow_gt).mean()))
    return metric_mean(values)


def metric_fieldnames() -> list[str]:
    return ["video", "label", "frames", "psnr", "ssim", "lpips", "tof", "masked_psnr", "masked_mae"]


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.8g}"
    return value


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    os.makedirs(str(Path(path).parent), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"label": label, "videos": len(rows), "frames": sum(int(r["frames"]) for r in rows)}
    for name in metric_fieldnames()[3:]:
        summary[name] = metric_mean([float(r[name]) for r in rows])
    return summary


def compare_rows(origin_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    origin_by_video = {str(row["video"]): row for row in origin_rows}
    model_by_video = {str(row["video"]): row for row in model_rows}
    rows: list[dict[str, Any]] = []
    for video in sorted(set(origin_by_video) & set(model_by_video)):
        origin = origin_by_video[video]
        model = model_by_video[video]
        row: dict[str, Any] = {"video": video, "frames": min(int(origin["frames"]), int(model["frames"]))}
        wins = 0
        losses = 0
        for metric in metric_fieldnames()[3:]:
            origin_value = float(origin[metric])
            model_value = float(model[metric])
            row[f"origin_{metric}"] = origin_value
            row[f"model_{metric}"] = model_value
            row[f"delta_{metric}"] = model_value - origin_value
            if math.isnan(origin_value) or math.isnan(model_value):
                row[f"winner_{metric}"] = ""
            elif metric in HIGHER_IS_BETTER:
                row[f"winner_{metric}"] = "model" if model_value > origin_value else "origin"
            elif metric in LOWER_IS_BETTER:
                row[f"winner_{metric}"] = "model" if model_value < origin_value else "origin"
            else:
                row[f"winner_{metric}"] = ""
            if row[f"winner_{metric}"] == "model":
                wins += 1
            elif row[f"winner_{metric}"] == "origin":
                losses += 1
        row["winner_overall"] = "model" if wins > losses else ("origin" if losses > wins else "tie")
        rows.append(row)
    return rows


def comparison_fieldnames() -> list[str]:
    fields = ["video", "frames"]
    for metric in metric_fieldnames()[3:]:
        fields.extend([f"origin_{metric}", f"model_{metric}", f"delta_{metric}", f"winner_{metric}"])
    fields.append("winner_overall")
    return fields


def main(
    origin_dir: str = "video_data/model_test_output/origin",
    model_dir: str = "video_data/model_test_output/my_model",
    gt_dir: str = "video_data/right_eye",
    output_dir: str = "evaluation_script/results",
    frames_chunk: int = 16,
    generated_is_sbs: bool = True,
    enable_lpips: bool = False,
    enable_tof: bool = False,
    mask_dir: str | None = None,
    mask_threshold: float = 0.5,
) -> None:
    cfg = EvalConfig(
        frames_chunk=frames_chunk,
        generated_is_sbs=generated_is_sbs,
        enable_lpips=enable_lpips,
        enable_tof=enable_tof,
        mask_dir=mask_dir,
        mask_threshold=mask_threshold,
    )
    os.makedirs(output_dir, exist_ok=True)

    origin_csv = str(Path(output_dir) / "origin_metrics.csv")
    model_csv = str(Path(output_dir) / "model_metrics.csv")
    comparison_csv = str(Path(output_dir) / "comparison.csv")
    summary_json = str(Path(output_dir) / "summary.json")

    origin_rows = evaluate_one_folder("origin", origin_dir, gt_dir, origin_csv, cfg)
    model_rows = evaluate_one_folder("model", model_dir, gt_dir, model_csv, cfg)
    comparison = compare_rows(origin_rows, model_rows)
    write_csv(comparison_csv, comparison, comparison_fieldnames())

    summary = {
        "origin": summarize(origin_rows, "origin"),
        "model": summarize(model_rows, "model"),
        "paired_videos": len(comparison),
        "settings": {
            "origin_dir": origin_dir,
            "model_dir": model_dir,
            "gt_dir": gt_dir,
            "frames_chunk": frames_chunk,
            "generated_is_sbs": generated_is_sbs,
            "enable_lpips": enable_lpips,
            "enable_tof": enable_tof,
            "mask_dir": mask_dir,
            "mask_threshold": mask_threshold,
        },
    }
    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(f"Wrote origin metrics: {origin_csv}")
    print(f"Wrote model metrics : {model_csv}")
    print(f"Wrote comparison    : {comparison_csv}")
    print(f"Wrote summary       : {summary_json}")


if __name__ == "__main__":
    Fire(main)
