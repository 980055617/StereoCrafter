#!/usr/bin/env python3
# =============================================
# File: /workspace/stereocraft/scripts/plot_loss_transition.py
# ---------------------------------------------
# 目的: 学習ログの損失推移プロット
# =============================================

"""Plot train/val/test loss curves from CSV logs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _select_backend_env() -> None:
    # If DISPLAY is set but unusable (no Xauthority), drop it to avoid X errors.
    display = os.environ.get("DISPLAY")
    wayland = os.environ.get("WAYLAND_DISPLAY")
    has_xauth = os.environ.get("XAUTHORITY") or os.access(
        Path.home() / ".Xauthority", os.R_OK
    )

    if display and not wayland and not has_xauth:
        os.environ.pop("DISPLAY", None)
        os.environ.setdefault("MPLBACKEND", "Agg")
    elif not display and not wayland:
        os.environ.setdefault("MPLBACKEND", "Agg")


_select_backend_env()

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


EpochLoss = Sequence[Tuple[int, float]]
LossRange = Dict[int, Tuple[float, float]]
StageBoundary = Tuple[float, str]
CurveSpec = Tuple[str, EpochLoss, LossRange | None, str, str]
TBinPoint = Tuple[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val/test loss transitions averaged by epoch, step, timestep, or video."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing train_log*.csv and val_log*.csv/test_log*.csv (takes precedence).",
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=None,
        help="CSV that tracks training loss per step/epoch.",
    )
    parser.add_argument(
        "--val-log",
        dest="val_log",
        type=Path,
        default=None,
        help="CSV for validation loss per step/epoch.",
    )
    parser.add_argument(
        "--test-log",
        dest="test_log",
        type=Path,
        default=None,
        help="CSV for test loss per step/epoch.",
    )
    parser.add_argument(
        "--test-label",
        type=str,
        default=None,
        help="Legend label for a single val/test curve.",
    )
    parser.add_argument(
        "--loss-column",
        type=str,
        default="loss_noise_mse",
        help="Column name that stores the loss values to average.",
    )
    parser.add_argument(
        "--group-by",
        choices=("epoch", "step", "t", "video"),
        default="epoch",
        help="X-axis grouping mode: average by epoch, step, timestep (t), or video.",
    )
    parser.add_argument(
        "--epoch-column",
        type=str,
        default="epoch",
        help="Column name that stores epoch numbers.",
    )
    parser.add_argument(
        "--step-column",
        type=str,
        default="step",
        help="Column name that stores step numbers (used when --group-by=step).",
    )
    parser.add_argument(
        "--timestep-column",
        type=str,
        default="timestep",
        help="Column name that stores diffusion timesteps (used when --group-by=t).",
    )
    parser.add_argument(
        "--step-bin-size",
        type=int,
        default=100,
        help="Bin size for step min/max envelope (used when --group-by=step).",
    )
    parser.add_argument(
        "--t-bin-size",
        type=int,
        default=50,
        help="Bin width for timestep bars (used when --group-by=t). Example: 50 -> 0-49,50-99,...",
    )
    parser.add_argument(
        "--t-group-edges",
        type=str,
        default=None,
        help=(
            "Comma-separated timestep bin edges for --group-by=epoch/step. "
            "Example: '0,200,400,600,800,1000' to draw one curve per t-bin."
        ),
    )
    parser.add_argument(
        "--t-group-layout",
        choices=("overlay", "separate"),
        default="overlay",
        help="Rendering layout for --t-group-edges mode: overlay all groups or separate subplots.",
    )
    parser.add_argument(
        "--stage-column",
        type=str,
        default="stage",
        help="Column name for stage labels. Used to stitch stage-wise logs continuously.",
    )
    parser.add_argument(
        "--disable-stage-stitch",
        action="store_true",
        help="Disable stage-wise x-axis stitching and boundary markers.",
    )
    parser.add_argument(
        "--log-rank",
        type=int,
        default=1,
        help="Rank id used when resolving logs from --log-dir (e.g., rank1 logs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the rendered figure (PNG).",
    )
    parser.add_argument(
        "--video-column",
        type=str,
        default="video",
        help="Column name for video id/path (used when --group-by=video).",
    )
    parser.add_argument(
        "--video-top-k",
        type=int,
        default=30,
        help="Show only top-K videos in --group-by=video mode (set <=0 to show all).",
    )
    parser.add_argument(
        "--video-sort-by",
        choices=("avg", "p95", "max"),
        default="avg",
        help="Sort key for --group-by=video mode.",
    )
    parser.add_argument(
        "--video-t-max-edges",
        type=str,
        default="0,200,400,600,800,1000",
        help=(
            "Comma-separated timestep bin edges used to plot per-video max points in --group-by=video mode. "
            "Default creates 5 bins: 0-200,...,800-1000."
        ),
    )
    parser.add_argument(
        "--video-show-global-max",
        action="store_true",
        help="Show global max points in --group-by=video mode (off by default).",
    )
    return parser.parse_args()


def ensure_csv(path: Path, label: str) -> None:
    if not path:
        raise ValueError(f"{label} log path is empty.")
    if not path.exists():
        raise FileNotFoundError(f"{label} log not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} log is not a file: {path}")


def find_unique_log(log_dir: Path, prefix: str, log_rank: int | None) -> Path | None:
    if log_rank is not None:
        ranked = sorted(log_dir.glob(f"{prefix}_rank{log_rank}_*.csv"))
        if len(ranked) > 1:
            joined = ", ".join(str(match) for match in ranked)
            raise ValueError(
                f"Multiple {prefix} logs for rank{log_rank} found in {log_dir}: {joined}"
            )
        if len(ranked) == 1:
            return ranked[0]

    plain_matches: List[Path] = []
    for match in sorted(log_dir.glob(f"{prefix}*.csv")):
        suffix = match.stem[len(prefix) :]
        if suffix.startswith("_rank"):
            continue
        plain_matches.append(match)

    if len(plain_matches) > 1:
        joined = ", ".join(str(match) for match in plain_matches)
        raise ValueError(f"Multiple {prefix} logs found in {log_dir}: {joined}")
    if len(plain_matches) == 1:
        return plain_matches[0]

    if log_rank is None:
        any_matches = sorted(log_dir.glob(f"{prefix}*.csv"))
        if len(any_matches) > 1:
            joined = ", ".join(str(match) for match in any_matches)
            raise ValueError(f"Multiple {prefix} logs found in {log_dir}: {joined}")
        if len(any_matches) == 1:
            return any_matches[0]

    return None


def resolve_logs_from_dir(
    log_dir: Path, log_rank: int | None
) -> tuple[Path, Path | None, Path | None]:
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log directory is not a directory: {log_dir}")

    train_log = find_unique_log(log_dir, "train_log", log_rank)
    if train_log is None:
        rank_note = f" for rank{log_rank}" if log_rank is not None else ""
        raise FileNotFoundError(f"train_log*.csv{rank_note} not found in {log_dir}")
    val_log = find_unique_log(log_dir, "val_log", log_rank)
    test_log = find_unique_log(log_dir, "test_log", log_rank)
    return train_log, val_log, test_log


def read_group_losses(
    csv_path: Path,
    x_column: str,
    loss_column: str,
    *,
    stage_column: str,
    stitch_stage: bool,
) -> tuple[Dict[int, List[float]], List[StageBoundary]]:
    ensure_csv(csv_path, csv_path.stem)

    grouped_losses: Dict[int, List[float]] = defaultdict(list)
    stage_boundaries: List[StageBoundary] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        x_key = x_column
        if x_key not in fieldnames and x_key == "step" and "batch" in fieldnames:
            x_key = "batch"
        loss_key = loss_column
        if loss_key not in fieldnames and loss_key == "loss_total":
            if "loss_noise_mse" in fieldnames:
                loss_key = "loss_noise_mse"
        if x_key not in fieldnames or loss_key not in fieldnames:
            raise KeyError(
                f"{csv_path}: Missing columns "
                f"(needed '{x_key}' and '{loss_key}')"
            )
        stage_key = stage_column if stage_column in fieldnames else None
        stage_rows: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for row in reader:
            epoch_raw = row.get(x_key)
            loss_raw = row.get(loss_key)
            if epoch_raw is None or loss_raw is None:
                continue
            epoch = int(epoch_raw)
            try:
                loss_val = float(loss_raw)
            except ValueError:
                continue
            stage_name = "stage_0"
            if stage_key:
                stage_raw = row.get(stage_key)
                if stage_raw is not None and str(stage_raw).strip():
                    stage_name = str(stage_raw).strip()
            stage_rows[stage_name].append((epoch, loss_val))

    if not stage_rows:
        return grouped_losses, stage_boundaries

    if not (stitch_stage and len(stage_rows) > 1):
        for rows in stage_rows.values():
            for x_val, loss_val in rows:
                grouped_losses[x_val].append(loss_val)
        return grouped_losses, stage_boundaries

    prev_end: int | None = None
    for stage_name, rows in stage_rows.items():
        xs = [x_val for x_val, _ in rows]
        if not xs:
            continue
        stage_start_raw = min(xs)
        stage_offset = 0
        if prev_end is not None and stage_start_raw <= prev_end:
            stage_offset = (prev_end + 1) - stage_start_raw

        mapped_xs: List[int] = []
        for x_val, loss_val in rows:
            mapped_x = x_val + stage_offset
            grouped_losses[mapped_x].append(loss_val)
            mapped_xs.append(mapped_x)

        stage_start = min(mapped_xs)
        stage_end = max(mapped_xs)
        if prev_end is not None:
            stage_boundaries.append((float(stage_start) - 0.5, stage_name))
        prev_end = stage_end

    return grouped_losses, stage_boundaries

def maybe_drop_incomplete_last_epoch(
    grouped_losses: Dict[int, List[float]],
    label: str,
) -> Dict[int, List[float]]:
    if len(grouped_losses) < 2:
        return grouped_losses

    epochs = sorted(grouped_losses)
    last_epoch = epochs[-1]
    reference_epoch = epochs[-2]
    last_count = len(grouped_losses.get(last_epoch, []))
    reference_count = len(grouped_losses.get(reference_epoch, []))
    if last_count != reference_count:
        print(
            f"{label}: dropping incomplete epoch {last_epoch} "
            f"(rows={last_count}, expected={reference_count})."
        )
        grouped_losses.pop(last_epoch, None)
    return grouped_losses


def read_epoch_averages(
    csv_path: Path,
    x_column: str,
    loss_column: str,
    *,
    label: str,
    stage_column: str,
    stitch_stage: bool,
    drop_incomplete_last: bool = True,
) -> tuple[EpochLoss, LossRange, List[StageBoundary]]:
    grouped_losses, stage_boundaries = read_group_losses(
        csv_path,
        x_column,
        loss_column,
        stage_column=stage_column,
        stitch_stage=stitch_stage,
    )
    if drop_incomplete_last:
        grouped_losses = maybe_drop_incomplete_last_epoch(grouped_losses, label)

    averages: List[Tuple[int, float]] = []
    ranges: LossRange = {}
    for epoch, values in grouped_losses.items():
        if not values:
            continue
        avg = sum(values) / len(values)
        averages.append((epoch, avg))
        ranges[epoch] = (min(values), max(values))
    averages.sort(key=lambda item: item[0])
    return averages, ranges, stage_boundaries


def parse_t_group_edges(raw: str | None) -> list[int]:
    if raw is None:
        return []
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(parts) < 2:
        raise ValueError("--t-group-edges requires at least 2 integers.")
    try:
        edges = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--t-group-edges must be comma-separated integers.") from exc
    if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
        raise ValueError("--t-group-edges must be strictly increasing.")
    return edges


def read_group_averages_by_t_edges(
    csv_path: Path,
    *,
    x_column: str,
    timestep_column: str,
    loss_column: str,
    stage_column: str,
    stitch_stage: bool,
    t_edges: Sequence[int],
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, LossRange], list[StageBoundary]]:
    ensure_csv(csv_path, csv_path.stem)
    if len(t_edges) < 2:
        raise ValueError("t_edges must contain at least two values.")

    grouped: dict[str, dict[int, list[float]]] = {}
    ranges: dict[str, LossRange] = {}
    stage_boundaries: list[StageBoundary] = []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        x_key = x_column
        if x_key not in fieldnames and x_key == "step" and "batch" in fieldnames:
            x_key = "batch"
        t_key = timestep_column
        loss_key = loss_column
        if loss_key not in fieldnames and loss_key == "loss_total":
            if "loss_noise_mse" in fieldnames:
                loss_key = "loss_noise_mse"
        if x_key not in fieldnames or t_key not in fieldnames or loss_key not in fieldnames:
            raise KeyError(
                f"{csv_path}: Missing columns (needed '{x_key}', '{t_key}', '{loss_key}')"
            )
        stage_key = stage_column if stage_column in fieldnames else None
        stage_rows: Dict[str, List[Tuple[int, int, float]]] = defaultdict(list)
        for row in reader:
            x_raw = row.get(x_key)
            t_raw = row.get(t_key)
            loss_raw = row.get(loss_key)
            if x_raw is None or t_raw is None or loss_raw is None:
                continue
            try:
                x_val = int(x_raw)
                t_val = int(t_raw)
                loss_val = float(loss_raw)
            except ValueError:
                continue
            stage_name = "stage_0"
            if stage_key:
                stage_raw = row.get(stage_key)
                if stage_raw is not None and str(stage_raw).strip():
                    stage_name = str(stage_raw).strip()
            stage_rows[stage_name].append((x_val, t_val, loss_val))

    prev_end: int | None = None
    for stage_name, rows in stage_rows.items():
        if not rows:
            continue
        xs = [x_val for x_val, _, _ in rows]
        stage_start_raw = min(xs)
        stage_offset = 0
        if stitch_stage and prev_end is not None and stage_start_raw <= prev_end:
            stage_offset = (prev_end + 1) - stage_start_raw
        mapped_stage_xs: list[int] = []
        for x_val, t_val, loss_val in rows:
            mapped_x = x_val + stage_offset
            mapped_stage_xs.append(mapped_x)
            for idx in range(len(t_edges) - 1):
                lo = int(t_edges[idx])
                hi = int(t_edges[idx + 1])
                if t_val < lo:
                    break
                if lo <= t_val < hi:
                    label = f"t{lo}-{hi}"
                    if label not in grouped:
                        grouped[label] = defaultdict(list)
                    grouped[label][mapped_x].append(loss_val)
                    break
        if mapped_stage_xs:
            stage_start = min(mapped_stage_xs)
            stage_end = max(mapped_stage_xs)
            if stitch_stage and prev_end is not None:
                stage_boundaries.append((float(stage_start) - 0.5, stage_name))
            prev_end = stage_end

    averages_by_label: dict[str, list[tuple[int, float]]] = {}
    for label, x_map in grouped.items():
        avgs: list[tuple[int, float]] = []
        rmap: LossRange = {}
        for x_val, vals in x_map.items():
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            avgs.append((x_val, avg))
            rmap[x_val] = (min(vals), max(vals))
        avgs.sort(key=lambda item: item[0])
        averages_by_label[label] = avgs
        ranges[label] = rmap
    return averages_by_label, ranges, stage_boundaries


def print_group_table(
    name: str,
    averages: EpochLoss,
    *,
    group_label: str,
    ranges: LossRange | None = None,
) -> None:
    if not averages:
        print(f"No {name} data to report.")
        return
    print(f"{name} {group_label} averages:")
    if group_label == "epoch" and ranges:
        for epoch, loss in averages:
            min_loss, max_loss = ranges.get(epoch, (loss, loss))
            print(
                f"  {group_label} {epoch:4d}: avg={loss:.6f} "
                f"min={min_loss:.6f} max={max_loss:.6f}"
            )
        return
    for x_val, loss in averages:
        print(f"  {group_label} {x_val:4d}: {loss:.6f}")


def make_t_bins(
    averages: EpochLoss,
    ranges: LossRange | None,
    *,
    t_bin_size: int,
) -> tuple[list[TBinPoint], list[float], list[float], list[str]]:
    if t_bin_size <= 0:
        raise ValueError("--t-bin-size must be > 0")

    grouped: dict[int, list[float]] = defaultdict(list)
    grouped_min: dict[int, list[float]] = defaultdict(list)
    grouped_max: dict[int, list[float]] = defaultdict(list)
    for t_val, avg in averages:
        bin_idx = int(t_val) // t_bin_size
        grouped[bin_idx].append(float(avg))
        if ranges is not None and t_val in ranges:
            min_loss, max_loss = ranges[t_val]
            grouped_min[bin_idx].append(float(min_loss))
            grouped_max[bin_idx].append(float(max_loss))
        else:
            grouped_min[bin_idx].append(float(avg))
            grouped_max[bin_idx].append(float(avg))

    bars: list[TBinPoint] = []
    mins: list[float] = []
    maxs: list[float] = []
    labels: list[str] = []
    for bin_idx in sorted(grouped):
        lo = bin_idx * t_bin_size
        hi = lo + t_bin_size - 1
        vals = grouped[bin_idx]
        bars.append((lo, sum(vals) / len(vals)))
        mins.append(min(grouped_min[bin_idx]))
        maxs.append(max(grouped_max[bin_idx]))
        labels.append(f"{lo}-{hi}")
    return bars, mins, maxs, labels


def print_t_bin_table(
    name: str,
    bars: Sequence[TBinPoint],
    mins: Sequence[float],
    maxs: Sequence[float],
    labels: Sequence[str],
) -> None:
    if not bars:
        print(f"No {name} timestep bins to report.")
        return
    print(f"{name} timestep-bin averages:")
    for (start_t, avg), min_loss, max_loss, label in zip(bars, mins, maxs, labels):
        _ = start_t
        print(f"  t {label:>9}: avg={avg:.6f} min={min_loss:.6f} max={max_loss:.6f}")


def read_video_stats(
    csv_path: Path,
    *,
    video_column: str,
    timestep_column: str,
    loss_column: str,
) -> list[tuple[str, float, float, float, int]]:
    ensure_csv(csv_path, csv_path.stem)
    video_losses: dict[str, list[float]] = defaultdict(list)
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if video_column not in fieldnames or timestep_column not in fieldnames or loss_column not in fieldnames:
            raise KeyError(
                f"{csv_path}: Missing columns (needed '{video_column}', '{timestep_column}' and '{loss_column}')"
            )
        for row in reader:
            video_raw = row.get(video_column)
            t_raw = row.get(timestep_column)
            loss_raw = row.get(loss_column)
            if video_raw is None or t_raw is None or loss_raw is None:
                continue
            video = str(video_raw).strip()
            if not video:
                continue
            try:
                t_val = int(t_raw)
                loss_val = float(loss_raw)
            except ValueError:
                continue
            video_losses[video].append(loss_val)

    stats: list[tuple[str, float, float, float, int]] = []
    for video, values in video_losses.items():
        if not values:
            continue
        sv = sorted(values)
        p95 = sv[int(0.95 * (len(sv) - 1))]
        stats.append((video, sum(values) / len(values), p95, max(values), len(values)))
    return stats


def build_video_t_bin_max(
    csv_path: Path,
    *,
    video_column: str,
    timestep_column: str,
    loss_column: str,
    t_edges: Sequence[int],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    ensure_csv(csv_path, csv_path.stem)
    labels = [f"t{int(t_edges[i])}-{int(t_edges[i+1])}" for i in range(len(t_edges) - 1)]
    out: dict[str, dict[str, float]] = defaultdict(dict)
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if video_column not in fieldnames or timestep_column not in fieldnames or loss_column not in fieldnames:
            raise KeyError(
                f"{csv_path}: Missing columns (needed '{video_column}', '{timestep_column}' and '{loss_column}')"
            )
        for row in reader:
            video_raw = row.get(video_column)
            t_raw = row.get(timestep_column)
            loss_raw = row.get(loss_column)
            if video_raw is None or t_raw is None or loss_raw is None:
                continue
            video = str(video_raw).strip()
            if not video:
                continue
            try:
                t_val = int(t_raw)
                loss_val = float(loss_raw)
            except ValueError:
                continue
            for i in range(len(t_edges) - 1):
                lo = int(t_edges[i])
                hi = int(t_edges[i + 1])
                if t_val < lo:
                    break
                if lo <= t_val < hi:
                    label = labels[i]
                    prev = out[video].get(label)
                    out[video][label] = loss_val if prev is None else max(prev, loss_val)
                    break
    return labels, out


def print_video_table(stats: Sequence[tuple[str, float, float, float, int]], *, top_k: int) -> None:
    if not stats:
        print("No video stats to report.")
        return
    shown = list(stats if top_k <= 0 else stats[:top_k])
    print("Video loss stats (video n avg p95 max):")
    for video, avg, p95, vmax, count in shown:
        print(f"  {video}: n={count} avg={avg:.6f} p95={p95:.6f} max={vmax:.6f}")


def plot_video_bars(
    stats: Sequence[tuple[str, float, float, float, int]],
    t_bin_labels: Sequence[str],
    video_t_bin_max: dict[str, dict[str, float]],
    *,
    sort_by: str,
    top_k: int,
    show_global_max: bool,
    output: Path | None,
) -> None:
    if not stats:
        print("Nothing to plot.")
        return
    key_idx = {"avg": 1, "p95": 2, "max": 3}[sort_by]
    ordered = sorted(stats, key=lambda item: item[key_idx], reverse=True)
    shown = ordered if top_k <= 0 else ordered[:top_k]
    print_video_table(shown, top_k=top_k if top_k > 0 else 0)

    labels = [f"{item[0]} (n={item[4]})" for item in shown]
    video_keys = [item[0] for item in shown]
    avgs = [item[1] for item in shown]
    p95s = [item[2] for item in shown]
    maxs = [item[3] for item in shown]
    ys = list(range(len(shown)))

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(shown))))
    ax.barh(ys, avgs, color="#1f77b4", alpha=0.85, label="avg")
    ax.scatter(p95s, ys, color="#ff7f0e", s=18, label="p95", zorder=3)
    if show_global_max:
        ax.scatter(maxs, ys, color="#d62728", s=18, label="max", zorder=3)
    t_colors = ["#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2"]
    for idx, t_label in enumerate(t_bin_labels):
        xs: list[float] = []
        ys_local: list[int] = []
        for y, video_key in enumerate(video_keys):
            val = video_t_bin_max.get(video_key, {}).get(t_label)
            if val is None:
                continue
            xs.append(val)
            ys_local.append(y)
        if xs:
            color = t_colors[idx % len(t_colors)]
            ax.scatter(xs, ys_local, color=color, s=14, marker="x", alpha=0.9, label=f"{t_label} max")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Loss")
    ax.set_ylabel("Video")
    ax.set_title(f"Video-wise Loss ({sort_by} desc, top {len(shown)})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"Saved plot to: {output}")
    plt.close(fig)


def plot_curves(
    curves: Sequence[CurveSpec],
    output: Path | None,
    *,
    group_label: str,
    step_bin_size: int,
    stage_boundaries: Sequence[StageBoundary] | None = None,
) -> None:
    if not any(data for _, data, _, _, _ in curves):
        print("Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data, ranges, marker, color in curves:
        if not data:
            continue
        epochs, losses = zip(*data)
        plot_marker = None if group_label == "step" else marker
        line_alpha = 0.25 if group_label == "step" else 1.0
        line_width = 0.6 if group_label == "step" else 1.5
        ax.plot(
            epochs,
            losses,
            marker=plot_marker,
            label=label,
            color=color,
            alpha=line_alpha,
            linewidth=line_width,
            zorder=1,
        )
        if group_label == "epoch" and ranges:
            band_x: List[float] = []
            band_min: List[float] = []
            band_max: List[float] = []
            for x_val, _ in data:
                minmax = ranges.get(x_val)
                if minmax is None:
                    continue
                min_loss, max_loss = minmax
                band_x.append(float(x_val))
                band_min.append(min_loss)
                band_max.append(max_loss)
            if band_x:
                ax.fill_between(
                    band_x,
                    band_min,
                    band_max,
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                    zorder=2,
                )
                ax.plot(
                    band_x,
                    band_min,
                    color=color,
                    alpha=0.45,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=3,
                )
                ax.plot(
                    band_x,
                    band_max,
                    color=color,
                    alpha=0.45,
                    linewidth=0.8,
                    linestyle="--",
                    zorder=3,
                )
        if group_label == "step" and step_bin_size > 1:
            bins: dict[int, list[float]] = {}
            for x_val, y_val in data:
                bin_idx = int(x_val) // step_bin_size
                if bin_idx not in bins:
                    bins[bin_idx] = [float(x_val), float(x_val), float(y_val), float(y_val)]
                else:
                    stats = bins[bin_idx]
                    stats[0] = min(stats[0], float(x_val))
                    stats[1] = max(stats[1], float(x_val))
                    stats[2] = min(stats[2], float(y_val))
                    stats[3] = max(stats[3], float(y_val))
            bin_x: list[float] = []
            bin_min: list[float] = []
            bin_max: list[float] = []
            for bin_idx in sorted(bins):
                min_x, max_x, min_y, max_y = bins[bin_idx]
                bin_x.append((min_x + max_x) * 0.5)
                bin_min.append(min_y)
                bin_max.append(max_y)
            ax.fill_between(
                bin_x,
                bin_min,
                bin_max,
                color=color,
                alpha=0.25,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                bin_x,
                bin_min,
                color=color,
                alpha=0.6,
                linewidth=1.0,
                zorder=3,
            )
            ax.plot(
                bin_x,
                bin_max,
                color=color,
                alpha=0.6,
                linewidth=1.0,
                zorder=3,
            )

    if stage_boundaries:
        for boundary_x, _ in stage_boundaries:
            ax.axvline(
                boundary_x,
                color="#666666",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
                zorder=0,
            )

    ax.set_title(f"Loss Transition per {group_label.capitalize()}")
    ax.set_xlabel(group_label.capitalize())
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if stage_boundaries:
        y_min, y_max = ax.get_ylim()
        label_y = y_max - (y_max - y_min) * 0.03
        for boundary_x, stage_name in stage_boundaries:
            ax.text(
                boundary_x + 0.2,
                label_y,
                f"stage={stage_name}",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color="#555555",
            )

    ax.legend()
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"Saved plot to: {output}")

    plt.close(fig)


def plot_curves_separate(
    curves: Sequence[CurveSpec],
    output: Path | None,
    *,
    group_label: str,
    step_bin_size: int,
    stage_boundaries: Sequence[StageBoundary] | None = None,
) -> None:
    active_curves = [(label, data, ranges, marker, color) for label, data, ranges, marker, color in curves if data]
    if not active_curves:
        print("Nothing to plot.")
        return

    rows = len(active_curves)
    fig, axes = plt.subplots(rows, 1, figsize=(11, max(3.5 * rows, 4.5)), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, (label, data, ranges, marker, color) in zip(axes, active_curves):
        xs, ys = zip(*data)
        plot_marker = None if group_label == "step" else marker
        line_alpha = 0.25 if group_label == "step" else 1.0
        line_width = 0.6 if group_label == "step" else 1.5
        ax.plot(
            xs,
            ys,
            marker=plot_marker,
            label=label,
            color=color,
            alpha=line_alpha,
            linewidth=line_width,
            zorder=1,
        )
        if group_label == "epoch" and ranges:
            band_x: List[float] = []
            band_min: List[float] = []
            band_max: List[float] = []
            for x_val, _ in data:
                minmax = ranges.get(x_val)
                if minmax is None:
                    continue
                min_loss, max_loss = minmax
                band_x.append(float(x_val))
                band_min.append(min_loss)
                band_max.append(max_loss)
            if band_x:
                ax.fill_between(
                    band_x,
                    band_min,
                    band_max,
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                    zorder=2,
                )
                ax.plot(band_x, band_min, color=color, alpha=0.45, linewidth=0.8, linestyle="--", zorder=3)
                ax.plot(band_x, band_max, color=color, alpha=0.45, linewidth=0.8, linestyle="--", zorder=3)
        if group_label == "step" and step_bin_size > 1:
            bins: dict[int, list[float]] = {}
            for x_val, y_val in data:
                bin_idx = int(x_val) // step_bin_size
                if bin_idx not in bins:
                    bins[bin_idx] = [float(x_val), float(x_val), float(y_val), float(y_val)]
                else:
                    stats = bins[bin_idx]
                    stats[0] = min(stats[0], float(x_val))
                    stats[1] = max(stats[1], float(x_val))
                    stats[2] = min(stats[2], float(y_val))
                    stats[3] = max(stats[3], float(y_val))
            bx: list[float] = []
            bmin: list[float] = []
            bmax: list[float] = []
            for bin_idx in sorted(bins):
                min_x, max_x, min_y, max_y = bins[bin_idx]
                bx.append((min_x + max_x) * 0.5)
                bmin.append(min_y)
                bmax.append(max_y)
            ax.fill_between(bx, bmin, bmax, color=color, alpha=0.25, linewidth=0, zorder=2)
            ax.plot(bx, bmin, color=color, alpha=0.6, linewidth=1.0, zorder=3)
            ax.plot(bx, bmax, color=color, alpha=0.6, linewidth=1.0, zorder=3)
        if stage_boundaries:
            for boundary_x, _ in stage_boundaries:
                ax.axvline(
                    boundary_x,
                    color="#666666",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=0,
                )

        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label)

    axes[-1].set_xlabel(group_label.capitalize())
    fig.suptitle(f"Loss Transition per {group_label.capitalize()} (separate)")
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"Saved plot to: {output}")
    plt.close(fig)


def plot_t_bars(
    *,
    train_data: EpochLoss,
    train_ranges: LossRange | None,
    output: Path | None,
    t_bin_size: int,
) -> None:
    bars, mins, maxs, labels = make_t_bins(
        train_data,
        train_ranges,
        t_bin_size=t_bin_size,
    )
    print_t_bin_table("Train", bars, mins, maxs, labels)
    if not bars:
        print("Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = list(range(len(bars)))
    avgs = [avg for _, avg in bars]
    bar_color = "#1f77b4"
    ax.bar(xs, avgs, color=bar_color, alpha=0.8, width=0.85, label="train avg")
    ax.vlines(xs, mins, maxs, color="#0f3b66", linewidth=1.4, alpha=0.9, label="min-max")
    ax.scatter(xs, mins, color="#0f3b66", s=10, alpha=0.9)
    ax.scatter(xs, maxs, color="#0f3b66", s=10, alpha=0.9)
    ax.set_title("Loss by Timestep Bin")
    ax.set_xlabel("Timestep Bin")
    ax.set_ylabel("Loss")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        print(f"Saved plot to: {output}")
    plt.close(fig)


def extract_run_tag(path: Path | None) -> str | None:
    """
    Try to pull a timestamp-ish suffix from log filenames like
    train_log_20251216_055222.csv -> 20251216_055222.
    Falls back to None when no suffix is present.
    """
    if not path:
        return None
    stem = path.stem

    for prefix in ("train_log_", "val_log_", "test_log_"):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return stem[len(prefix) :]

    parts = stem.split("_")
    if len(parts) >= 2 and all(part.isdigit() for part in parts[-2:]):
        return "_".join(parts[-2:])
    return None


def main() -> int:
    args = parse_args()
    train_log = args.train_log
    val_log = args.val_log
    test_log = args.test_log
    if args.log_dir:
        train_log, val_log, test_log = resolve_logs_from_dir(args.log_dir, args.log_rank)

    if train_log is None:
        raise ValueError("Specify --log-dir or --train-log.")

    group_by = args.group_by
    if group_by == "epoch":
        x_column = args.epoch_column
        group_label = "epoch"
    elif group_by == "step":
        x_column = args.step_column
        group_label = "step"
    elif group_by == "video":
        x_column = args.video_column
        group_label = "video"
    else:
        x_column = args.timestep_column
        group_label = "timestep"
    drop_incomplete = group_by == "epoch"
    t_group_edges = parse_t_group_edges(args.t_group_edges)
    t_group_mode = bool(t_group_edges) and group_by in {"epoch", "step"}

    stitch_stage = (not args.disable_stage_stitch) and group_by in {"epoch", "step"}

    if group_by == "video":
        if val_log or test_log:
            print("group-by=video: ignoring val/test logs and plotting train only.")
        video_t_edges = parse_t_group_edges(args.video_t_max_edges)
        stats = read_video_stats(
            train_log,
            video_column=args.video_column,
            timestep_column=args.timestep_column,
            loss_column=args.loss_column,
        )
        t_bin_labels, video_t_bin_max = build_video_t_bin_max(
            train_log,
            video_column=args.video_column,
            timestep_column=args.timestep_column,
            loss_column=args.loss_column,
            t_edges=video_t_edges,
        )
        output_path = args.output
        if output_path is None:
            filename = f"loss_video_{args.video_sort_by}.png"
            output_path = (args.log_dir / filename) if args.log_dir else train_log.with_name(filename)
            print(f"Saving figure automatically to: {output_path}")
        plot_video_bars(
            stats,
            t_bin_labels,
            video_t_bin_max,
            sort_by=args.video_sort_by,
            top_k=args.video_top_k,
            show_global_max=bool(args.video_show_global_max),
            output=output_path,
        )
        return 0

    if t_group_mode:
        if val_log or test_log:
            print("t-group mode: ignoring val/test logs and plotting train only.")
        train_groups, train_group_ranges, stage_boundaries = read_group_averages_by_t_edges(
            train_log,
            x_column=x_column,
            timestep_column=args.timestep_column,
            loss_column=args.loss_column,
            stage_column=args.stage_column,
            stitch_stage=stitch_stage,
            t_edges=t_group_edges,
        )
        palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        curves: List[CurveSpec] = []
        for idx, label in enumerate(sorted(train_groups)):
            data = train_groups[label]
            if not data:
                continue
            print_group_table(label, data, group_label=group_label, ranges=train_group_ranges.get(label))
            color = palette[idx % len(palette)]
            marker = "o" if group_by == "epoch" else None
            curves.append((label, data, train_group_ranges.get(label), marker, color))
        output_path = args.output
        if output_path is None:
            filename = f"loss_{group_by}_tgroups.png"
            output_path = (args.log_dir / filename) if args.log_dir else train_log.with_name(filename)
            print(f"Saving figure automatically to: {output_path}")
        if args.t_group_layout == "separate":
            plot_curves_separate(
                curves,
                output_path,
                group_label=group_label,
                step_bin_size=args.step_bin_size,
                stage_boundaries=stage_boundaries,
            )
        else:
            plot_curves(
                curves,
                output_path,
                group_label=group_label,
                step_bin_size=args.step_bin_size,
                stage_boundaries=stage_boundaries,
            )
        return 0

    train_data, train_ranges, stage_boundaries = read_epoch_averages(
        train_log,
        x_column,
        args.loss_column,
        label="train",
        stage_column=args.stage_column,
        stitch_stage=stitch_stage,
        drop_incomplete_last=drop_incomplete,
    )

    val_data: EpochLoss = []
    val_ranges: LossRange = {}
    if group_by != "t" and val_log:
        if val_log.exists():
            try:
                val_data, val_ranges, _ = read_epoch_averages(
                    val_log,
                    x_column,
                    args.loss_column,
                    label="val",
                    stage_column=args.stage_column,
                    stitch_stage=stitch_stage,
                    drop_incomplete_last=drop_incomplete,
                )
            except KeyError as exc:
                if group_by == "step":
                    print(f"Val log missing '{x_column}', skipping: {val_log}")
                else:
                    raise exc
        else:
            print(f"Val log not found, skipping: {val_log}")

    test_data: EpochLoss = []
    test_ranges: LossRange = {}
    if group_by != "t" and test_log:
        if test_log.exists():
            try:
                test_data, test_ranges, _ = read_epoch_averages(
                    test_log,
                    x_column,
                    args.loss_column,
                    label="test",
                    stage_column=args.stage_column,
                    stitch_stage=stitch_stage,
                    drop_incomplete_last=drop_incomplete,
                )
            except KeyError as exc:
                if group_by == "step":
                    print(f"Test log missing '{x_column}', skipping: {test_log}")
                else:
                    raise exc
        else:
            print(f"Test log not found, skipping: {test_log}")

    output_path = args.output
    if output_path is None:
        filename = f"loss_{group_by}.png"
        if args.log_dir:
            output_path = args.log_dir / filename
        else:
            run_tag = (
                extract_run_tag(val_log)
                or extract_run_tag(test_log)
                or extract_run_tag(train_log)
            )
            filename = f"loss_{group_by}_{run_tag}.png" if run_tag else filename
            output_path = train_log.with_name(filename)
        print(f"Saving figure automatically to: {output_path}")

    if group_by == "t":
        if val_log or test_log:
            print("group-by=t: ignoring val/test logs and plotting train only.")
        plot_t_bars(
            train_data=train_data,
            train_ranges=train_ranges,
            output=output_path,
            t_bin_size=args.t_bin_size,
        )
        return 0

    print_group_table("Train", train_data, group_label=group_label, ranges=train_ranges)
    use_custom_label = args.test_label is not None and not (val_data and test_data)
    curves: List[CurveSpec] = [("train", train_data, train_ranges, "o", "#1f77b4")]
    if val_data:
        label = args.test_label if use_custom_label and not test_data else "val"
        print_group_table(label, val_data, group_label=group_label, ranges=val_ranges)
        curves.append((label, val_data, val_ranges, "s", "#ff7f0e"))
    if test_data:
        label = args.test_label if use_custom_label and not val_data else "test"
        print_group_table(label, test_data, group_label=group_label, ranges=test_ranges)
        curves.append((label, test_data, test_ranges, "^", "#2ca02c"))
    plot_curves(
        curves,
        output_path,
        group_label=group_label,
        step_bin_size=args.step_bin_size,
        stage_boundaries=stage_boundaries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
