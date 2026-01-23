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
CurveSpec = Tuple[str, EpochLoss, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/val/test loss transitions averaged by epoch or step."
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
        choices=("epoch", "step"),
        default="epoch",
        help="X-axis grouping mode: average by epoch or step.",
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
        "--step-bin-size",
        type=int,
        default=100,
        help="Bin size for step min/max envelope (used when --group-by=step).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the rendered figure (PNG).",
    )
    return parser.parse_args()


def ensure_csv(path: Path, label: str) -> None:
    if not path:
        raise ValueError(f"{label} log path is empty.")
    if not path.exists():
        raise FileNotFoundError(f"{label} log not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} log is not a file: {path}")


def find_unique_log(log_dir: Path, prefix: str) -> Path | None:
    matches = sorted(log_dir.glob(f"{prefix}*.csv"))
    if not matches:
        return None
    if len(matches) > 1:
        joined = ", ".join(str(match) for match in matches)
        raise ValueError(f"Multiple {prefix} logs found in {log_dir}: {joined}")
    return matches[0]


def resolve_logs_from_dir(log_dir: Path) -> tuple[Path, Path | None, Path | None]:
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log directory is not a directory: {log_dir}")

    train_log = find_unique_log(log_dir, "train_log")
    if train_log is None:
        raise FileNotFoundError(f"train_log*.csv not found in {log_dir}")
    val_log = find_unique_log(log_dir, "val_log")
    test_log = find_unique_log(log_dir, "test_log")
    return train_log, val_log, test_log


def read_group_losses(
    csv_path: Path,
    x_column: str,
    loss_column: str,
) -> Dict[int, List[float]]:
    ensure_csv(csv_path, csv_path.stem)

    epoch_losses: Dict[int, List[float]] = defaultdict(list)
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
            epoch_losses[epoch].append(loss_val)
    return epoch_losses

def maybe_drop_incomplete_last_epoch(
    epoch_losses: Dict[int, List[float]],
    label: str,
) -> Dict[int, List[float]]:
    if len(epoch_losses) < 2:
        return epoch_losses

    epochs = sorted(epoch_losses)
    last_epoch = epochs[-1]
    reference_epoch = epochs[-2]
    last_count = len(epoch_losses.get(last_epoch, []))
    reference_count = len(epoch_losses.get(reference_epoch, []))
    if last_count != reference_count:
        print(
            f"{label}: dropping incomplete epoch {last_epoch} "
            f"(rows={last_count}, expected={reference_count})."
        )
        epoch_losses.pop(last_epoch, None)
    return epoch_losses


def read_epoch_averages(
    csv_path: Path,
    x_column: str,
    loss_column: str,
    *,
    label: str,
    drop_incomplete_last: bool = True,
) -> EpochLoss:
    epoch_losses = read_group_losses(csv_path, x_column, loss_column)
    if drop_incomplete_last:
        epoch_losses = maybe_drop_incomplete_last_epoch(epoch_losses, label)

    averages: List[Tuple[int, float]] = []
    for epoch, values in epoch_losses.items():
        if not values:
            continue
        avg = sum(values) / len(values)
        averages.append((epoch, avg))
    averages.sort(key=lambda item: item[0])
    return averages


def print_group_table(name: str, averages: EpochLoss, *, group_label: str) -> None:
    if not averages:
        print(f"No {name} data to report.")
        return
    print(f"{name} {group_label} averages:")
    for epoch, loss in averages:
        print(f"  {group_label} {epoch:4d}: {loss:.6f}")


def plot_curves(
    curves: Sequence[CurveSpec],
    output: Path | None,
    *,
    group_label: str,
    step_bin_size: int,
) -> None:
    if not any(data for _, data, _, _ in curves):
        print("Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data, marker, color in curves:
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

    ax.set_title(f"Loss Transition per {group_label.capitalize()}")
    ax.set_xlabel(group_label.capitalize())
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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
        train_log, val_log, test_log = resolve_logs_from_dir(args.log_dir)

    if train_log is None:
        raise ValueError("Specify --log-dir or --train-log.")

    group_by = args.group_by
    x_column = args.epoch_column if group_by == "epoch" else args.step_column
    drop_incomplete = group_by == "epoch"

    train_data = read_epoch_averages(
        train_log,
        x_column,
        args.loss_column,
        label="train",
        drop_incomplete_last=drop_incomplete,
    )

    val_data: EpochLoss = []
    if val_log:
        if val_log.exists():
            try:
                val_data = read_epoch_averages(
                    val_log,
                    x_column,
                    args.loss_column,
                    label="val",
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
    if test_log:
        if test_log.exists():
            try:
                test_data = read_epoch_averages(
                    test_log,
                    x_column,
                    args.loss_column,
                    label="test",
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

    print_group_table("Train", train_data, group_label=group_by)
    use_custom_label = args.test_label is not None and not (val_data and test_data)
    curves: List[CurveSpec] = [("train", train_data, "o", "#1f77b4")]
    if val_data:
        label = args.test_label if use_custom_label and not test_data else "val"
        print_group_table(label, val_data, group_label=group_by)
        curves.append((label, val_data, "s", "#ff7f0e"))
    if test_data:
        label = args.test_label if use_custom_label and not val_data else "test"
        print_group_table(label, test_data, group_label=group_by)
        curves.append((label, test_data, "^", "#2ca02c"))
    plot_curves(
        curves,
        output_path,
        group_label=group_by,
        step_bin_size=args.step_bin_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
