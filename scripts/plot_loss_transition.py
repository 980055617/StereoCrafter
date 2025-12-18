#!/usr/bin/env python3
"""
Aggregate train/test log CSVs and visualize average loss transitions per epoch.

Typical usage (weights/Test/ is the default destination for SD inpainting runs):
    python scripts/plot_train_test_loss.py \
        --train-log weights/Test/train_log.csv \
        --eval-log weights/Test/eval_log.csv \
        --output weights/Test/loss_transition.png

`--eval-log` can point to either eval/test logs (any CSV that stores epoch + loss).
When the same epoch appears multiple times the script averages those values before
plotting, so the resulting curves reflect epoch-level progression.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


EpochLoss = Sequence[Tuple[int, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/test loss transitions averaged per epoch."
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=Path("weights/Test/train_log.csv"),
        help="CSV that tracks training loss per step/epoch.",
    )
    parser.add_argument(
        "--eval-log",
        dest="eval_log",
        type=Path,
        default=Path("weights/Test/eval_log.csv"),
        help="CSV for validation/test loss per epoch (set to '' to skip).",
    )
    parser.add_argument(
        "--test-label",
        type=str,
        default="eval",
        help="Legend label for the validation/test curve.",
    )
    parser.add_argument(
        "--loss-column",
        type=str,
        default="loss_noise_mse",
        help="Column name that stores the loss values to average.",
    )
    parser.add_argument(
        "--epoch-column",
        type=str,
        default="epoch",
        help="Column name that stores epoch numbers.",
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


def read_epoch_averages(
    csv_path: Path,
    epoch_column: str,
    loss_column: str,
) -> EpochLoss:
    ensure_csv(csv_path, csv_path.stem)

    epoch_losses: Dict[int, List[float]] = defaultdict(list)
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if epoch_column not in reader.fieldnames or loss_column not in reader.fieldnames:
            raise KeyError(
                f"{csv_path}: Missing columns "
                f"(needed '{epoch_column}' and '{loss_column}')"
            )
        for row in reader:
            epoch_raw = row.get(epoch_column)
            loss_raw = row.get(loss_column)
            if epoch_raw is None or loss_raw is None:
                continue
            epoch = int(epoch_raw)
            try:
                loss_val = float(loss_raw)
            except ValueError:
                continue
            epoch_losses[epoch].append(loss_val)

    averages: List[Tuple[int, float]] = []
    for epoch, values in epoch_losses.items():
        if not values:
            continue
        avg = sum(values) / len(values)
        averages.append((epoch, avg))
    averages.sort(key=lambda item: item[0])
    return averages


def print_epoch_table(name: str, averages: EpochLoss) -> None:
    if not averages:
        print(f"No {name} data to report.")
        return
    print(f"{name} epoch averages:")
    for epoch, loss in averages:
        print(f"  epoch {epoch:4d}: {loss:.6f}")


def plot_curves(
    train_data: EpochLoss,
    test_data: EpochLoss,
    output: Path | None,
    test_label: str,
) -> None:
    if not train_data and not test_data:
        print("Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_data:
        train_epochs, train_losses = zip(*train_data)
        ax.plot(
            train_epochs,
            train_losses,
            marker="o",
            label="train",
            color="#1f77b4",
        )
    if test_data:
        test_epochs, test_losses = zip(*test_data)
        ax.plot(
            test_epochs,
            test_losses,
            marker="s",
            label=test_label,
            color="#ff7f0e",
        )

    ax.set_title("Loss Transition per Epoch")
    ax.set_xlabel("Epoch")
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


def display_available() -> bool:
    backend = plt.get_backend().lower()
    if "agg" in backend:
        return False

    # Presence of a display environment variable is enough for most cases.
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return False


def align_balanced_passes(
    train_data: EpochLoss, test_data: EpochLoss
) -> Tuple[EpochLoss, EpochLoss]:
    """
    Trim the newest epochs if train/test have run different counts after epoch 1.

    We assume any mismatch happens at the very end (latest epoch still running).
    When detected we drop the extra newest points from whichever side is longer
    so plots only show fully-completed epochs.
    """

    if not train_data or not test_data:
        return train_data, test_data

    train_tail = train_data[1:]
    test_tail = test_data[1:]
    if not train_tail or not test_tail:
        return train_data, test_data

    matched_len = min(len(train_tail), len(test_tail))
    if matched_len == len(train_tail) and matched_len == len(test_tail):
        return train_data, test_data

    if matched_len == 0:
        print("Train/test second epochs not both available yet, plotting only epoch 1.")
    else:
        print(
            "Latest train/test epochs mismatch, dropping unfinished epoch before plotting."
        )

    trimmed_train = list(train_data[:1]) + list(train_tail[:matched_len])
    trimmed_test = list(test_data[:1]) + list(test_tail[:matched_len])
    return trimmed_train, trimmed_test


def extract_run_tag(path: Path | None) -> str | None:
    """
    Try to pull a timestamp-ish suffix from log filenames like
    train_log_20251216_055222.csv -> 20251216_055222.
    Falls back to None when no suffix is present.
    """
    if not path:
        return None
    stem = path.stem

    for prefix in ("train_log_", "eval_log_"):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            return stem[len(prefix) :]

    parts = stem.split("_")
    if len(parts) >= 2 and all(part.isdigit() for part in parts[-2:]):
        return "_".join(parts[-2:])
    return None


def main() -> int:
    args = parse_args()
    train_data = read_epoch_averages(args.train_log, args.epoch_column, args.loss_column)

    test_data: EpochLoss = []
    if args.eval_log:
        if args.eval_log.exists():
            test_data = read_epoch_averages(
                args.eval_log, args.epoch_column, args.loss_column
            )
        else:
            print(f"Eval log not found, skipping: {args.eval_log}")

    train_data, test_data = align_balanced_passes(train_data, test_data)

    output_path = args.output
    if output_path is None:
        # Default to saving next to the train log.
        run_tag = extract_run_tag(args.eval_log) or extract_run_tag(args.train_log)
        filename = (
            f"loss_transition_{run_tag}.png" if run_tag else "loss_transition.png"
        )
        output_path = args.train_log.with_name(filename)
        print(f"Saving figure automatically to: {output_path}")

    print_epoch_table("Train", train_data)
    print_epoch_table(args.test_label.capitalize(), test_data)
    plot_curves(train_data, test_data, output_path, test_label=args.test_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
