#!/usr/bin/env python3
"""
Check the dataset split (train/val/test) and report how many videos are
longer/shorter than a threshold (default: 30 seconds).

Defaults match config/train.json:
- train_glob: /workspace/stereocraft/video_data/train/*.mp4
- dataset_split_ratios: [8, 1, 1]
- dataset_split_seed: 7

Usage:
    python scripts/check_split_durations.py
    python scripts/check_split_durations.py --threshold 20
    python scripts/check_split_durations.py --train-glob "/path/to/*.mp4"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from decord import VideoReader, cpu


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def split_paths(
    paths: Sequence[str],
    ratios: Sequence[float] | None,
    seed: int,
) -> Dict[str, List[str]]:
    """Replicate the train/val/test split logic used in training."""
    keys = ("train", "val", "test")
    if not ratios:
        return {"train": list(paths), "val": [], "test": []}

    ratios = [float(r) for r in ratios]
    if len(ratios) != 3:
        raise ValueError("dataset_split_ratios must have exactly three numbers: [train, val, test].")
    if any(r < 0 for r in ratios):
        raise ValueError("dataset_split_ratios cannot contain negative values.")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("dataset_split_ratios must sum to a positive value.")

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)

    normalized = [r / ratio_sum for r in ratios]
    raw_counts = [n * total for n in normalized]
    counts = [math.floor(v) for v in raw_counts]
    remainder = total - sum(counts)
    if remainder > 0:
        fractional_order = sorted(
            range(len(raw_counts)),
            key=lambda idx: (raw_counts[idx] - counts[idx]),
            reverse=True,
        )
        for idx in fractional_order[:remainder]:
            counts[idx] += 1

    split: Dict[str, List[str]] = {k: [] for k in keys}
    cursor = 0
    for key, count in zip(keys, counts):
        if count > 0:
            split[key] = shuffled[cursor : cursor + count]
        cursor += count
    return split


def video_duration_seconds(path: str) -> float:
    vr = VideoReader(path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    if fps <= 0:
        raise RuntimeError(f"FPS is zero/unknown for video: {path}")
    frame_count = len(vr)
    if frame_count <= 0:
        raise RuntimeError(f"No frames found in video: {path}")
    return frame_count / fps


def summarize_split(paths: Sequence[str], threshold: float) -> Dict[str, float]:
    long_count = 0
    short_count = 0
    unknown = 0
    for path in paths:
        try:
            duration = video_duration_seconds(path)
        except Exception as exc:
            print(f"[WARN] Could not read duration for {path}: {exc}", file=sys.stderr)
            unknown += 1
            continue
        if duration >= threshold:
            long_count += 1
        else:
            short_count += 1
    known = long_count + short_count
    def pct(x: int) -> float:
        return round((x / known * 100.0) if known else 0.0, 1)
    return {
        "total": len(paths),
        "known": known,
        "unknown": unknown,
        "long_count": long_count,
        "short_count": short_count,
        "long_pct": pct(long_count),
        "short_pct": pct(short_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the train/val/test split and report duration distribution."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/train.json"),
        help="Path to training config JSON (default: config/train.json).",
    )
    parser.add_argument(
        "--train-glob",
        type=str,
        default=None,
        help="Glob pattern for videos. Overrides config train_glob when set.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Override dataset_split_ratios, e.g., --ratios 8 1 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override dataset_split_seed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Duration threshold in seconds (default: 30).",
    )
    args = parser.parse_args()

    if args.config and not args.config.exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 1

    cfg = load_config(args.config) if args.config else {}

    train_glob = args.train_glob or cfg.get("train_glob")
    if not train_glob:
        print("train_glob is not set. Provide --train-glob or config train_glob.", file=sys.stderr)
        return 1

    ratios = args.ratios if args.ratios is not None else cfg.get("dataset_split_ratios")
    seed = args.seed if args.seed is not None else cfg.get("dataset_split_seed", 42)

    video_paths = sorted(glob.glob(train_glob))
    if not video_paths:
        print(f"No videos matched glob: {train_glob}", file=sys.stderr)
        return 1

    split = split_paths(video_paths, ratios, seed)

    print(f"train_glob: {train_glob}")
    print(f"dataset_split_ratios: {ratios if ratios is not None else '[all train]'} (seed={seed})")
    print(f"threshold: {args.threshold:.1f} seconds")
    print("-" * 60)

    for key in ("train", "val", "test"):
        paths = split.get(key, [])
        if not paths:
            print(f"{key}: 0 videos")
            continue
        stats = summarize_split(paths, args.threshold)
        print(
            f"{key}: total={stats['total']} | known={stats['known']} | unknown={stats['unknown']}"
        )
        print(
            f"  >= {args.threshold:.1f}s : {stats['long_count']} ({stats['long_pct']}%)"
        )
        print(
            f"  <  {args.threshold:.1f}s : {stats['short_count']} ({stats['short_pct']}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
