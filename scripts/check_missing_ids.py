#!/usr/bin/env python3
# =============================================
# File: /workspace/stereocraft/scripts/check_missing_ids.py
# ---------------------------------------------
# 目的: video_data内の欠番検出
# =============================================

"""Scan subfolders and report missing numeric IDs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def to_ranges(values: Sequence[int]) -> List[str]:
    """Convert sorted ints to compact ranges like '1-3', '5'."""
    if not values:
        return []
    ranges: List[Tuple[int, int]] = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        ranges.append((start, prev))
        start = prev = v
    ranges.append((start, prev))
    out: List[str] = []
    for lo, hi in ranges:
        if lo == hi:
            out.append(f"{lo:04d}")
        else:
            out.append(f"{lo:04d}-{hi:04d}")
    return out


def find_missing(nums: Sequence[int]) -> List[int]:
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    present = set(nums)
    return [v for v in range(lo, hi + 1) if v not in present]


def scan_folder(folder: Path, ext: str) -> List[int]:
    pattern = re.compile(r"^(\d+)_.*\." + re.escape(ext) + r"$", re.IGNORECASE)
    ids: List[int] = []
    for path in folder.glob(f"*.{ext}"):
        m = pattern.match(path.name)
        if not m:
            continue
        try:
            ids.append(int(m.group(1)))
        except ValueError:
            continue
    return ids


def iter_subfolders(root: Path) -> Iterable[Path]:
    for child in root.iterdir():
        if child.is_dir():
            yield child


def main() -> int:
    parser = argparse.ArgumentParser(description="Report missing numeric IDs per video_data subfolder.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("video_data"),
        help="Root folder containing subfolders to scan (default: video_data).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        help="File extension to match (default: mp4).",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        print(f"Root not found: {root}")
        return 1
    any_output = False
    for sub in sorted(iter_subfolders(root)):
        ids = scan_folder(sub, args.ext)
        if not ids:
            continue
        missing = find_missing(ids)
        if not missing:
            continue
        any_output = True
        print(f"{sub.name}: min={min(ids):04d} max={max(ids):04d} count={len(ids)} missing={len(missing)}")
        print("  missing ranges:", ", ".join(to_ranges(missing)))
    if not any_output:
        print("No gaps found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
