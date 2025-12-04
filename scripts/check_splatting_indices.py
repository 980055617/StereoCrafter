#!/usr/bin/env python3
"""Check for numbering gaps among *_splatting_results files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a directory for files like 0001_splatting_results.mp4 and report "
            "any missing or duplicated indices up to the last observed index."
        )
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("video_data/splatting"),
        help="Folder that contains the splatting result files.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_splatting_results",
        help="Suffix that follows the numeric prefix (set empty to ignore).",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".mp4",
        help="File extension to match (set empty to accept every extension).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First expected index. Defaults to the smallest index that was found.",
    )
    return parser.parse_args()


def iter_matches(
    directory: Path, suffix: str, extension: str
) -> Iterable[Tuple[Path, int, int]]:
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if extension and path.suffix != extension:
            continue
        stem = path.stem
        if suffix:
            if not stem.endswith(suffix):
                continue
            numeric = stem[: -len(suffix)]
        else:
            numeric = stem
        if not numeric.isdigit():
            continue
        yield path, int(numeric), len(numeric)


def format_index(value: int, width: int) -> str:
    return f"{value:0{width}d}" if width > 0 else str(value)


def main() -> int:
    args = parse_args()
    directory = args.directory

    if not directory.exists():
        print(f"[error] Directory not found: {directory}")
        return 2
    if not directory.is_dir():
        print(f"[error] Path is not a directory: {directory}")
        return 2

    counts: Counter[int] = Counter()
    widths = []
    for path, idx, width in iter_matches(directory, args.suffix, args.extension):
        counts[idx] += 1
        widths.append(width)

    if not counts:
        print(
            f"[warn] No files matched in {directory} "
            f"(suffix='{args.suffix}', extension='{args.extension}')"
        )
        return 2

    start_index = args.start if args.start is not None else min(counts)
    max_index = max(counts)
    pad_width = max(widths, default=0)

    missing = [
        idx for idx in range(start_index, max_index + 1) if counts[idx] == 0
    ]
    duplicates = sorted(idx for idx, amount in counts.items() if amount > 1)

    print(
        f"Scanned {sum(counts.values())} files between "
        f"{format_index(start_index, pad_width)} and "
        f"{format_index(max_index, pad_width)} in {directory}."
    )

    exit_code = 0
    if missing:
        formatted = ", ".join(format_index(idx, pad_width) for idx in missing)
        print(f"[missing] {len(missing)} gap(s): {formatted}")
        exit_code = 1
    else:
        print("[missing] None 🎉")

    if duplicates:
        formatted = ", ".join(format_index(idx, pad_width) for idx in duplicates)
        print(f"[duplicate] Indices appear multiple times: {formatted}")
        exit_code = 1
    else:
        print("[duplicate] None")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
