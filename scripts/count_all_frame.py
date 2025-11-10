#!/usr/bin/env python3
"""
count_all_frame.py
Count the total number of frames for every video file inside a folder.

Usage:
    python count_all_frame.py /path/to/folder --recursive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".gif")


def iter_video_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file())
    else:
        yield from (p for p in root.iterdir() if p.is_file())


def count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        # fallback: manual counting
        frame_count = 0
        while True:
            success, _ = cap.read()
            if not success:
                break
            frame_count += 1
    cap.release()
    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Count total frames of all videos in a folder.")
    parser.add_argument("folder", type=Path, help="Folder containing video files.")
    parser.add_argument(
        "--ext",
        nargs="*",
        default=list(VIDEO_EXTENSIONS),
        help="Video file extensions to include (default: common formats).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside subdirectories.",
    )
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 1

    extensions = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.ext)
    total_frames = 0
    matched_files = []

    for path in iter_video_files(folder, args.recursive):
        if path.suffix.lower() not in extensions:
            continue
        try:
            frames = count_frames(path)
            print(f"{path}: {frames} frames")
            total_frames += frames
            matched_files.append(path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)

    if not matched_files:
        print("No video files matched the given extensions.")
    else:
        print("-" * 40)
        print(f"Total frames across {len(matched_files)} videos: {total_frames}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
