#!/usr/bin/env python3
"""
Utility script to ensure splatting result videos stay in sync with their reference clips.

Common usage patterns:
  * Delete mismatched outputs immediately:
        python scripts/cleanup_splatting_results.py \
            --reference-dir video_data/left_eye --result-dir video_data/splatting
  * Inspect mismatches without deleting (確認のみ):
        python scripts/cleanup_splatting_results.py --dry-run --verbose
            # `--dry-run` prints would-delete entries, `--verbose` shows per-file stats.

By default it compares frame counts/resolution/fps and deletes the splatting result
whenever a mismatch is detected. You can optionally sample decoded frames for a stricter
comparison via `--sample-frames`.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np


@dataclass
class VideoMeta:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare reference videos with *_splatting_results outputs and remove "
            "results that do not match."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("video_data/left_eye"),
        help="Directory containing the reference/original videos.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("video_data/splatting"),
        help="Directory containing *_splatting_results videos.",
    )
    parser.add_argument(
        "--reference-suffix",
        type=str,
        default="",
        help="Suffix to strip from reference stems when matching (e.g. '_train').",
    )
    parser.add_argument(
        "--result-suffix",
        type=str,
        default="_splatting_results",
        help="Suffix to strip from result stems before matching.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".mp4",
        help="Video file extension to scan for in both folders.",
    )
    parser.add_argument(
        "--fps-tol",
        type=float,
        default=0.01,
        help="Allowed absolute FPS difference before flagging a mismatch.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=0,
        help=(
            "Number of frames to sample (spread across the clip) for pixel-level checks. "
            "Set to 0 to skip sampling."
        ),
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=0,
        help=(
            "Maximum per-channel absolute pixel difference allowed when comparing sampled frames."
        ),
    )
    parser.add_argument(
        "--enforce-resolution",
        action="store_true",
        help="Treat differing resolutions as mismatches that trigger deletion.",
    )
    parser.add_argument(
        "--confirm-frame-count",
        action="store_true",
        help=(
            "When frame counts differ, re-read every frame to verify the counts before deciding."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would be deleted without removing anything.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-file comparison logs.",
    )
    return parser.parse_args()


def strip_suffix(value: str, suffix: str) -> str:
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def read_video_meta(path: Path) -> VideoMeta:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return VideoMeta(path=path, frame_count=frame_count, fps=fps, width=width, height=height)


def build_reference_index(
    reference_dir: Path, reference_suffix: str, extension: str
) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for video in sorted(reference_dir.glob(f"*{extension}")):
        key = strip_suffix(video.stem, reference_suffix)
        if not key:
            continue
        index[key] = video
    return index


def select_sample_indices(frame_count: int, sample_frames: int) -> Sequence[int]:
    if sample_frames <= 0 or frame_count <= 0:
        return []
    sample_frames = min(sample_frames, frame_count)
    positions = np.linspace(0, frame_count - 1, sample_frames)
    return [int(round(pos)) for pos in positions]


def read_frame_at(capture: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    if not ok:
        return None
    return frame


def count_frames_precisely(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for frame recount: {path}")

    count = 0
    try:
        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            count += 1
    finally:
        capture.release()
    return count


def sample_frames_match(
    reference_path: Path,
    result_path: Path,
    sample_indices: Sequence[int],
    pixel_threshold: int,
) -> Tuple[bool, Optional[str]]:
    if not sample_indices:
        return True, None

    ref_cap = cv2.VideoCapture(str(reference_path))
    res_cap = cv2.VideoCapture(str(result_path))
    if not ref_cap.isOpened():
        return False, f"Failed to open reference video {reference_path}"
    if not res_cap.isOpened():
        return False, f"Failed to open result video {result_path}"

    try:
        for idx in sample_indices:
            ref_frame = read_frame_at(ref_cap, idx)
            res_frame = read_frame_at(res_cap, idx)
            if ref_frame is None or res_frame is None:
                return False, f"Failed to read frame {idx}"

            diff = cv2.absdiff(ref_frame, res_frame)
            if diff.size == 0:
                return False, f"Empty diff for frame {idx}"
            if diff.max() > pixel_threshold:
                return False, f"Frame {idx} differs (max diff {int(diff.max())})"
    finally:
        ref_cap.release()
        res_cap.release()
    return True, None


def compare_videos(
    reference_path: Path,
    result_path: Path,
    fps_tol: float,
    sample_frames: int,
    pixel_threshold: int,
    enforce_resolution: bool,
    confirm_frame_count: bool,
) -> Tuple[bool, List[str], List[str]]:
    issues: List[str] = []
    warnings: List[str] = []
    try:
        ref_meta = read_video_meta(reference_path)
    except RuntimeError as exc:
        raise RuntimeError(f"Reference video unreadable: {reference_path}") from exc
    try:
        res_meta = read_video_meta(result_path)
    except RuntimeError as exc:
        issues.append(str(exc))
        return False, issues, warnings

    ref_frames = ref_meta.frame_count
    res_frames = res_meta.frame_count

    frame_counts_mismatch = ref_frames != res_frames
    if frame_counts_mismatch and confirm_frame_count:
        try:
            precise_ref = count_frames_precisely(reference_path)
            precise_res = count_frames_precisely(result_path)
        except RuntimeError as exc:
            issues.append(str(exc))
            return False, issues, warnings

        if precise_ref != ref_frames or precise_res != res_frames:
            warnings.append(
                "Frame recount adjusted metadata counts: "
                f"{ref_frames}->{precise_ref} (reference), "
                f"{res_frames}->{precise_res} (result)"
            )
        ref_frames = precise_ref
        res_frames = precise_res
        ref_meta.frame_count = ref_frames
        res_meta.frame_count = res_frames
        frame_counts_mismatch = ref_frames != res_frames

    if frame_counts_mismatch:
        issues.append(f"frame count mismatch ({ref_frames} vs {res_frames})")
    if ref_meta.width != res_meta.width or ref_meta.height != res_meta.height:
        warning_msg = (
            f"resolution mismatch ({ref_meta.width}x{ref_meta.height} vs "
            f"{res_meta.width}x{res_meta.height})"
        )
        if enforce_resolution:
            issues.append(warning_msg)
        else:
            warnings.append(warning_msg)
    if not math.isfinite(ref_meta.fps) or not math.isfinite(res_meta.fps):
        issues.append("FPS metadata missing")
    elif abs(ref_meta.fps - res_meta.fps) > fps_tol:
        issues.append(f"fps mismatch ({ref_meta.fps:.3f} vs {res_meta.fps:.3f})")

    if not issues and sample_frames > 0:
        indices = select_sample_indices(ref_meta.frame_count, sample_frames)
        frames_equal, reason = sample_frames_match(
            reference_path, result_path, indices, pixel_threshold
        )
        if not frames_equal:
            issues.append(reason or "sampled frames differ")

    return len(issues) == 0, issues, warnings


def ensure_directory(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{kind} path is not a directory: {path}")


def main() -> int:
    args = parse_args()
    ensure_directory(args.reference_dir, "Reference")
    ensure_directory(args.result_dir, "Result")

    reference_index = build_reference_index(
        args.reference_dir, args.reference_suffix, args.extension
    )
    if not reference_index:
        print("No reference videos found; aborting.", file=sys.stderr)
        return 1

    total_results = 0
    deleted = 0
    skipped_missing = 0
    mismatched: List[Tuple[Path, List[str]]] = []

    for result_video in sorted(args.result_dir.glob(f"*{args.extension}")):
        total_results += 1
        key = strip_suffix(result_video.stem, args.result_suffix)
        reference_path = reference_index.get(key)
        if not reference_path:
            skipped_missing += 1
            if args.verbose:
                print(f"[skip] No reference found for {result_video.name}")
            continue

        matches, issues, warnings = compare_videos(
            reference_path,
            result_video,
            args.fps_tol,
            args.sample_frames,
            args.pixel_threshold,
            args.enforce_resolution,
            args.confirm_frame_count,
        )
        if warnings and args.verbose:
            print(f"[warn] {result_video.name}: {', '.join(warnings)}")
        if matches:
            if args.verbose:
                print(f"[keep] {result_video.name} matches {reference_path.name}")
            continue

        mismatched.append((result_video, issues))
        if args.dry_run:
            print(f"[dry-run] Would delete {result_video.name}: {', '.join(issues)}")
            continue

        result_video.unlink()
        deleted += 1
        print(f"[delete] {result_video.name}: {', '.join(issues)}")

    kept = total_results - deleted
    print(
        f"Checked {total_results} result videos | deleted {deleted} | "
        f"missing reference {skipped_missing} | kept {kept}"
    )

    if args.dry_run and mismatched:
        print("\nMismatched files (dry-run):")
        for path, issues in mismatched:
            print(f"  - {path.name}: {', '.join(issues)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
