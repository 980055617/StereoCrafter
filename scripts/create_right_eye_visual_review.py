"""Create fixed visual review sheets for right-eye inpainting candidates.

This is intentionally image-first. Pixel metrics have been a weak proxy for
right-eye usability in the 0160 overfit work, so this script standardizes the
human review artifacts: full-frame contact sheets plus fixed ROI zoom sheets.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path


@dataclass(frozen=True)
class Roi:
    name: str
    x: int
    y: int
    width: int
    height: int


def parse_int_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_candidate(raw: str) -> Candidate:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    return Candidate(label=label, path=Path(path))


def parse_roi(raw: str) -> Roi:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("roi must be NAME:X,Y,W,H")
    name, values = raw.split(":", 1)
    parts = [part.strip() for part in values.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("roi must be NAME:X,Y,W,H")
    try:
        x, y, width, height = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("roi width and height must be positive")
    return Roi(name=name.strip(), x=x, y=y, width=width, height=height)


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    return cap


def read_frame(path: Path, frame_idx: int) -> np.ndarray:
    cap = open_video(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_idx} from {path}")
    return frame


def video_info(path: Path) -> dict[str, int | float | str]:
    cap = open_video(path)
    info = {
        "path": str(path),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return info


def center_crop(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    if height > src_h or width > src_w:
        raise ValueError(f"crop {height}x{width} exceeds source {src_h}x{src_w}")
    top = (src_h - height) // 2
    left = (src_w - width) // 2
    return frame[top : top + height, left : left + width]


def split_train_tile(frame: np.ndarray, target_height: int, target_width: int) -> dict[str, np.ndarray]:
    src_h, src_w = frame.shape[:2]
    half_h = src_h // 2
    half_w = src_w // 2
    left = frame[:half_h, :half_w]
    gt_right = frame[:half_h, half_w : half_w * 2]
    mask = frame[half_h : half_h * 2, :half_w]
    warped = frame[half_h : half_h * 2, half_w : half_w * 2]
    return {
        "left": center_crop(left, target_height, target_width),
        "gt_right": center_crop(gt_right, target_height, target_width),
        "mask": center_crop(mask, target_height, target_width),
        "warped": center_crop(warped, target_height, target_width),
    }


def extract_candidate_right(frame: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    src_w = frame.shape[1]
    right = frame[:, src_w // 2 :]
    return center_crop(right, target_height, target_width)


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit_tile(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def make_full_sheet(
    *,
    train_parts: dict[str, np.ndarray],
    candidates: list[tuple[str, np.ndarray]],
    tile_width: int,
    tile_height: int,
) -> np.ndarray:
    tiles = [
        draw_label(fit_tile(train_parts["gt_right"], tile_width, tile_height), "gt_right"),
        draw_label(fit_tile(train_parts["warped"], tile_width, tile_height), "warped"),
    ]
    tiles.extend(draw_label(fit_tile(frame, tile_width, tile_height), label) for label, frame in candidates)

    cols = min(4, len(tiles))
    rows: list[np.ndarray] = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), cols):
        row_tiles = tiles[start : start + cols]
        row_tiles.extend([blank] * (cols - len(row_tiles)))
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def crop_roi(frame: np.ndarray, roi: Roi) -> np.ndarray:
    return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]


def make_zoom_sheet(
    *,
    roi: Roi,
    train_parts: dict[str, np.ndarray],
    candidates: list[tuple[str, np.ndarray]],
    zoom_scale: int,
) -> np.ndarray:
    tiles = [
        ("gt_right", crop_roi(train_parts["gt_right"], roi)),
        ("warped", crop_roi(train_parts["warped"], roi)),
    ]
    tiles.extend((label, crop_roi(frame, roi)) for label, frame in candidates)
    zoomed = [
        draw_label(
            cv2.resize(frame, (roi.width * zoom_scale, roi.height * zoom_scale), interpolation=cv2.INTER_NEAREST),
            label,
        )
        for label, frame in tiles
    ]
    return np.concatenate(zoomed, axis=1)


def write_manifest(
    *,
    output_dir: Path,
    train_tile: Path,
    candidates: list[Candidate],
    frames: list[int],
    rois: list[Roi],
    target_height: int,
    target_width: int,
) -> None:
    with (output_dir / "video_info.csv").open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["label", "path", "frames", "width", "height", "fps"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        train_info = video_info(train_tile)
        writer.writerow({"label": "train_tile", **train_info})
        for candidate in candidates:
            info = video_info(candidate.path)
            writer.writerow({"label": candidate.label, **info})

    with (output_dir / "README.md").open("w", encoding="utf-8") as fp:
        fp.write("# Right-Eye Visual Review\n\n")
        fp.write(f"Train tile: `{train_tile}`\n\n")
        fp.write(f"Target crop: `{target_height}x{target_width}` center crop\n\n")
        fp.write("Candidates:\n\n")
        for candidate in candidates:
            fp.write(f"- `{candidate.label}`: `{candidate.path}`\n")
        fp.write("\nFrames:\n\n")
        for frame_idx in frames:
            fp.write(f"- `frame_{frame_idx:04d}_full.jpg`\n")
            for roi in rois:
                fp.write(f"- `frame_{frame_idx:04d}_{roi.name}.jpg`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tile", default="video_data/train/0160_train.mp4", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True, help="LABEL=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames", type=parse_int_list, default=parse_int_list("25,75,125"))
    parser.add_argument("--target-height", type=int, default=576)
    parser.add_argument("--target-width", type=int, default=1024)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--tile-height", type=int, default=180)
    parser.add_argument("--zoom-scale", type=int, default=2)
    parser.add_argument(
        "--roi",
        action="append",
        type=parse_roi,
        default=[
            Roi("sign_center", 250, 230, 420, 180),
            Roi("train_right", 660, 90, 300, 260),
            Roi("foreground_left", 0, 260, 280, 250),
        ],
        help="NAME:X,Y,W,H in target-crop coordinates",
    )
    args = parser.parse_args()

    candidates: list[Candidate] = args.candidate
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        if not candidate.path.exists():
            raise FileNotFoundError(candidate.path)
    if not args.train_tile.exists():
        raise FileNotFoundError(args.train_tile)

    for frame_idx in args.frames:
        train_parts = split_train_tile(
            read_frame(args.train_tile, frame_idx),
            target_height=args.target_height,
            target_width=args.target_width,
        )
        candidate_frames = [
            (
                candidate.label,
                extract_candidate_right(
                    read_frame(candidate.path, frame_idx),
                    target_height=args.target_height,
                    target_width=args.target_width,
                ),
            )
            for candidate in candidates
        ]

        full_sheet = make_full_sheet(
            train_parts=train_parts,
            candidates=candidate_frames,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
        full_path = output_dir / f"frame_{frame_idx:04d}_full.jpg"
        cv2.imwrite(str(full_path), full_sheet)
        print(full_path)

        for roi in args.roi:
            zoom_sheet = make_zoom_sheet(
                roi=roi,
                train_parts=train_parts,
                candidates=candidate_frames,
                zoom_scale=args.zoom_scale,
            )
            zoom_path = output_dir / f"frame_{frame_idx:04d}_{roi.name}.jpg"
            cv2.imwrite(str(zoom_path), zoom_sheet)
            print(zoom_path)

    write_manifest(
        output_dir=output_dir,
        train_tile=args.train_tile,
        candidates=candidates,
        frames=args.frames,
        rois=args.roi,
        target_height=args.target_height,
        target_width=args.target_width,
    )


if __name__ == "__main__":
    main()
