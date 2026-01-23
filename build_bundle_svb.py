#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zlib

try:
    import lz4.frame as lz4f  # type: ignore
except Exception:
    lz4f = None

try:
    import pycocotools.mask as mask_utils  # type: ignore
except Exception:
    mask_utils = None

MAGIC = b"SVB1"
VERSION = 2

TYPE_OTHER = 0
TYPE_PERSON = 1
TYPE_ANIMAL = 2

ROT_Q = (0, 0, 0, 32767)
COCO_LHIP = 11
COCO_RHIP = 12


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class TrackState:
    anchor_z: Optional[float] = None
    joints_rel: Optional[np.ndarray] = None
    kp_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Quest SVB bundle from video/depth/metadata.")
    parser.add_argument("--video_mp4", required=True, help="Input SBS video (mp4).")
    parser.add_argument("--depth_npz", required=True, help="Input depth npz (T,H,W).")
    parser.add_argument("--metadata_json", required=True, help="Input metadata json.")
    parser.add_argument("--out_bundle", required=True, help="Output bundle.svb path.")
    parser.add_argument("--left_eye_origin", choices=["full", "left"], default="full")
    parser.add_argument("--align128", type=int, choices=[0, 1], default=1)
    parser.add_argument("--crop_mode", choices=["topleft"], default="topleft")
    parser.add_argument("--fovx_deg", type=float, default=70.0)
    parser.add_argument("--sample_k", type=int, default=7)
    parser.add_argument("--conf_th", type=float, default=0.4)
    parser.add_argument("--ema_alpha", type=float, default=0.8)
    parser.add_argument("--quant_pos_scale", type=float, default=0.002)
    parser.add_argument("--quant_joint_scale", type=float, default=0.002)
    parser.add_argument("--frame_compress", choices=["none", "zlib", "lz4"], default="lz4")
    parser.add_argument("--anchor_from", choices=["mask", "bbox"], default="mask")
    parser.add_argument("--fps", type=float, default=None, help="Manual fallback if video probing fails.")
    parser.add_argument("--width", type=int, default=None, help="Manual fallback if video probing fails.")
    parser.add_argument("--height", type=int, default=None, help="Manual fallback if video probing fails.")
    parser.add_argument("--transcode_video", type=int, choices=[0, 1], default=1)
    parser.add_argument("--transcode_crf", type=int, default=18)
    parser.add_argument("--transcode_preset", type=str, default="veryfast")
    parser.add_argument("--transcode_profile", choices=["baseline", "main"], default="main")
    parser.add_argument("--transcode_level", type=str, default="4.1")
    parser.add_argument("--transcode_audio_bitrate", type=str, default="128k")
    parser.add_argument("--transcode_audio_rate", type=int, default=48000)
    parser.add_argument("--debug_meta", type=int, choices=[0, 1], default=0)
    parser.add_argument("--debug_meta_every", type=int, default=50)
    parser.add_argument("--debug_meta_first_frames", type=int, default=3)
    parser.add_argument("--debug_meta_max_objects", type=int, default=10)
    parser.add_argument("--debug_meta_decode_after", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def parse_frame_rate(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            num_f = float(num)
            den_f = float(den)
            if den_f != 0:
                return num_f / den_f
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def resolve_bin(env_key: str, default_name: str, required: bool = True) -> Optional[str]:
    override = os.environ.get(env_key)
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        resolved = shutil.which(override)
        if resolved:
            return resolved
        if required:
            raise RuntimeError(f"{env_key}={override} not found or not executable.")
        return None
    resolved = shutil.which(default_name)
    if resolved is None and required:
        raise RuntimeError(f"{default_name} not found. Please install ffmpeg.")
    return resolved


def require_ffmpeg() -> str:
    return resolve_bin("FFMPEG_BIN", "ffmpeg", required=True)  # type: ignore[return-value]


def require_ffprobe() -> str:
    return resolve_bin("FFPROBE_BIN", "ffprobe", required=True)  # type: ignore[return-value]


def list_ffmpeg_video_encoders() -> List[str]:
    ffmpeg_bin = require_ffmpeg()
    result = subprocess.run(
        [ffmpeg_bin, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffmpeg -encoders failed:\n{stderr}")
    encoders: List[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("--") or line.startswith("Encoders:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.append(parts[1])
    return encoders


def select_h264_encoder() -> str:
    encoders = set(list_ffmpeg_video_encoders())
    if "libx264" in encoders:
        return "libx264"
    if "libopenh264" in encoders:
        return "libopenh264"
    raise RuntimeError(
        "No H.264 encoder found (libx264/libopenh264). Install ffmpeg with H.264 support."
    )


def map_profile_for_encoder(encoder: str, profile: str) -> str:
    if encoder == "libopenh264":
        return "constrained_baseline" if profile == "baseline" else "main"
    return profile


def run_ffprobe_json(cmd: Sequence[str]) -> Dict[str, Any]:
    ffprobe_bin = require_ffprobe()
    cmd = [ffprobe_bin, *cmd[1:]]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffprobe failed: {' '.join(cmd)}\n{stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse ffprobe JSON output.") from exc


def probe_ffprobe_avg_fps(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "json",
        path,
    ]
    data = run_ffprobe_json(cmd)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no video stream for {path}")
    fps = parse_frame_rate(streams[0].get("avg_frame_rate"))
    if fps <= 0:
        raise RuntimeError("Failed to parse avg_frame_rate from ffprobe.")
    return fps


def probe_ffprobe_has_audio(path: str) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        path,
    ]
    data = run_ffprobe_json(cmd)
    streams = data.get("streams") or []
    return bool(streams)


def transcode_for_quest(
    input_path: str, output_path: str, fps: float, args: argparse.Namespace
) -> Dict[str, Any]:
    ffmpeg_bin = require_ffmpeg()
    if fps <= 0:
        raise RuntimeError("Invalid FPS for transcoding.")
    encoder = select_h264_encoder()
    profile = map_profile_for_encoder(encoder, args.transcode_profile)
    level_applied: Optional[str] = None
    preset_applied: Optional[str] = None
    crf_applied: Optional[int] = None
    video_opts = [
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        profile,
    ]
    if encoder == "libx264":
        video_opts.extend(
            [
                "-level",
                args.transcode_level,
                "-preset",
                args.transcode_preset,
                "-crf",
                str(args.transcode_crf),
            ]
        )
        level_applied = args.transcode_level
        preset_applied = args.transcode_preset
        crf_applied = args.transcode_crf
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        *video_opts,
        "-vsync",
        "cfr",
        "-r",
        f"{fps:.6f}",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        args.transcode_audio_bitrate,
        "-ar",
        str(args.transcode_audio_rate),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if encoder == "libopenh264" and "Incorrect library version loaded" in stderr:
            raise RuntimeError(
                "libopenh264 runtime mismatch. Install a compatible libopenh264 or "
                "use an ffmpeg build with libx264. You can set FFMPEG_BIN to a different ffmpeg.\n"
                f"{stderr}"
            )
        raise RuntimeError(f"ffmpeg transcode failed: {' '.join(cmd)}\n{stderr}")
    return {
        "encoder": encoder,
        "profile_applied": profile,
        "level_applied": level_applied,
        "preset_applied": preset_applied,
        "crf_applied": crf_applied,
    }


def probe_ffprobe(path: str) -> Optional[VideoMeta]:
    ffprobe_bin = resolve_bin("FFPROBE_BIN", "ffprobe", required=False)
    if ffprobe_bin is None:
        return None
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    frame_count = int(stream.get("nb_frames") or 0)
    return VideoMeta(width=width, height=height, fps=fps, frame_count=frame_count)


def probe_cv2(path: str) -> Optional[VideoMeta]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return VideoMeta(width=width, height=height, fps=fps, frame_count=frame_count)


def get_video_meta(
    path: str, manual_width: Optional[int], manual_height: Optional[int], manual_fps: Optional[float]
) -> VideoMeta:
    meta_ff = probe_ffprobe(path)
    meta_cv = None
    width = height = 0
    fps = 0.0
    frame_count = 0

    if meta_ff:
        width = meta_ff.width
        height = meta_ff.height
        fps = meta_ff.fps
        frame_count = meta_ff.frame_count
        if width <= 0 or height <= 0 or fps <= 0:
            meta_cv = probe_cv2(path)
            if meta_cv:
                if width <= 0:
                    width = meta_cv.width
                if height <= 0:
                    height = meta_cv.height
                if fps <= 0:
                    fps = meta_cv.fps
                if frame_count <= 0:
                    frame_count = meta_cv.frame_count
    else:
        meta_cv = probe_cv2(path)
        if meta_cv:
            width = meta_cv.width
            height = meta_cv.height
            fps = meta_cv.fps
            frame_count = meta_cv.frame_count

    if width <= 0 and manual_width is not None:
        width = manual_width
    if height <= 0 and manual_height is not None:
        height = manual_height
    if fps <= 0 and manual_fps is not None:
        fps = manual_fps

    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(
            "Failed to read video metadata. Install ffprobe or cv2, or re-run with "
            "--fps --width --height."
        )
    return VideoMeta(width=width, height=height, fps=fps, frame_count=frame_count)


def load_depth(path: str) -> np.ndarray:
    with np.load(path) as data:
        if not data.files:
            raise RuntimeError(f"No arrays found in depth npz: {path}")
        depth = np.asarray(data[data.files[0]])
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 2:
        depth = depth[np.newaxis, :, :]
    if depth.ndim != 3:
        raise RuntimeError(f"Unsupported depth shape {depth.shape}; expected (T,H,W).")
    return depth.astype(np.float32)


def normalize_conf(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        return 0.0
    if conf <= 0:
        return 0.0
    if conf <= 1.0:
        return conf
    if conf <= 2.0:
        return conf / 2.0
    if conf <= 100.0:
        return conf / 100.0
    return 1.0


def is_object_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {"bbox", "box", "keypoints", "segmentation", "mask", "bbox_xyxy", "xyxy"}
    return any(key in value for key in keys)


def extract_frames(data: Any) -> List[Any]:
    if isinstance(data, list):
        if data and all(isinstance(obj, dict) for obj in data):
            has_frame_key = any(
                any(k in obj for k in ["frame_index", "frame_id", "frame"]) for obj in data
            )
            has_frame_objects = any(
                any(k in obj for k in ["objects", "detections", "instances", "annotations"]) for obj in data
            )
            if has_frame_key and not has_frame_objects:
                return group_objects_by_frame(data)
        return data
    if isinstance(data, dict):
        if isinstance(data.get("images"), list) and isinstance(data.get("annotations"), list):
            return convert_coco_format(data)
        for key in ["frames", "frame_objects", "frame_data", "frame_annotations", "annotations_per_frame"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        for key in ["objects", "annotations"]:
            if key in data and isinstance(data[key], list):
                objs = data[key]
                if any(
                    isinstance(obj, dict) and any(k in obj for k in ["frame_index", "frame_id", "frame"])
                    for obj in objs
                ):
                    return group_objects_by_frame(objs)
    raise RuntimeError("Unsupported metadata format; expected per-frame lists or frames key.")


def convert_coco_format(data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    images = [img for img in data.get("images", []) if isinstance(img, dict)]
    annotations = [ann for ann in data.get("annotations", []) if isinstance(ann, dict)]
    categories = [cat for cat in data.get("categories", []) if isinstance(cat, dict)]

    cat_name_map: Dict[int, str] = {}
    for cat in categories:
        cat_id = cat.get("id")
        name = cat.get("name")
        if cat_id is None or name is None:
            continue
        try:
            cat_name_map[int(cat_id)] = str(name)
        except Exception:
            continue

    image_id_to_frame: Dict[int, int] = {}
    frame_ids: List[int] = []
    for idx, img in enumerate(images):
        image_id = img.get("id")
        if image_id is None:
            continue
        frame_id = img.get("frame_id")
        if frame_id is None:
            frame_id = img.get("frame", img.get("frame_index", idx))
        try:
            frame_idx = int(frame_id)
        except Exception:
            frame_idx = idx
        try:
            image_id_to_frame[int(image_id)] = frame_idx
        except Exception:
            continue
        frame_ids.append(frame_idx)

    if not frame_ids:
        return []

    max_frame = max(frame_ids)
    frames: List[List[Dict[str, Any]]] = [[] for _ in range(max_frame + 1)]
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        try:
            frame_idx = image_id_to_frame[int(image_id)]
        except Exception:
            continue
        category_id = ann.get("category_id")
        if category_id is not None:
            try:
                cat_id_int = int(category_id)
            except Exception:
                cat_id_int = None
            if cat_id_int is not None and cat_id_int in cat_name_map:
                ann = dict(ann)
                ann["category_name"] = cat_name_map[cat_id_int]
        frames[frame_idx].append(ann)
    return frames


def normalize_kp_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_skeleton_edges(raw_edges: Any, kp_count: int) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    if not isinstance(raw_edges, list):
        return edges
    for pair in raw_edges:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            a = int(pair[0])
            b = int(pair[1])
        except Exception:
            continue
        edges.append((a, b))
    if not edges:
        return edges
    min_idx = min(min(a, b) for a, b in edges)
    max_idx = max(max(a, b) for a, b in edges)
    if min_idx == 1 and max_idx == kp_count:
        edges = [(a - 1, b - 1) for a, b in edges]
    normalized: List[Tuple[int, int]] = []
    for a, b in edges:
        if 0 <= a < kp_count and 0 <= b < kp_count:
            normalized.append((a, b))
    return normalized


def infer_root_indices(kp_names: Sequence[str], kp_count: int) -> List[int]:
    if kp_names:
        name_map = {normalize_kp_name(name): idx for idx, name in enumerate(kp_names)}
        pairs = [
            ("left_hip", "right_hip"),
            ("l_hip", "r_hip"),
            ("lhip", "rhip"),
            ("l_b_knee", "r_b_knee"),
            ("left_back_knee", "right_back_knee"),
            ("l_b_paw", "r_b_paw"),
            ("left_back_paw", "right_back_paw"),
        ]
        for left, right in pairs:
            if left in name_map and right in name_map:
                return [name_map[left], name_map[right]]
        for single in ["pelvis", "root", "center"]:
            if single in name_map:
                return [name_map[single]]
    if kp_count > max(COCO_LHIP, COCO_RHIP):
        return [COCO_LHIP, COCO_RHIP]
    return []


def infer_anchor_indices(cat_id: int, kp_names: Sequence[str], kp_count: int) -> List[int]:
    if kp_names:
        name_map = {normalize_kp_name(name): idx for idx, name in enumerate(kp_names)}
        if cat_id == TYPE_PERSON:
            pairs = [
                ("left_hip", "right_hip"),
                ("l_hip", "r_hip"),
                ("lhip", "rhip"),
            ]
            for left, right in pairs:
                if left in name_map and right in name_map:
                    return [name_map[left], name_map[right]]
        if cat_id == TYPE_ANIMAL:
            withers = name_map.get("withers") or name_map.get("wither")
            tailbase = (
                name_map.get("tailbase") or name_map.get("tail_base") or name_map.get("tail")
            )
            if withers is not None and tailbase is not None:
                return [withers, tailbase]
    if cat_id == TYPE_PERSON and kp_count > max(COCO_LHIP, COCO_RHIP):
        return [COCO_LHIP, COCO_RHIP]
    return []


def default_category_specs() -> Dict[int, Dict[str, Any]]:
    return {
        TYPE_OTHER: {
            "id": TYPE_OTHER,
            "name": "other",
            "kp_names": [],
            "kp_count": 0,
            "skeleton_edges": [],
            "root_indices": [],
            "anchor_indices": [],
        },
        TYPE_PERSON: {
            "id": TYPE_PERSON,
            "name": "person",
            "kp_names": [],
            "kp_count": 17,
            "skeleton_edges": [],
            "root_indices": [COCO_LHIP, COCO_RHIP],
            "anchor_indices": [COCO_LHIP, COCO_RHIP],
        },
        TYPE_ANIMAL: {
            "id": TYPE_ANIMAL,
            "name": "animal",
            "kp_names": [],
            "kp_count": 20,
            "skeleton_edges": [],
            "root_indices": [],
            "anchor_indices": [],
        },
    }


def parse_category_specs(data: Any) -> Dict[int, Dict[str, Any]]:
    specs: Dict[int, Dict[str, Any]] = {}
    if isinstance(data, dict):
        categories = data.get("categories")
        if isinstance(categories, list):
            for cat in categories:
                if not isinstance(cat, dict):
                    continue
                cat_id = cat.get("id")
                if cat_id is None:
                    continue
                try:
                    cat_id_int = int(cat_id)
                except Exception:
                    continue
                name = str(cat.get("name") or f"cat_{cat_id_int}")
                kp_names = cat.get("keypoints")
                if isinstance(kp_names, list):
                    kp_names = [str(kp) for kp in kp_names]
                else:
                    kp_names = []
                kp_count = len(kp_names)
                edges = normalize_skeleton_edges(cat.get("skeleton"), kp_count)
                root_indices = infer_root_indices(kp_names, kp_count)
                anchor_indices = infer_anchor_indices(cat_id_int, kp_names, kp_count)
                specs[cat_id_int] = {
                    "id": cat_id_int,
                    "name": name,
                    "kp_names": kp_names,
                    "kp_count": kp_count,
                    "skeleton_edges": edges,
                    "root_indices": root_indices,
                    "anchor_indices": anchor_indices,
                }
    if not specs:
        specs = default_category_specs()
    return specs


def build_category_table(cat_specs: Dict[int, Dict[str, Any]]) -> bytes:
    entries: List[bytes] = []
    for cat_id in sorted(cat_specs.keys()):
        spec = cat_specs[cat_id]
        name = str(spec.get("name") or f"cat_{cat_id}")
        name_bytes = name.encode("utf-8")
        kp_count = int(spec.get("kp_count", 0))
        edges = spec.get("skeleton_edges") or []
        entry = bytearray()
        entry.extend(struct.pack("<HHH", cat_id & 0xFFFF, kp_count & 0xFFFF, len(name_bytes) & 0xFFFF))
        entry.extend(name_bytes)
        entry.extend(struct.pack("<H", len(edges) & 0xFFFF))
        for a, b in edges:
            entry.extend(struct.pack("<HH", int(a) & 0xFFFF, int(b) & 0xFFFF))
        entries.append(bytes(entry))
    table = bytearray()
    table.extend(struct.pack("<H", len(entries)))
    for entry in entries:
        table.extend(entry)
    return bytes(table)


def group_objects_by_frame(objects: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    frames: Dict[int, List[Dict[str, Any]]] = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        frame_idx = obj.get("frame_index")
        if frame_idx is None:
            frame_idx = obj.get("frame_id", obj.get("frame"))
        if frame_idx is None:
            continue
        try:
            frame_idx_int = int(frame_idx)
        except Exception:
            continue
        frames.setdefault(frame_idx_int, []).append(obj)
    if not frames:
        return []
    max_idx = max(frames.keys())
    return [frames.get(i, []) for i in range(max_idx + 1)]


def extract_objects(frame_entry: Any) -> List[Dict[str, Any]]:
    if frame_entry is None:
        return []
    if isinstance(frame_entry, list):
        return [obj for obj in frame_entry if isinstance(obj, dict)]
    if isinstance(frame_entry, dict):
        for key in ["objects", "detections", "instances", "annotations"]:
            if key in frame_entry and isinstance(frame_entry[key], list):
                return [obj for obj in frame_entry[key] if isinstance(obj, dict)]
        if is_object_dict(frame_entry):
            return [frame_entry]
    return []


def parse_bbox(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    if "bbox" in obj:
        bbox = obj["bbox"]
        if isinstance(bbox, dict):
            if all(k in bbox for k in ["x", "y", "w", "h"]):
                return float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
            if all(k in bbox for k in ["xmin", "ymin", "xmax", "ymax"]):
                x0 = float(bbox["xmin"])
                y0 = float(bbox["ymin"])
                x1 = float(bbox["xmax"])
                y1 = float(bbox["ymax"])
                return x0, y0, x1 - x0, y1 - y0
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            if obj.get("bbox_mode", "").lower().endswith("xyxy"):
                return x0, y0, x1 - x0, y1 - y0
            return x0, y0, x1, y1
    for key in ["bbox_xyxy", "xyxy"]:
        if key in obj:
            bbox = obj[key]
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                return x0, y0, x1 - x0, y1 - y0
    if all(k in obj for k in ["x", "y", "w", "h"]):
        return float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"])
    return None


def parse_keypoints(
    obj: Dict[str, Any], expected_kp_count: int
) -> Tuple[Optional[List[Tuple[float, float, float]]], List[int], int]:
    for key in ["keypoints", "pose", "joints", "kpts"]:
        if key not in obj:
            continue
        kpts = obj[key]
        if kpts is None:
            return None, [], 0
        scores = obj.get("keypoint_scores") or obj.get("scores")
        points: List[Tuple[float, float, float]] = []
        vis_list: List[int] = []

        if isinstance(kpts, (list, tuple)) and kpts and all(isinstance(v, (int, float)) for v in kpts):
            given = obj.get("num_keypoints")
            if given is None:
                given = len(kpts) // 3
            try:
                given_count = max(0, int(given))
            except Exception:
                given_count = len(kpts) // 3
            max_points = len(kpts) // 3
            read_n = min(given_count, max_points)
            for idx in range(read_n):
                u = float(kpts[idx * 3])
                v = float(kpts[idx * 3 + 1])
                vis_raw = float(kpts[idx * 3 + 2])
                vis_int = int(round(vis_raw))
                vis_u8 = max(0, min(255, vis_int))
                if isinstance(scores, (list, tuple)) and idx < len(scores) and vis_int > 0:
                    conf = float(scores[idx])
                else:
                    conf = float(vis_int) / 2.0 if vis_int > 0 else 0.0
                points.append((u, v, conf))
                vis_list.append(vis_u8)
        elif isinstance(kpts, (list, tuple)) and kpts and all(isinstance(v, (list, tuple)) for v in kpts):
            given_count = len(kpts)
            for idx, kp in enumerate(kpts):
                if len(kp) < 2:
                    continue
                u = float(kp[0])
                v = float(kp[1])
                vis_raw = float(kp[2]) if len(kp) >= 3 else 0.0
                vis_int = int(round(vis_raw))
                vis_u8 = max(0, min(255, vis_int))
                if isinstance(scores, (list, tuple)) and idx < len(scores) and vis_int > 0:
                    conf = float(scores[idx])
                else:
                    conf = float(vis_int) / 2.0 if vis_int > 0 else 0.0
                points.append((u, v, conf))
                vis_list.append(vis_u8)
        else:
            return None, [], 0

        expected = max(0, int(expected_kp_count))
        if expected > len(points):
            for _ in range(expected - len(points)):
                points.append((0.0, 0.0, 0.0))
                vis_list.append(0)
        elif expected and expected < len(points):
            points = points[:expected]
            vis_list = vis_list[:expected]
        return points, vis_list, given_count
    return None, [], 0


def compute_anchor_from_keypoints(
    keypoints: Optional[List[Tuple[float, float, float]]],
    kp_vis: Sequence[int],
    anchor_indices: Sequence[int],
) -> Optional[Tuple[float, float]]:
    if keypoints is None or len(anchor_indices) < 2:
        return None
    coords: List[Tuple[float, float]] = []
    for idx in anchor_indices[:2]:
        if idx < 0 or idx >= len(keypoints):
            return None
        u, v, conf = keypoints[idx]
        if not math.isfinite(u) or not math.isfinite(v):
            return None
        if idx < len(kp_vis) and kp_vis[idx] <= 0:
            return None
        if normalize_conf(conf) <= 0:
            return None
        coords.append((u, v))
    if len(coords) < 2:
        return None
    return (coords[0][0] + coords[1][0]) * 0.5, (coords[0][1] + coords[1][1]) * 0.5


def normalize_type(obj: Dict[str, Any], keypoints: Optional[List[Tuple[float, float, float]]]) -> int:
    raw_value = None
    for key in ["type", "category", "label", "class", "category_name", "name"]:
        if key in obj:
            raw_value = obj[key]
            break
    if raw_value is None and "category_id" in obj:
        try:
            cat_id = int(obj["category_id"])
        except Exception:
            cat_id = None
        if cat_id == 1:
            return TYPE_PERSON
        if cat_id == 2:
            return TYPE_ANIMAL
        if cat_id is not None:
            return TYPE_OTHER
    if isinstance(raw_value, (int, float)):
        value_int = int(raw_value)
        if value_int in (TYPE_OTHER, TYPE_PERSON, TYPE_ANIMAL):
            return value_int
    if isinstance(raw_value, str):
        name = raw_value.lower()
        if "human" in name or "person" in name:
            return TYPE_PERSON
        animal_terms = [
            "animal",
            "dog",
            "cat",
            "horse",
            "cow",
            "sheep",
            "bird",
            "bear",
            "zebra",
            "giraffe",
            "elephant",
            "lion",
            "tiger",
            "monkey",
        ]
        if any(term in name for term in animal_terms):
            return TYPE_ANIMAL
        if "rigid" in name or "other" in name:
            return TYPE_OTHER
    if keypoints is not None:
        return TYPE_PERSON
    return TYPE_OTHER


def clamp_bbox(x: float, y: float, w: float, h: float, w_eye: int, height: int) -> Tuple[float, float, float, float]:
    x0 = max(0.0, x)
    y0 = max(0.0, y)
    x1 = min(float(w_eye), x + w)
    y1 = min(float(height), y + h)
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    return x0, y0, w, h


def clamp_point(u: float, v: float, w_eye: int, height: int) -> Tuple[float, float]:
    if not math.isfinite(u) or not math.isfinite(v):
        return 0.0, 0.0
    u = max(0.0, min(u, float(w_eye - 1)))
    v = max(0.0, min(v, float(height - 1)))
    return u, v


def adjust_left_eye(value: float, w_eye: int, width: int, left_eye_origin: str) -> float:
    if left_eye_origin == "full" and value >= w_eye and value < width:
        return value - w_eye
    return value


def compute_crop_params(meta_w: int, meta_h: int, align128: int, crop_mode: str) -> Tuple[int, int, int, int]:
    crop_x0 = 0
    crop_y0 = 0
    crop_w = meta_w
    crop_h = meta_h
    if align128:
        crop_w = (meta_w // 128) * 128
        crop_h = (meta_h // 128) * 128
    if crop_mode != "topleft":
        print(f"Warning: unsupported crop_mode {crop_mode}; using topleft.")
    if crop_w <= 0 or crop_h <= 0:
        print(f"Warning: computed crop {crop_w}x{crop_h} invalid; using meta size {meta_w}x{meta_h}.")
        crop_w = meta_w
        crop_h = meta_h
    return crop_x0, crop_y0, crop_w, crop_h


def intersect_bbox_with_crop(
    bbox: Tuple[float, float, float, float], crop_x0: int, crop_y0: int, crop_w: int, crop_h: int
) -> Optional[Tuple[float, float, float, float]]:
    x, y, w, h = bbox
    x0 = x
    y0 = y
    x1 = x + w
    y1 = y + h
    ix0 = max(x0, float(crop_x0))
    iy0 = max(y0, float(crop_y0))
    ix1 = min(x1, float(crop_x0 + crop_w))
    iy1 = min(y1, float(crop_y0 + crop_h))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


def meta_to_eye_point(
    u_meta: float,
    v_meta: float,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    eye_w: int,
    eye_h: int,
) -> Optional[Tuple[float, float]]:
    if crop_w <= 0 or crop_h <= 0 or eye_w <= 0 or eye_h <= 0:
        return None
    u_crop = u_meta - crop_x0
    v_crop = v_meta - crop_y0
    if u_crop < 0 or v_crop < 0 or u_crop >= crop_w or v_crop >= crop_h:
        return None
    scale_x = float(eye_w) / float(crop_w)
    scale_y = float(eye_h) / float(crop_h)
    return u_crop * scale_x, v_crop * scale_y


def meta_to_eye_xyxy(
    x0_meta: float,
    y0_meta: float,
    x1_meta: float,
    y1_meta: float,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    eye_w: int,
    eye_h: int,
) -> Tuple[float, float, float, float]:
    if crop_w <= 0 or crop_h <= 0 or eye_w <= 0 or eye_h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    scale_x = float(eye_w) / float(crop_w)
    scale_y = float(eye_h) / float(crop_h)
    x0_eye = (x0_meta - crop_x0) * scale_x
    y0_eye = (y0_meta - crop_y0) * scale_y
    x1_eye = (x1_meta - crop_x0) * scale_x
    y1_eye = (y1_meta - crop_y0) * scale_y
    return x0_eye, y0_eye, x1_eye, y1_eye


def eye_to_meta_point(
    u_eye: float,
    v_eye: float,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    eye_w: int,
    eye_h: int,
) -> Optional[Tuple[float, float]]:
    if crop_w <= 0 or crop_h <= 0 or eye_w <= 0 or eye_h <= 0:
        return None
    scale_x = float(crop_w) / float(eye_w)
    scale_y = float(crop_h) / float(eye_h)
    u_meta = u_eye * scale_x + crop_x0
    v_meta = v_eye * scale_y + crop_y0
    return u_meta, v_meta


def normalize_track_id(
    raw_value: Any, track_id_map: Dict[Any, int], next_track_id: int
) -> Tuple[Optional[int], int]:
    if raw_value is None:
        return None, next_track_id
    if isinstance(raw_value, (np.integer, int)):
        track_id = int(raw_value)
        track_id = max(0, min(0xFFFFFFFF, track_id))
        return track_id, max(next_track_id, track_id + 1)
    try:
        track_id = int(raw_value)
        track_id = max(0, min(0xFFFFFFFF, track_id))
        return track_id, max(next_track_id, track_id + 1)
    except Exception:
        if raw_value not in track_id_map:
            track_id_map[raw_value] = next_track_id
            next_track_id += 1
        track_id = max(0, min(0xFFFFFFFF, track_id_map[raw_value]))
        track_id_map[raw_value] = track_id
        return track_id, next_track_id


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1 = ax + aw
    ay1 = ay + ah
    bx1 = bx + bw
    by1 = by + bh
    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def assign_track_ids(
    objects: List[Dict[str, Any]],
    prev_objects: List[Dict[str, Any]],
    next_track_id: int,
    iou_th: float = 0.3,
) -> Tuple[List[Dict[str, Any]], int]:
    unmatched = [i for i, obj in enumerate(objects) if obj["track_id"] is None]
    if not unmatched:
        return objects, next_track_id
    candidates = []
    for i in unmatched:
        for j, prev in enumerate(prev_objects):
            if prev["track_id"] is None:
                continue
            if objects[i]["category_id"] != prev["category_id"]:
                continue
            iou = bbox_iou(objects[i]["bbox"], prev["bbox"])
            if iou >= iou_th:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True, key=lambda item: item[0])
    used_i = set()
    used_j = set()
    for _, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        objects[i]["track_id"] = prev_objects[j]["track_id"]
        used_i.add(i)
        used_j.add(j)
    for i in unmatched:
        if objects[i]["track_id"] is None:
            objects[i]["track_id"] = next_track_id
            next_track_id += 1
    return objects, next_track_id


def decode_mask(segmentation: Any, height: int, width: int) -> Optional[np.ndarray]:
    if mask_utils is None or segmentation is None:
        return None
    try:
        if isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation:
            rle = segmentation
        elif isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        else:
            return None
        mask = mask_utils.decode(rle)
    except Exception:
        return None
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def compute_mask_centroid(
    segmentation: Any,
    meta_h: int,
    meta_w: int,
    crop_x0: int = 0,
    crop_y0: int = 0,
    crop_w: Optional[int] = None,
    crop_h: Optional[int] = None,
) -> Optional[Tuple[float, float]]:
    mask = decode_mask(segmentation, meta_h, meta_w)
    if mask is None:
        return None
    if crop_w is not None and crop_h is not None:
        x1 = min(meta_w, crop_x0 + crop_w)
        y1 = min(meta_h, crop_y0 + crop_h)
        if crop_x0 < 0 or crop_y0 < 0 or x1 <= crop_x0 or y1 <= crop_y0:
            return None
        mask = mask[crop_y0:y1, crop_x0:x1]
        if mask.size == 0:
            return None
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        u = float(xs.mean()) + float(crop_x0)
        v = float(ys.mean()) + float(crop_y0)
        return u, v
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    u = float(xs.mean())
    v = float(ys.mean())
    return u, v


def sample_depth_median(
    depth_frame: np.ndarray,
    u_eye: float,
    v_eye: float,
    sample_k: int,
    meta_w: int,
    meta_h: int,
    eye_w: int,
    eye_h: int,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
) -> Optional[float]:
    if depth_frame is None:
        return None
    if not math.isfinite(u_eye) or not math.isfinite(v_eye):
        return None
    if meta_w <= 0 or meta_h <= 0 or eye_w <= 0 or eye_h <= 0 or crop_w <= 0 or crop_h <= 0:
        return None
    meta_uv = eye_to_meta_point(u_eye, v_eye, crop_x0, crop_y0, crop_w, crop_h, eye_w, eye_h)
    if meta_uv is None:
        return None
    u_meta, v_meta = meta_uv
    h, w = depth_frame.shape
    x = int(round(u_meta * w / float(meta_w)))
    y = int(round(v_meta * h / float(meta_h)))
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    k = max(1, int(sample_k))
    half = k // 2
    x0 = max(0, x - half)
    x1 = min(w, x + half + 1)
    y0 = max(0, y - half)
    y1 = min(h, y + half + 1)
    window = depth_frame[y0:y1, x0:x1]
    if window.size == 0:
        return None
    flat = window.reshape(-1)
    mask = np.isfinite(flat) & (flat > 0)
    if not np.any(mask):
        return None
    return float(np.median(flat[mask]))


def smooth_value(prev: Optional[float], value: Optional[float], alpha: float) -> float:
    if value is None:
        if prev is None:
            return 0.0
        return prev
    if prev is None:
        return value
    return alpha * prev + (1.0 - alpha) * value


def quantize_int16(value: float, scale: float) -> int:
    if scale <= 0:
        raise ValueError("Quantization scale must be positive.")
    q = int(round(value / scale))
    return max(-32768, min(32767, q))


def quantize_array_int16(values: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Quantization scale must be positive.")
    q = np.round(values / scale).astype(np.int64)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q


def compute_root(joints3d: np.ndarray, valid: np.ndarray, root_indices: Sequence[int]) -> np.ndarray:
    roots = []
    for idx in root_indices:
        if 0 <= idx < len(valid) and valid[idx]:
            roots.append(joints3d[idx])
    if roots:
        return np.mean(np.stack(roots, axis=0), axis=0)
    if np.any(valid):
        return np.mean(joints3d[valid], axis=0)
    return np.zeros(3, dtype=np.float32)


def compute_joints_rel(
    keypoints: List[Tuple[float, float, float]],
    depth_frame: np.ndarray,
    w_eye: int,
    height: int,
    fovx_deg: float,
    sample_k: int,
    conf_th: float,
    ema_alpha: float,
    prev_joints_rel: Optional[np.ndarray],
    meta_w: int,
    meta_h: int,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    root_indices: Sequence[int],
) -> np.ndarray:
    kp_count = len(keypoints)
    joints3d = np.zeros((kp_count, 3), dtype=np.float32)
    valid = np.zeros(kp_count, dtype=bool)
    if prev_joints_rel is not None and prev_joints_rel.shape[0] != kp_count:
        prev_joints_rel = None
    fovx_rad = math.radians(fovx_deg)
    fx = 1.0 / math.tan(fovx_rad / 2.0)
    fy = fx * (float(w_eye) / float(height))

    for idx, (u, v, conf) in enumerate(keypoints):
        conf_n = normalize_conf(conf)
        if conf_n < conf_th:
            continue
        z = sample_depth_median(
            depth_frame,
            u,
            v,
            sample_k,
            meta_w,
            meta_h,
            w_eye,
            height,
            crop_x0,
            crop_y0,
            crop_w,
            crop_h,
        )
        if z is None:
            continue
        x_ndc = (u / float(w_eye) - 0.5) * 2.0
        y_ndc = (0.5 - v / float(height)) * 2.0
        X = x_ndc * z / fx
        Y = y_ndc * z / fy
        joints3d[idx] = (X, Y, z)
        valid[idx] = True

    root = compute_root(joints3d, valid, root_indices)
    joints_rel_raw = joints3d - root

    if prev_joints_rel is not None:
        for idx in range(kp_count):
            if not valid[idx]:
                joints_rel_raw[idx] = prev_joints_rel[idx]
        joints_rel = ema_alpha * prev_joints_rel + (1.0 - ema_alpha) * joints_rel_raw
    else:
        joints_rel = joints_rel_raw

    return joints_rel


def choose_compression(name: str) -> Tuple[int, Any, str]:
    if name == "none":
        return 0, lambda data: data, "none"
    if name == "lz4":
        if lz4f is not None:
            return 2, lambda data: lz4f.compress(data), "lz4"
        print("lz4 not available, falling back to zlib.")
        return 1, lambda data: zlib.compress(data), "zlib"
    return 1, lambda data: zlib.compress(data), "zlib"


def _fmt_float(value: Optional[float], precision: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "None"
    return f"{value:.{precision}f}"


def log_header_and_categories(
    *,
    version: int,
    compress_id: int,
    compress_name: str,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    w_eye: int,
    fovx_deg: float,
    quant_pos_scale: float,
    quant_joint_scale: float,
    category_table_offset: int,
    category_table_size: int,
    index_table_offset: int,
    cat_specs: Dict[int, Dict[str, Any]],
    label: str = "",
    include_categories: bool = True,
) -> None:
    suffix = f" ({label})" if label else ""
    print(
        "[meta] header"
        f"{suffix}: version={version} compress={compress_name} compress_id={compress_id} "
        f"width={width} height={height} fps={fps:.3f} num_frames={num_frames} w_eye={w_eye} "
        f"fovx={fovx_deg:.3f} quant_pos_scale={quant_pos_scale} quant_joint_scale={quant_joint_scale} "
        f"category_table_offset={category_table_offset} category_table_size={category_table_size} "
        f"index_table_offset={index_table_offset}"
    )
    if not include_categories:
        return
    print(f"[meta] categories: count={len(cat_specs)}")
    for cat_id in sorted(cat_specs.keys()):
        spec = cat_specs[cat_id]
        name = str(spec.get("name") or f"cat_{cat_id}")
        kp_count = int(spec.get("kp_count", 0))
        edges = spec.get("skeleton_edges") or []
        roots = spec.get("root_indices") or []
        print(
            f"[meta] category id={cat_id} name={name} kp_count={kp_count} "
            f"edges={len(edges)} roots={roots}"
        )


def log_frame_summary(
    frame_idx: int,
    offset: int,
    raw_payload_len: int,
    compressed_len: int,
    objects_count: int,
    debug_objects: Optional[List[Dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    print(
        f"[frame {frame_idx:04d}] offset={offset} raw={raw_payload_len}B "
        f"comp={compressed_len}B objects={objects_count}"
    )
    if not debug_objects:
        return
    max_objects = max(0, int(args.debug_meta_max_objects))
    for obj_idx, obj in enumerate(debug_objects[:max_objects]):
        bbox_x, bbox_y, bbox_w, bbox_h = obj.get("bbox", (0.0, 0.0, 0.0, 0.0))
        anchor_u, anchor_v = obj.get("anchor_uv", (None, None))
        anchor_str = (
            f"({anchor_u:.1f},{anchor_v:.1f})"
            if anchor_u is not None and anchor_v is not None
            else "None"
        )
        z_raw = _fmt_float(obj.get("anchor_z_raw"))
        z_ema = _fmt_float(obj.get("anchor_z"))
        z_q = obj.get("anchor_z_q")
        cat_name = obj.get("category_name") or ""
        base = (
            f"[obj {obj_idx}] tid={obj.get('track_id')} "
            f"cat={obj.get('category_id')}({cat_name}) "
            f"bbox=({bbox_x:.1f},{bbox_y:.1f},{bbox_w:.1f},{bbox_h:.1f}) "
            f"anchor={anchor_str} z_raw={z_raw} z_ema={z_ema} z_q={z_q} "
            f"skeleton={1 if obj.get('has_skeleton') else 0}"
        )
        if obj.get("has_skeleton"):
            kp_count = int(obj.get("kp_count", 0))
            vis_count = int(obj.get("vis_count", 0))
            joints_q_min = obj.get("joints_q_min")
            joints_q_max = obj.get("joints_q_max")
            extra = f" kp={kp_count} vis={vis_count}/{kp_count}"
            if joints_q_min is not None and joints_q_max is not None:
                extra += f" joints_q[min,max]={joints_q_min},{joints_q_max}"
            base += extra
        print(base)
    if objects_count > max_objects:
        print(f"[frame {frame_idx:04d}] ... {objects_count - max_objects} more objects")


def _parse_category_table_bytes(data: bytes) -> Dict[int, Dict[str, Any]]:
    pos = 0
    if len(data) < 2:
        return {}
    (count,) = struct.unpack_from("<H", data, pos)
    pos += 2
    categories: Dict[int, Dict[str, Any]] = {}
    for _ in range(count):
        if pos + 6 > len(data):
            break
        cat_id, kp_count, name_len = struct.unpack_from("<HHH", data, pos)
        pos += 6
        if pos + name_len > len(data):
            break
        name = data[pos : pos + name_len].decode("utf-8", errors="replace")
        pos += name_len
        if pos + 2 > len(data):
            break
        (edges_len,) = struct.unpack_from("<H", data, pos)
        pos += 2
        edges = []
        for _ in range(edges_len):
            if pos + 4 > len(data):
                break
            a, b = struct.unpack_from("<HH", data, pos)
            pos += 4
            edges.append((a, b))
        categories[int(cat_id)] = {
            "name": name,
            "kp_count": int(kp_count),
            "edges_len": int(edges_len),
            "edges": edges,
        }
    return categories


def _decompress_payload(compress_id: int, data: bytes) -> Optional[bytes]:
    if compress_id == 0:
        return data
    if compress_id == 1:
        return zlib.decompress(data)
    if compress_id == 2:
        if lz4f is None:
            print("[verify] lz4 not available; skipping payload decode.")
            return None
        return lz4f.decompress(data)
    print(f"[verify] unknown compress_id={compress_id}; skipping payload decode.")
    return None


def verify_meta_bin(path: str, debug_first_frames: int) -> None:
    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        prefix = handle.read(6)
        if len(prefix) < 6:
            print("[verify] failed to read header prefix.")
            return
        magic, version = struct.unpack("<4sH", prefix)
        if magic != MAGIC:
            print("[verify] invalid magic.")
            return
        if version >= 2:
            handle.seek(0)
            header_bytes = handle.read(HEADER_V2_STRUCT.size)
            if len(header_bytes) < HEADER_V2_STRUCT.size:
                print("[verify] failed to read v2 header.")
                return
            (
                _magic,
                version,
                compress_id,
                width,
                height,
                fps,
                num_frames,
                w_eye,
                _reserved,
                fovx_deg,
                quant_pos_scale,
                quant_joint_scale,
                category_table_offset,
                category_table_size,
                index_table_offset,
            ) = HEADER_V2_STRUCT.unpack(header_bytes)
        else:
            handle.seek(0)
            header_bytes = handle.read(HEADER_V1_STRUCT.size)
            if len(header_bytes) < HEADER_V1_STRUCT.size:
                print("[verify] failed to read v1 header.")
                return
            (
                _magic,
                version,
                compress_id,
                width,
                height,
                fps,
                num_frames,
                w_eye,
                _reserved,
                fovx_deg,
                quant_pos_scale,
                quant_joint_scale,
                index_table_offset,
            ) = HEADER_V1_STRUCT.unpack(header_bytes)
            category_table_offset = 0
            category_table_size = 0
        compress_name = {0: "none", 1: "zlib", 2: "lz4"}.get(int(compress_id), "unknown")
        print(
            "[verify] header: "
            f"version={version} compress={compress_name} compress_id={compress_id} "
            f"width={width} height={height} fps={fps:.3f} num_frames={num_frames} w_eye={w_eye} "
            f"fovx={fovx_deg:.3f} quant_pos_scale={quant_pos_scale} "
            f"quant_joint_scale={quant_joint_scale} category_table_offset={category_table_offset} "
            f"category_table_size={category_table_size} index_table_offset={index_table_offset}"
        )

        categories: Dict[int, Dict[str, Any]] = {}
        if category_table_size:
            handle.seek(category_table_offset)
            data = handle.read(category_table_size)
            categories = _parse_category_table_bytes(data)
            print(f"[verify] categories: count={len(categories)}")
            for cat_id in sorted(categories.keys()):
                spec = categories[cat_id]
                print(
                    f"[verify] category id={cat_id} name={spec.get('name')} "
                    f"kp_count={spec.get('kp_count')} edges={spec.get('edges_len')}"
                )
        else:
            print("[verify] categories: none")

        handle.seek(index_table_offset)
        offsets_data = handle.read(int(num_frames) * 8)
        offsets: List[int] = []
        for idx in range(min(int(num_frames), len(offsets_data) // 8)):
            (off,) = struct.unpack_from("<Q", offsets_data, idx * 8)
            offsets.append(int(off))
        offsets_ok = True
        for idx, off in enumerate(offsets):
            if off <= 0 or off + 4 > file_size:
                print(f"[verify] invalid offset for frame {idx}: {off}")
                offsets_ok = False
                continue
            handle.seek(off)
            comp_len_bytes = handle.read(4)
            if len(comp_len_bytes) < 4:
                print(f"[verify] failed to read compressed length for frame {idx}")
                offsets_ok = False
                continue
            (comp_len,) = struct.unpack("<I", comp_len_bytes)
            if off + 4 + comp_len > file_size:
                print(f"[verify] frame {idx} compressed data out of range")
                offsets_ok = False
        if offsets_ok:
            print("[verify] index offsets ok")

        decode_frames = min(int(debug_first_frames), len(offsets))
        if decode_frames <= 0:
            return
        kp_counts = {cat_id: int(spec.get("kp_count", 0)) for cat_id, spec in categories.items()}
        for frame_idx in range(decode_frames):
            off = offsets[frame_idx]
            if off <= 0 or off + 4 > file_size:
                continue
            handle.seek(off)
            comp_len_bytes = handle.read(4)
            if len(comp_len_bytes) < 4:
                continue
            (comp_len,) = struct.unpack("<I", comp_len_bytes)
            comp_data = handle.read(comp_len)
            if len(comp_data) < comp_len:
                print(f"[verify] frame {frame_idx} compressed read short")
                continue
            try:
                payload = _decompress_payload(int(compress_id), comp_data)
            except Exception as exc:
                print(f"[verify] frame {frame_idx} decompress failed: {exc}")
                continue
            if payload is None:
                continue
            if len(payload) < 2:
                print(f"[verify] frame {frame_idx} payload too small")
                continue
            pos = 0
            (objects_count,) = struct.unpack_from("<H", payload, pos)
            pos += 2
            first_track_id: Optional[int] = None
            for obj_idx in range(objects_count):
                if pos + 30 > len(payload):
                    break
                (track_id,) = struct.unpack_from("<I", payload, pos)
                pos += 4
                category_id = payload[pos]
                pos += 1
                flags = payload[pos]
                pos += 1
                pos += 8  # bbox
                pos += 4  # anchor uv
                pos += 2  # anchor z
                pos += 2  # anchor scale
                pos += 8  # rot
                if obj_idx == 0:
                    first_track_id = int(track_id)
                if flags & 1:
                    kp_count = kp_counts.get(int(category_id), 0)
                    pos += kp_count * 3 * 2
                    pos += kp_count
            print(
                f"[verify] frame {frame_idx} decoded objects={objects_count} "
                f"first_track_id={first_track_id}"
            )


HEADER_V1_STRUCT = struct.Struct("<4sHHHHfIHHfffQ")
HEADER_V2_STRUCT = struct.Struct("<4sHHHHfIHHfffQIQ")


def pack_header(
    compress_id: int,
    width: int,
    height: int,
    fps: float,
    num_frames: int,
    w_eye: int,
    fovx_deg: float,
    quant_pos_scale: float,
    quant_joint_scale: float,
    category_table_offset: int,
    category_table_size: int,
    index_table_offset: int,
) -> bytes:
    if VERSION >= 2:
        return HEADER_V2_STRUCT.pack(
            MAGIC,
            VERSION,
            compress_id,
            width,
            height,
            fps,
            num_frames,
            w_eye,
            0,
            fovx_deg,
            quant_pos_scale,
            quant_joint_scale,
            category_table_offset,
            category_table_size,
            index_table_offset,
        )
    return HEADER_V1_STRUCT.pack(
        MAGIC,
        VERSION,
        compress_id,
        width,
        height,
        fps,
        num_frames,
        w_eye,
        0,
        fovx_deg,
        quant_pos_scale,
        quant_joint_scale,
        index_table_offset,
    )


def build_bundle(args: argparse.Namespace, video_mp4_path: str, transcode_info: Dict[str, Any]) -> None:
    video_meta = get_video_meta(video_mp4_path, args.width, args.height, args.fps)
    width = video_meta.width
    height = video_meta.height
    fps = video_meta.fps
    w_eye = width // 2

    depth = load_depth(args.depth_npz)
    full_meta_h, full_meta_w = depth.shape[1], depth.shape[2]
    with open(args.metadata_json, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cat_specs = parse_category_specs(metadata)
    defaults = default_category_specs()
    for cat_id, spec in defaults.items():
        cat_specs.setdefault(cat_id, spec)
    frames = extract_frames(metadata)
    cat_name_map = {
        cat_id: str(spec.get("name") or f"cat_{cat_id}") for cat_id, spec in cat_specs.items()
    }
    debug_enabled = bool(args.debug_meta)
    debug_every = max(1, int(args.debug_meta_every))
    debug_first_frames = max(0, int(args.debug_meta_first_frames))

    t_depth = depth.shape[0]
    t_meta = len(frames)
    t_video = video_meta.frame_count if video_meta.frame_count > 0 else None
    candidates = [t_depth, t_meta]
    if t_video is not None:
        candidates.append(t_video)
    num_frames = min(candidates) if candidates else 0

    if num_frames <= 0:
        raise RuntimeError("No frames to process after aligning depth/metadata/video.")

    if t_depth != num_frames or t_meta != num_frames or (t_video is not None and t_video != num_frames):
        print(
            f"Frame alignment: depth={t_depth}, meta={t_meta}, video={t_video or 'unknown'} -> {num_frames}"
        )

    crop_x0, crop_y0, crop_w, crop_h = compute_crop_params(
        full_meta_w, full_meta_h, args.align128, args.crop_mode
    )
    if (
        crop_x0 != 0
        or crop_y0 != 0
        or crop_w != full_meta_w
        or crop_h != full_meta_h
    ):
        depth = depth[:, crop_y0 : crop_y0 + crop_h, crop_x0 : crop_x0 + crop_w]
    meta_h, meta_w = depth.shape[1], depth.shape[2]
    if crop_w != w_eye or crop_h != height:
        print(
            f"Warning: crop {crop_w}x{crop_h} != eye {w_eye}x{height}; scaling coordinates."
        )

    compress_id, compress_fn, compress_name = choose_compression(args.frame_compress)

    track_state: Dict[int, TrackState] = {}
    track_id_map: Dict[Any, int] = {}
    next_track_id = 1
    prev_objects: List[Dict[str, Any]] = []

    out_dir = os.path.dirname(os.path.abspath(args.out_bundle)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".meta.bin", dir=out_dir) as tmp:
        meta_path = tmp.name

    category_table = build_category_table(cat_specs)
    category_table_offset = HEADER_V2_STRUCT.size if VERSION >= 2 else 0
    category_table_size = len(category_table) if VERSION >= 2 else 0
    if debug_enabled:
        log_header_and_categories(
            version=VERSION,
            compress_id=compress_id,
            compress_name=compress_name,
            width=width,
            height=height,
            fps=fps,
            num_frames=num_frames,
            w_eye=w_eye,
            fovx_deg=args.fovx_deg,
            quant_pos_scale=args.quant_pos_scale,
            quant_joint_scale=args.quant_joint_scale,
            category_table_offset=category_table_offset,
            category_table_size=category_table_size,
            index_table_offset=0,
            cat_specs=cat_specs,
            label="initial",
            include_categories=True,
        )

    try:
        with open(meta_path, "wb") as meta_f:
            meta_f.write(
                pack_header(
                    compress_id=compress_id,
                    width=width,
                    height=height,
                    fps=fps,
                    num_frames=num_frames,
                    w_eye=w_eye,
                    fovx_deg=args.fovx_deg,
                    quant_pos_scale=args.quant_pos_scale,
                    quant_joint_scale=args.quant_joint_scale,
                    category_table_offset=category_table_offset,
                    category_table_size=category_table_size,
                    index_table_offset=0,
                )
            )
            if category_table_size:
                meta_f.write(category_table)
            offsets: List[int] = []
            for frame_idx in range(num_frames):
                debug_frame = debug_enabled and (
                    frame_idx < debug_first_frames
                    or frame_idx % debug_every == 0
                    or frame_idx + 1 == num_frames
                )
                debug_detail = debug_enabled and frame_idx < debug_first_frames
                debug_objects: Optional[List[Dict[str, Any]]] = [] if debug_detail else None
                frame_entry = frames[frame_idx] if frame_idx < len(frames) else None
                raw_objects = extract_objects(frame_entry)
                objects: List[Dict[str, Any]] = []
                for raw_obj in raw_objects:
                    bbox = parse_bbox(raw_obj)
                    if bbox is None:
                        continue
                    x, y, w, h = bbox
                    x = adjust_left_eye(x, w_eye, width, args.left_eye_origin)
                    bbox_xyxy = intersect_bbox_with_crop((x, y, w, h), crop_x0, crop_y0, crop_w, crop_h)
                    if bbox_xyxy is None:
                        continue
                    ix0, iy0, ix1, iy1 = bbox_xyxy
                    x0, y0, x1, y1 = meta_to_eye_xyxy(
                        ix0, iy0, ix1, iy1, crop_x0, crop_y0, crop_w, crop_h, w_eye, height
                    )
                    x = x0
                    y = y0
                    w = x1 - x0
                    h = y1 - y0
                    x, y, w, h = clamp_bbox(x, y, w, h, w_eye, height)
                    if w <= 0 or h <= 0:
                        continue
                    raw_category = raw_obj.get("category_id")
                    if raw_category is not None:
                        try:
                            category_id = int(raw_category)
                        except Exception:
                            category_id = TYPE_OTHER
                    else:
                        if "keypoints" in raw_obj:
                            category_id = TYPE_PERSON
                        else:
                            category_id = normalize_type(raw_obj, None)
                    if category_id not in cat_specs:
                        category_id = TYPE_OTHER
                    if category_id > 255:
                        raise RuntimeError(f"category_id {category_id} exceeds uint8 range")
                    cat_spec = cat_specs.get(category_id, cat_specs.get(TYPE_OTHER, {}))
                    expected_kp = int(cat_spec.get("kp_count", 0))
                    keypoints, kp_vis, kp_given = parse_keypoints(raw_obj, expected_kp)
                    if expected_kp > 0 and kp_given != expected_kp:
                        ann_id = raw_obj.get("id", "unknown")
                        print(
                            f"Warning: frame {frame_idx} ann {ann_id} category {category_id} "
                            f"kp_count mismatch (given {kp_given} expected {expected_kp})"
                        )
                    if keypoints is not None:
                        adjusted = []
                        for idx, (u, v, conf) in enumerate(keypoints):
                            if not math.isfinite(u) or not math.isfinite(v):
                                adjusted.append((0.0, 0.0, 0.0))
                                if idx < len(kp_vis):
                                    kp_vis[idx] = 0
                                continue
                            u = adjust_left_eye(u, w_eye, width, args.left_eye_origin)
                            uv_eye = meta_to_eye_point(
                                u, v, crop_x0, crop_y0, crop_w, crop_h, w_eye, height
                            )
                            if uv_eye is None:
                                adjusted.append((0.0, 0.0, 0.0))
                                if idx < len(kp_vis):
                                    kp_vis[idx] = 0
                                continue
                            u_eye, v_eye = uv_eye
                            u_eye, v_eye = clamp_point(u_eye, v_eye, w_eye, height)
                            adjusted.append((u_eye, v_eye, conf))
                        keypoints = adjusted
                    raw_track_id = None
                    for key in ["track_id", "trackId", "track", "instance_id", "instanceId", "object_id", "id"]:
                        if key in raw_obj:
                            raw_track_id = raw_obj[key]
                            break
                    track_id, next_track_id = normalize_track_id(raw_track_id, track_id_map, next_track_id)
                    segmentation = raw_obj.get("segmentation") or raw_obj.get("mask")
                    has_skeleton = expected_kp > 0 and kp_given > 0 and keypoints is not None
                    if not has_skeleton:
                        keypoints = None
                        kp_vis = []
                    objects.append(
                        {
                            "track_id": track_id,
                            "category_id": category_id,
                            "bbox": (x, y, w, h),
                            "keypoints": keypoints,
                            "kp_vis": kp_vis,
                            "kp_expected": expected_kp,
                            "root_indices": cat_spec.get("root_indices", []),
                            "anchor_indices": cat_spec.get("anchor_indices", []),
                            "has_skeleton": has_skeleton,
                            "segmentation": segmentation,
                        }
                    )

                objects, next_track_id = assign_track_ids(objects, prev_objects, next_track_id)
                prev_objects = objects

                payload = bytearray()
                payload.extend(struct.pack("<H", len(objects)))
                depth_frame = depth[frame_idx]

                for obj in objects:
                    track_id = int(obj["track_id"])
                    category_id = int(obj["category_id"])
                    bbox_x, bbox_y, bbox_w, bbox_h = obj["bbox"]

                    anchor_uv = None
                    if category_id in (TYPE_PERSON, TYPE_ANIMAL):
                        anchor_uv = compute_anchor_from_keypoints(
                            obj.get("keypoints"),
                            obj.get("kp_vis") or [],
                            obj.get("anchor_indices") or [],
                        )
                    if anchor_uv is None:
                        anchor_uv_meta = compute_mask_centroid(
                            obj["segmentation"],
                            full_meta_h,
                            full_meta_w,
                            crop_x0=crop_x0,
                            crop_y0=crop_y0,
                            crop_w=crop_w,
                            crop_h=crop_h,
                        )
                        if anchor_uv_meta is not None:
                            anchor_u_meta, anchor_v_meta = anchor_uv_meta
                            anchor_u_meta = adjust_left_eye(
                                anchor_u_meta, w_eye, width, args.left_eye_origin
                            )
                            anchor_uv = meta_to_eye_point(
                                anchor_u_meta,
                                anchor_v_meta,
                                crop_x0,
                                crop_y0,
                                crop_w,
                                crop_h,
                                w_eye,
                                height,
                            )
                    if anchor_uv is None:
                        anchor_u = bbox_x + bbox_w * 0.5
                        anchor_v = bbox_y + bbox_h * 0.5
                    else:
                        anchor_u, anchor_v = anchor_uv
                    anchor_u, anchor_v = clamp_point(anchor_u, anchor_v, w_eye, height)

                    anchor_z_raw = sample_depth_median(
                        depth_frame,
                        anchor_u,
                        anchor_v,
                        args.sample_k,
                        meta_w,
                        meta_h,
                        w_eye,
                        height,
                        crop_x0,
                        crop_y0,
                        crop_w,
                        crop_h,
                    )

                    state = track_state.setdefault(track_id, TrackState())
                    anchor_z = smooth_value(state.anchor_z, anchor_z_raw, args.ema_alpha)
                    state.anchor_z = anchor_z

                    anchor_z_q = quantize_int16(anchor_z, args.quant_pos_scale)
                    anchor_scale_q = 65535

                    keypoints = obj["keypoints"]
                    kp_vis = obj.get("kp_vis") or []
                    kp_expected = int(obj.get("kp_expected", 0))
                    has_skeleton = bool(obj.get("has_skeleton", False))
                    flags = 1 if has_skeleton else 0
                    joints_rel_q_local = None

                    payload.extend(struct.pack("<I", track_id))
                    payload.extend(struct.pack("<B", category_id))
                    payload.extend(struct.pack("<B", flags))
                    payload.extend(
                        struct.pack(
                            "<HHHH",
                            int(round(bbox_x)),
                            int(round(bbox_y)),
                            int(round(bbox_w)),
                            int(round(bbox_h)),
                        )
                    )
                    payload.extend(struct.pack("<HH", int(round(anchor_u)), int(round(anchor_v))))
                    payload.extend(struct.pack("<h", anchor_z_q))
                    payload.extend(struct.pack("<H", anchor_scale_q))
                    payload.extend(struct.pack("<hhhh", *ROT_Q))

                    if has_skeleton and keypoints is not None:
                        if state.kp_count != kp_expected:
                            state.joints_rel = None
                            state.kp_count = kp_expected
                        prev_joints_rel = state.joints_rel
                        joints_rel = compute_joints_rel(
                            keypoints=keypoints,
                            depth_frame=depth_frame,
                            w_eye=w_eye,
                            height=height,
                            fovx_deg=args.fovx_deg,
                            sample_k=args.sample_k,
                            conf_th=args.conf_th,
                            ema_alpha=args.ema_alpha,
                            prev_joints_rel=prev_joints_rel,
                            meta_w=meta_w,
                            meta_h=meta_h,
                            crop_x0=crop_x0,
                            crop_y0=crop_y0,
                            crop_w=crop_w,
                            crop_h=crop_h,
                            root_indices=obj.get("root_indices", []),
                        )
                        state.joints_rel = joints_rel
                        joints_rel_q = quantize_array_int16(joints_rel, args.quant_joint_scale)
                        joints_rel_q_local = joints_rel_q
                        payload.extend(
                            struct.pack("<" + "h" * (kp_expected * 3), *joints_rel_q.reshape(-1).tolist())
                        )
                        if len(kp_vis) != kp_expected:
                            kp_vis = (kp_vis + [0] * kp_expected)[:kp_expected]
                        payload.extend(struct.pack("<" + "B" * kp_expected, *kp_vis))
                    if debug_detail and debug_objects is not None:
                        vis_count = 0
                        joints_q_min = None
                        joints_q_max = None
                        if has_skeleton:
                            vis_count = sum(1 for v in kp_vis if v > 0)
                            if joints_rel_q_local is not None:
                                joints_q_min = int(joints_rel_q_local.min())
                                joints_q_max = int(joints_rel_q_local.max())
                        debug_objects.append(
                            {
                                "track_id": track_id,
                                "category_id": category_id,
                                "category_name": cat_name_map.get(category_id, f"cat_{category_id}"),
                                "bbox": (bbox_x, bbox_y, bbox_w, bbox_h),
                                "anchor_uv": (anchor_u, anchor_v),
                                "anchor_z_raw": anchor_z_raw,
                                "anchor_z": anchor_z,
                                "anchor_z_q": anchor_z_q,
                                "has_skeleton": has_skeleton,
                                "kp_count": kp_expected,
                                "vis_count": vis_count,
                                "joints_q_min": joints_q_min,
                                "joints_q_max": joints_q_max,
                            }
                        )

                compressed = compress_fn(bytes(payload))
                offset = meta_f.tell()
                if debug_frame:
                    log_frame_summary(
                        frame_idx,
                        offset,
                        len(payload),
                        len(compressed),
                        len(objects),
                        debug_objects,
                        args,
                    )
                offsets.append(offset)
                meta_f.write(struct.pack("<I", len(compressed)))
                meta_f.write(compressed)

                if frame_idx % 50 == 0 or frame_idx + 1 == num_frames:
                    print(f"Processed frame {frame_idx + 1}/{num_frames}")

            index_table_offset = meta_f.tell()
            for off in offsets:
                meta_f.write(struct.pack("<Q", off))
            meta_f.seek(0)
            meta_f.write(
                pack_header(
                    compress_id=compress_id,
                    width=width,
                    height=height,
                    fps=fps,
                    num_frames=num_frames,
                    w_eye=w_eye,
                    fovx_deg=args.fovx_deg,
                    quant_pos_scale=args.quant_pos_scale,
                    quant_joint_scale=args.quant_joint_scale,
                    category_table_offset=category_table_offset,
                    category_table_size=category_table_size,
                    index_table_offset=index_table_offset,
                )
            )
            meta_f.flush()
            if debug_enabled:
                log_header_and_categories(
                    version=VERSION,
                    compress_id=compress_id,
                    compress_name=compress_name,
                    width=width,
                    height=height,
                    fps=fps,
                    num_frames=num_frames,
                    w_eye=w_eye,
                    fovx_deg=args.fovx_deg,
                    quant_pos_scale=args.quant_pos_scale,
                    quant_joint_scale=args.quant_joint_scale,
                    category_table_offset=category_table_offset,
                    category_table_size=category_table_size,
                    index_table_offset=index_table_offset,
                    cat_specs=cat_specs,
                    label="final",
                    include_categories=False,
                )
            if args.debug_meta_decode_after:
                verify_meta_bin(meta_path, debug_first_frames)

        manifest = {
            "width": width,
            "height": height,
            "eye_w": w_eye,
            "eye_h": height,
            "meta_w": meta_w,
            "meta_h": meta_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "align128": args.align128,
            "crop_mode": args.crop_mode,
            "fps": fps,
            "num_frames": num_frames,
            "left_eye_origin": args.left_eye_origin,
            "fovx_deg": args.fovx_deg,
            "quant_pos_scale": args.quant_pos_scale,
            "quant_joint_scale": args.quant_joint_scale,
            "frame_compress": compress_name,
            "video_transcode": transcode_info,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "inputs": {
                "video_mp4": os.path.basename(args.video_mp4),
                "video_mp4_bundle_source": os.path.basename(video_mp4_path),
                "depth_npz": os.path.basename(args.depth_npz),
                "metadata_json": os.path.basename(args.metadata_json),
            },
        }

        with zipfile.ZipFile(args.out_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(video_mp4_path, arcname="video.mp4", compress_type=zipfile.ZIP_STORED)
            zf.write(meta_path, arcname="meta.bin", compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
    finally:
        if os.path.exists(meta_path):
            os.remove(meta_path)


def main() -> None:
    args = parse_args()
    transcode_info: Dict[str, Any] = {"enabled": bool(args.transcode_video)}
    video_mp4_path = args.video_mp4

    if args.transcode_video:
        with tempfile.TemporaryDirectory() as tmpdir:
            transcoded_path = os.path.join(tmpdir, "transcoded.mp4")
            fps = probe_ffprobe_avg_fps(args.video_mp4)
            has_audio = probe_ffprobe_has_audio(args.video_mp4)
            transcode_result = transcode_for_quest(args.video_mp4, transcoded_path, fps, args)
            transcode_info.update(
                {
                    "profile": args.transcode_profile,
                    "level": args.transcode_level,
                    "pix_fmt": "yuv420p",
                    "fps": fps,
                    "crf": args.transcode_crf,
                    "preset": args.transcode_preset,
                    "has_audio": has_audio,
                    "audio_bitrate": args.transcode_audio_bitrate,
                    "audio_rate": args.transcode_audio_rate,
                    "output_name": os.path.basename(transcoded_path),
                }
            )
            transcode_info.update(transcode_result)
            build_bundle(args, transcoded_path, transcode_info)
    else:
        build_bundle(args, video_mp4_path, transcode_info)


if __name__ == "__main__":
    main()
