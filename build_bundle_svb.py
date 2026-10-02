#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import zlib

# scripts/bundle_common is shared with the sidecar scripts (repo root on
# sys.path; the orchestrator runs this file with cwd=StereoCrafter).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.bundle_common.depth_io import load_depth_npz  # noqa: E402

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

# Axis conventions of pose.keypoints3d in the sidecar (pose.cameraAxes or
# pose.coordinateSystem) and of the joints written to meta.bin.
#
# BUNDLE_JOINT_AXES is what Unity consumes and must not move. Evidence: the
# shipped human/animal bundles were built with --pose_keypoints3d_flip_y 1
# applied to HMR2/AniMer joints, which both engines emit in the OpenCV frame
# (+y down); negating y once gives +y up, the manifest's camera_axes /
# pose_keypoints3d_policy.output_axes = x_right_y_up_z_forward, and
# camera_xyz_from_uv_depth (the anchor_xyz that camera_xyz_absolute joints are
# added to) uses y_ndc = 0.5 - v/h, i.e. y up. So meta.bin joints are
# root-relative offsets in x_right_y_up_z_forward.
#
# Source labels:
# - AXES_Y_DOWN: root and joints in ONE OpenCV frame (scripts/bundle_common/
#   geometry.py AXIS_CONVENTION, sidecars from 2026-09-17 on). Flip once for
#   the bundle; the root re-projects with the OpenCV formula.
# - AXES_Y_UP_LEGACY: the label the sidecar scripts wrote before 2026-09-17
#   while the joint offsets were still engine-native (+y down) and only the
#   animal root (project_animal_keypoints_to_camera.py) was genuinely +y up.
#   Flow-audit E1: trusting the label and flipping the whole array put the
#   root into +y down and the y-up re-projection then MIRRORED anchor_v about
#   the eye centre. Handled as what those files actually are: joints flipped
#   (unchanged for Unity), root treated as +y up.
# - no label: engine-native, same as AXES_Y_DOWN.
AXES_Y_DOWN = "x_right_y_down_z_forward"
AXES_Y_UP_LEGACY = "x_right_y_up_z_forward"
BUNDLE_JOINT_AXES = "x_right_y_up_z_forward"

TYPE_OTHER = 0
TYPE_PERSON = 1
TYPE_ANIMAL = 2
FLAG_SKELETON = 1 << 0
FLAG_SMPL = 1 << 1
FLAG_SMAL = 1 << 2
SMPL_BLOCK_VERSION = 1
SMPL_ROTATION_COUNT = 24
SMPL_BETA_COUNT = 10
SMAL_BLOCK_VERSION = 1
SMAL_ROTATION_COUNT = 35
SMAL_BETA_COUNT = 41

ROT_Q = (0, 0, 0, 32767)
COCO_LHIP = 11
COCO_RHIP = 12
ANIMER_SMAL_26_SKELETON = [
    (0, 24),
    (1, 24),
    (2, 24),
    (3, 14),
    (4, 15),
    (5, 16),
    (6, 17),
    (7, 18),
    (8, 12),
    (9, 13),
    (10, 7),
    (11, 7),
    (12, 18),
    (13, 18),
    (14, 8),
    (15, 9),
    (16, 10),
    (17, 11),
    (18, 24),
    (19, 25),
    (20, 0),
    (21, 1),
    (22, 24),
    (23, 24),
    (25, 7),
]


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
    joints_abs: Optional[np.ndarray] = None
    kp_count: int = 0
    bbox_area: Optional[float] = None
    last_good_anchor_u: Optional[float] = None
    last_good_anchor_v: Optional[float] = None
    last_good_anchor_z: Optional[float] = None
    last_good_anchor_source: Optional[str] = None
    placement_low_streak: int = 0
    last_seen_frame: Optional[int] = None

    def reset(self) -> None:
        """Cold start: forget every smoothed/held quantity. Applied at a shot's
        first frame, after any in-shot track gap and across a shot change
        (flow-audit B1/B2) -- the EMA and the last-good hold gate must never
        compare a fresh observation against a value from before the gap."""
        self.anchor_z = None
        self.joints_rel = None
        self.joints_abs = None
        self.bbox_area = None
        self.last_good_anchor_u = None
        self.last_good_anchor_v = None
        self.last_good_anchor_z = None
        self.last_good_anchor_source = None
        self.placement_low_streak = 0


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
    parser.add_argument(
        "--person_anchor_mask_median",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Take the person track's anchor_z as the median disparity over the whole "
            "SAM2 mask instead of a 7x7 window at the pelvis keypoint. Fixes the "
            "self-occlusion jumps in D-004 (the subject's own forearm crossing the "
            "anchor point); does not improve how well anchor_z tracks true distance. "
            "anchor_u/anchor_v are unchanged -- only z. person only: animal never "
            "reaches this branch (animal_camera_root) and `other` was measured in "
            "D-005 to give the same value either way."
        ),
    )
    parser.add_argument(
        "--background_drift_correction",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "DepthCrafter's raw disparity drifts in absolute scale over a few "
            "seconds even for pixels whose real depth cannot change (see "
            "docs/bundle-shared/D-004-anchor-z-accuracy.md and archive/D-005-depth-sampling-window.md -- confirmed on a "
            "provably-static background patch). This subtracts that drift, "
            "estimated per shot from a fixed set of never-occluded background "
            "pixels, from every track's raw sampled disparity before it "
            "becomes anchor_z. Applies only to the window-sampled depth path "
            "(not animal_camera_root). Measured to meaningfully improve "
            "D-004's R^2 metric on FINNAL_HUMAN (person 0.143->0.243, ball "
            "0.533->0.646) but to make it worse on FINNAL_TRAIN -- the single "
            "background reference can only correct the additive (b) term of "
            "disparity=a/Z+b, not the multiplicative (a) term, and that gap "
            "matters more on some content than others. A peak-occlusion gate "
            "(--background_drift_max_occlusion_frac) auto-disables it per shot "
            "on content resembling the failure case, but this is not yet "
            "validated broadly -- default OFF, opt in per video after "
            "checking the printed R^2/occlusion diagnostics. 0 restores the "
            "previous (uncorrected) behavior."
        ),
    )
    parser.add_argument(
        "--background_drift_min_px",
        type=int,
        default=2000,
        help=(
            "Minimum background pixel count (frame minus all tracked masks) "
            "for a frame's background disparity to be trusted directly; frames "
            "below this hold the nearest earlier trusted value."
        ),
    )
    parser.add_argument(
        "--background_drift_max_occlusion_frac",
        type=float,
        default=0.10,
        help=(
            "If tracked objects' actual segmentation masks (not bbox area) "
            "ever cover more than this fraction of the crop area within a "
            "shot, background_drift_correction is disabled for that shot (the "
            "background reference gets unreliable -- see "
            "docs/bundle-shared/D-004-anchor-z-accuracy.md and archive/D-005-depth-sampling-window.md). Measured: peak "
            "mask-based occlusion was 5.4%% on FINNAL_HUMAN (correction "
            "helped) vs 20.2%% on FINNAL_TRAIN (correction hurt); 10%% sits "
            "between them. This is a correlated proxy, not a proven causal "
            "threshold."
        ),
    )
    parser.add_argument(
        "--depth_scale_calibration",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Solves a per-shot (a, b) pair for disparity=a/Z+b (Z in meters) "
            "from two fixed static background patches at assumed real-world "
            "distances --depth_scale_z_near/--depth_scale_z_far, and adds "
            "them to manifest.json under depth_scale_calibration so Unity "
            "can compute 1/(disparity-b) for exact relative placement (see "
            "docs/bundle-shared/D-004-anchor-z-accuracy.md and archive/D-005-depth-sampling-window.md). Does not change any "
            "existing anchor_z/z01 value -- purely additive sidecar data. "
            "Requires --depth_scale_near_box/--depth_scale_far_box (both "
            "must be genuinely never covered by any tracked object across "
            "the whole video -- verify with a per-frame bbox overlap check "
            "before trusting a candidate box). Validated only on "
            "FINNAL_HUMAN so far (CV of predicted/true distance ratio "
            "0.086-0.098 against person+ball references) -- the two boxes' "
            "assumed real-world distances must come from an independent, "
            "trustworthy source per video; a guessed z_far previously "
            "produced a physically impossible negative implied distance. "
            "Default OFF."
        ),
    )
    parser.add_argument(
        "--depth_scale_near_box",
        type=str,
        default=None,
        help="Near reference patch, crop-space pixels 'x0,y0,x1,y1'. Required if --depth_scale_calibration 1.",
    )
    parser.add_argument(
        "--depth_scale_far_box",
        type=str,
        default=None,
        help="Far reference patch, crop-space pixels 'x0,y0,x1,y1'. Required if --depth_scale_calibration 1.",
    )
    parser.add_argument(
        "--depth_scale_z_near",
        type=float,
        default=None,
        help="Assumed real-world distance (meters) of the near reference patch. Required if --depth_scale_calibration 1.",
    )
    parser.add_argument(
        "--depth_scale_z_far",
        type=float,
        default=None,
        help="Assumed real-world distance (meters) of the far reference patch. Required if --depth_scale_calibration 1.",
    )
    parser.add_argument("--conf_th", type=float, default=0.4)
    parser.add_argument("--depth_gate_range", type=float, default=1e9)
    parser.add_argument("--depth_gate_iqr", type=float, default=0.06)
    parser.add_argument("--depth_gate_mad", type=float, default=0.05)
    parser.add_argument("--depth_gate_min_valid", type=int, default=9)
    parser.add_argument("--depth_gate_jump", type=float, default=1e9)
    parser.add_argument("--depth_gate_conf_margin", type=float, default=0.0)
    parser.add_argument("--depth_gate_prev_band", type=float, default=0.03)
    parser.add_argument("--depth_gate_prev_min_valid", type=int, default=9)
    parser.add_argument("--depth_gate_prev_min_frac", type=float, default=0.4)
    parser.add_argument("--depth_gate_use_anchor_fallback", type=int, choices=[0, 1], default=1)
    parser.add_argument("--placement_conf_hold_threshold", type=float, default=0.55)
    parser.add_argument("--placement_conf_edge_margin_px", type=float, default=4.0)
    parser.add_argument("--placement_conf_area_shrink_ratio", type=float, default=0.55)
    parser.add_argument("--placement_conf_anchor_jump_px", type=float, default=64.0)
    parser.add_argument("--placement_conf_depth_jump", type=float, default=0.08)
    parser.add_argument("--ema_alpha", type=float, default=0.8)
    parser.add_argument(
        "--quant_pos_scale",
        type=float,
        default=0.0002,
        help=(
            "anchor_z quantization step. For depth-sampled anchors, anchor_z is "
            "1 - normalized disparity, so the far field is compressed into a narrow "
            "band; at the old 0.002 step distinct objects in the same frame collapsed "
            "onto one step and Unity placed them at an identical camera Z (D-008). "
            "The ceiling on how fine this can go is the animal_camera_root path, "
            "which writes AniMer's camera-space root Z directly and is NOT bounded by "
            "1.0: at 0.0002 anchor_z_q covers up to 6.55, roughly 9x the largest value "
            "seen in an animal bundle. Clips whose anchors are all depth-sampled "
            "(anchor_z < 1 by construction) can safely use 0.0001."
        ),
    )
    parser.add_argument("--quant_joint_scale", type=float, default=0.002)
    parser.add_argument(
        "--joints_source",
        choices=["auto", "depth_from_2d", "pose_keypoints3d"],
        default="auto",
        help=(
            "auto uses pose.keypoints3d when present, otherwise falls back to 2D keypoints + depth. "
            "pose_keypoints3d treats pose-engine 3D as skeleton shape and uses depth only for anchor placement."
        ),
    )
    parser.add_argument(
        "--metrabs_joint_scale",
        type=float,
        default=0.001,
        help="Scale applied to MeTRAbs/metrabs_camera joints before bundling. Default converts mm to m.",
    )
    parser.add_argument(
        "--animer_joint_scale",
        type=float,
        default=1.0,
        help="Scale applied to AniMer/animer_smal joints before bundling.",
    )
    parser.add_argument(
        "--pose_keypoints3d_flip_y",
        choices=["auto", "0", "1"],
        default="auto",
        help=(
            "auto (default): derive the Y flip per object from the sidecar's axis "
            "label (pose.cameraAxes / pose.coordinateSystem: x_right_y_down_z_forward "
            "or the legacy x_right_y_up_z_forward) so that meta.bin joints are always "
            "x_right_y_up_z_forward and the animal root re-projects to the pixel it "
            "was lifted from. 0/1: the pre-2026-09-17 blind flag (whole array, root "
            "re-projected on the eye canvas with the y-up formula), kept only to "
            "reproduce old bundles bit-for-bit -- it mirrors anchor_v for the legacy "
            "animal sidecars (flow-audit E1)."
        ),
    )
    parser.add_argument(
        "--joints_space",
        choices=["camera_xyz_root_relative", "camera_xyz_absolute"],
        default="camera_xyz_root_relative",
    )
    parser.add_argument("--frame_compress", choices=["none", "zlib", "lz4"], default="lz4")
    parser.add_argument(
        "--shots_json",
        type=str,
        default=None,
        help=(
            "Optional JSON file: a list of [start, end) frame ranges covering the whole "
            "video, one per hard camera cut (same convention as "
            "scripts/run_rose_inpaint_with_shots.py --shots). Default: treat the whole "
            "video as a single shot. Written to manifest.json 'shots', and used to reset "
            "per-track EMA/placement-hold state at each shot's first frame so a legitimate "
            "camera-distance jump at a cut isn't smeared or gated like sensor noise."
        ),
    )
    parser.add_argument("--anchor_from", choices=["mask", "bbox"], default="mask")
    parser.add_argument("--fps", type=float, default=None, help="Manual fallback if video probing fails.")
    parser.add_argument("--width", type=int, default=None, help="Manual fallback if video probing fails.")
    parser.add_argument("--height", type=int, default=None, help="Manual fallback if video probing fails.")
    parser.add_argument("--transcode_video", type=int, choices=[0, 1], default=1)
    parser.add_argument("--transcode_crf", type=int, default=18)
    parser.add_argument("--transcode_preset", type=str, default="veryfast")
    parser.add_argument("--transcode_profile", choices=["baseline", "main"], default="main")
    parser.add_argument(
        "--transcode_level",
        type=str,
        default="auto",
        help=(
            "H.264 level for the Quest transcode. auto derives it from the video's "
            "macroblocks per frame and per second (4.1 when both fit, else 5.1, else "
            "5.2) instead of stamping 4.1 on a 60 fps or 1080p-tall SBS video (D6)."
        ),
    )
    parser.add_argument("--transcode_audio_bitrate", type=str, default="128k")
    parser.add_argument("--transcode_audio_rate", type=int, default=48000)
    parser.add_argument("--debug_meta", type=int, choices=[0, 1], default=0)
    parser.add_argument("--debug_meta_every", type=int, default=50)
    parser.add_argument("--debug_meta_first_frames", type=int, default=3)
    parser.add_argument("--debug_meta_max_objects", type=int, default=10)
    parser.add_argument("--debug_meta_decode_after", type=int, choices=[0, 1], default=0)
    parser.add_argument("--debug_frame", type=int, default=-1, help="Emit detailed joints debug for this frame index.")
    parser.add_argument(
        "--dump_manifest",
        type=str,
        default="",
        help="Optional path to dump raw manifest.json for verification.",
    )
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


# H.264 level limits (Table A-1): max macroblocks per frame, per second.
H264_LEVEL_LIMITS = [
    ("4.1", 8192, 245760),
    ("5.1", 36864, 983040),
    ("5.2", 36864, 2073600),
]


def select_h264_level(width: int, height: int, fps: float) -> str:
    """Smallest level in H264_LEVEL_LIMITS whose frame-size and throughput
    limits the video fits; the last one when nothing fits."""
    mb_per_frame = math.ceil(max(1, width) / 16.0) * math.ceil(max(1, height) / 16.0)
    mb_per_sec = mb_per_frame * max(0.0, float(fps))
    for level, max_mb_frame, max_mb_sec in H264_LEVEL_LIMITS:
        if mb_per_frame <= max_mb_frame and mb_per_sec <= max_mb_sec:
            return level
    return H264_LEVEL_LIMITS[-1][0]


def resolve_transcode_level(args: argparse.Namespace, input_path: str, fps: float) -> str:
    if str(args.transcode_level).lower() != "auto":
        return str(args.transcode_level)
    meta = probe_ffprobe(input_path) or probe_cv2(input_path)
    if meta is None or meta.width <= 0 or meta.height <= 0:
        raise RuntimeError(f"Cannot derive the H.264 level: failed to probe {input_path}.")
    return select_h264_level(meta.width, meta.height, fps)


def transcode_for_quest(
    input_path: str, output_path: str, fps: float, args: argparse.Namespace
) -> Dict[str, Any]:
    ffmpeg_bin = require_ffmpeg()
    if fps <= 0:
        raise RuntimeError("Invalid FPS for transcoding.")
    encoder = select_h264_encoder()
    profile = map_profile_for_encoder(encoder, args.transcode_profile)
    level = resolve_transcode_level(args, input_path, fps)
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
                level,
                "-preset",
                args.transcode_preset,
                "-crf",
                str(args.transcode_crf),
            ]
        )
        level_applied = level
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
        "level": level,
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
    depth, _disp_min, _disp_max = load_depth_with_disp_range(path)
    return depth


def load_depth_with_disp_range(
    path: str,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Load the (T,H,W) depth array plus the optional pre-normalization
    (disp_min, disp_max) written by depth_splatting_inference.py. Older depth
    npz files without those keys yield (None, None). Both the legacy float
    npz and the uint16_fixed format are read by the shared
    scripts/bundle_common/depth_io.py reader, so the two never diverge here.
    """
    loaded = load_depth_npz(path)
    return loaded.depth, loaded.disp_min, loaded.disp_max


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
    keys = {"bbox", "box", "keypoints", "segmentation", "mask", "bbox_xyxy", "xyxy", "sam2", "pose"}
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


def parse_index_list(value: Any, kp_count: int) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        return None
    indices: List[int] = []
    for item in raw_items:
        try:
            idx = int(item)
        except Exception:
            continue
        if 0 <= idx < kp_count:
            indices.append(idx)
    return indices or None


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
        if isinstance(categories, dict):
            name_to_id = {"other": TYPE_OTHER, "person": TYPE_PERSON, "human": TYPE_PERSON, "animal": TYPE_ANIMAL}
            for name_key, cat in categories.items():
                if not isinstance(cat, dict):
                    continue
                name = str(cat.get("name") or name_key)
                cat_id_int = int(cat.get("id", name_to_id.get(name.lower(), name_to_id.get(str(name_key).lower(), TYPE_OTHER))))
                kp_names = cat.get("keypoints")
                if isinstance(kp_names, list):
                    kp_names = [str(kp) for kp in kp_names]
                else:
                    kp_names = []
                kp_count = len(kp_names)
                edges = normalize_skeleton_edges(cat.get("skeleton"), kp_count)
                root_indices = infer_root_indices(kp_names, kp_count)
                explicit_root_indices = parse_index_list(
                    cat.get("root_joint_indices", cat.get("rootJointIndices")),
                    kp_count,
                )
                if explicit_root_indices is not None:
                    root_indices = explicit_root_indices
                if cat_id_int == TYPE_ANIMAL and kp_count == 26:
                    if not edges:
                        edges = ANIMER_SMAL_26_SKELETON
                    if explicit_root_indices is None and all(name.startswith("animer_joint_") for name in kp_names):
                        root_indices = []
                anchor_indices = infer_anchor_indices(cat_id_int, kp_names, kp_count)
                explicit_anchor_indices = parse_index_list(
                    cat.get("anchor_joint_indices", cat.get("anchorJointIndices")),
                    kp_count,
                )
                if explicit_anchor_indices is not None:
                    anchor_indices = explicit_anchor_indices
                specs[cat_id_int] = {
                    "id": cat_id_int,
                    "name": "person" if cat_id_int == TYPE_PERSON else ("animal" if cat_id_int == TYPE_ANIMAL else name),
                    "kp_names": kp_names,
                    "kp_count": kp_count,
                    "skeleton_edges": edges,
                    "root_indices": root_indices,
                    "anchor_indices": anchor_indices,
                    "engine": cat.get("engine"),
                    "keypoint_format": cat.get("keypoint_format"),
                    "coordinate_system": cat.get("coordinate_system"),
                }
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
                explicit_root_indices = parse_index_list(
                    cat.get("root_joint_indices", cat.get("rootJointIndices")),
                    kp_count,
                )
                if explicit_root_indices is not None:
                    root_indices = explicit_root_indices
                if cat_id_int == TYPE_ANIMAL and kp_count == 26:
                    if not edges:
                        edges = ANIMER_SMAL_26_SKELETON
                    if explicit_root_indices is None and all(name.startswith("animer_joint_") for name in kp_names):
                        root_indices = []
                anchor_indices = infer_anchor_indices(cat_id_int, kp_names, kp_count)
                explicit_anchor_indices = parse_index_list(
                    cat.get("anchor_joint_indices", cat.get("anchorJointIndices")),
                    kp_count,
                )
                if explicit_anchor_indices is not None:
                    anchor_indices = explicit_anchor_indices
                specs[cat_id_int] = {
                    "id": cat_id_int,
                    "name": name,
                    "kp_names": kp_names,
                    "kp_count": kp_count,
                    "skeleton_edges": edges,
                    "root_indices": root_indices,
                    "anchor_indices": anchor_indices,
                    "engine": cat.get("engine"),
                    "keypoint_format": cat.get("keypoint_format"),
                    "coordinate_system": cat.get("coordinate_system"),
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
    sam2 = obj.get("sam2")
    if isinstance(sam2, dict):
        bbox = sam2.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        bounds = sam2.get("bounds")
        if (
            isinstance(bounds, (list, tuple))
            and len(bounds) >= 2
            and isinstance(bounds[0], (list, tuple))
            and isinstance(bounds[1], (list, tuple))
            and len(bounds[0]) >= 2
            and len(bounds[1]) >= 2
        ):
            x0, y0 = float(bounds[0][0]), float(bounds[0][1])
            x1, y1 = float(bounds[1][0]), float(bounds[1][1])
            return x0, y0, x1 - x0, y1 - y0
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
    nested_pose = obj.get("pose")
    if isinstance(nested_pose, dict) and "keypoints2d" in nested_pose:
        obj = {**obj, "keypoints": nested_pose.get("keypoints2d")}
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


def pose_keypoints2d_units(obj: Dict[str, Any]) -> Optional[str]:
    """The sidecar's explicit pose.keypoints2dUnits ("pixels" or
    "crop_normalized"), or None for files written before the key existed."""
    pose = obj.get("pose")
    if not isinstance(pose, dict):
        return None
    units = pose.get("keypoints2dUnits")
    if not isinstance(units, str) or not units:
        return None
    units = units.lower()
    if units in ("pixels", "px", "pixel"):
        return "pixels"
    if "normal" in units:
        return "crop_normalized"
    return None


def keypoints_look_like_pixels(
    keypoints: Optional[List[Tuple[float, float, float]]],
    kp_vis: Sequence[int],
    width: int,
    height: int,
) -> bool:
    """Heuristic for sidecars without pose.keypoints2dUnits. width/height
    must be the SOURCE canvas the keypoints were measured on (flow-audit B5:
    passing the depth-plane size discarded valid keypoints in the right/
    bottom 20% of the frame)."""
    if keypoints is None:
        return False
    visible: List[Tuple[float, float]] = []
    for idx, (u, v, conf) in enumerate(keypoints):
        if idx < len(kp_vis) and kp_vis[idx] <= 0:
            continue
        if normalize_conf(conf) <= 0:
            continue
        if math.isfinite(u) and math.isfinite(v):
            visible.append((float(u), float(v)))
    if not visible:
        return False
    max_abs = max(max(abs(u), abs(v)) for u, v in visible)
    if max_abs <= 2.0:
        return False
    in_frame = sum(1 for u, v in visible if 0.0 <= u < float(width) and 0.0 <= v < float(height))
    return in_frame >= max(1, len(visible) // 4)


def denormalize_pose_keypoints2d_from_source_box(
    obj: Dict[str, Any],
    keypoints: Optional[List[Tuple[float, float, float]]],
    kp_vis: Sequence[int],
    width: int,
    height: int,
) -> Optional[List[Tuple[float, float, float]]]:
    if keypoints is None:
        return None
    pose = obj.get("pose")
    if not isinstance(pose, dict):
        return None
    source_box = pose.get("sourceBox")
    if not isinstance(source_box, dict):
        return None
    xywh = source_box.get("xywh")
    if not isinstance(xywh, (list, tuple)) or len(xywh) < 4:
        return None
    try:
        box_x, box_y, box_w, box_h = (float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3]))
    except Exception:
        return None
    if box_w <= 0.0 or box_h <= 0.0:
        return None

    visible = []
    for idx, (u, v, conf) in enumerate(keypoints):
        if idx < len(kp_vis) and kp_vis[idx] <= 0:
            continue
        if normalize_conf(conf) <= 0.0:
            continue
        if math.isfinite(u) and math.isfinite(v):
            visible.append((float(u), float(v)))
    if not visible:
        return None
    if max(max(abs(u), abs(v)) for u, v in visible) > 2.0:
        return None

    cx = box_x + box_w * 0.5
    cy = box_y + box_h * 0.5
    box_size = max(box_w, box_h)
    converted: List[Tuple[float, float, float]] = []
    in_frame = 0
    for u, v, conf in keypoints:
        if not math.isfinite(u) or not math.isfinite(v):
            converted.append((0.0, 0.0, 0.0))
            continue
        u_px = cx + float(u) * box_size
        v_px = cy + float(v) * box_size
        if 0.0 <= u_px < float(width) and 0.0 <= v_px < float(height):
            in_frame += 1
        converted.append((u_px, v_px, conf))
    if in_frame < max(1, len(visible) // 4):
        return None
    return converted


def parse_pose_keypoints3d(
    obj: Dict[str, Any], expected_kp_count: int
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[str], Optional[str], Optional[str]]:
    pose = obj.get("pose")
    if not isinstance(pose, dict):
        return None, np.zeros((0,), dtype=bool), None, None, None
    raw_points = pose.get("keypoints3d")
    if not isinstance(raw_points, (list, tuple)) or not raw_points:
        return None, np.zeros((0,), dtype=bool), None, None, None

    points: List[List[float]] = []
    valid: List[bool] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            points.append([0.0, 0.0, 0.0])
            valid.append(False)
            continue
        xyz = [float(point[0]), float(point[1]), float(point[2])]
        conf = float(point[3]) if len(point) >= 4 else 1.0
        finite = all(math.isfinite(v) for v in xyz)
        points.append(xyz if finite else [0.0, 0.0, 0.0])
        valid.append(finite and normalize_conf(conf) > 0.0)

    expected = max(0, int(expected_kp_count))
    if expected > len(points):
        for _ in range(expected - len(points)):
            points.append([0.0, 0.0, 0.0])
            valid.append(False)
    elif expected and expected < len(points):
        points = points[:expected]
        valid = valid[:expected]

    arr = np.asarray(points, dtype=np.float32)
    valid_arr = np.asarray(valid, dtype=bool)
    return (
        arr,
        valid_arr,
        str(pose.get("coordinateSystem") or ""),
        str(pose.get("engine") or ""),
        str(pose.get("keypointFormat") or ""),
    )


def _smpl_matrix_array(value: Any, expected: int) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if arr.shape == (expected, 3, 3):
        return arr
    if arr.size == expected * 9:
        return arr.reshape(expected, 3, 3)
    return None


def parse_smpl_payload(obj: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    pose = obj.get("pose")
    if not isinstance(pose, dict):
        return None
    smpl = pose.get("smpl")
    if not isinstance(smpl, dict):
        return None
    global_orient = _smpl_matrix_array(smpl.get("globalOrient"), 1)
    body_pose = _smpl_matrix_array(smpl.get("bodyPose"), 23)
    if global_orient is None or body_pose is None:
        return None
    try:
        betas = np.asarray(smpl.get("betas"), dtype=np.float32).reshape(-1)
        transl = np.asarray(smpl.get("transl"), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if betas.size < SMPL_BETA_COUNT or transl.size < 3:
        return None
    rotations = np.concatenate([global_orient.reshape(1, 3, 3), body_pose.reshape(23, 3, 3)], axis=0)
    return {
        "rotations": rotations.astype(np.float32),
        "betas": betas[:SMPL_BETA_COUNT].astype(np.float32),
        "transl": transl[:3].astype(np.float32),
    }


def pack_smpl_payload(smpl_payload: Dict[str, np.ndarray]) -> bytes:
    rotations = np.asarray(smpl_payload["rotations"], dtype="<f4").reshape(SMPL_ROTATION_COUNT, 3, 3)
    betas = np.asarray(smpl_payload["betas"], dtype="<f4").reshape(SMPL_BETA_COUNT)
    transl = np.asarray(smpl_payload["transl"], dtype="<f4").reshape(3)
    out = bytearray()
    out.extend(struct.pack("<HHH", SMPL_BLOCK_VERSION, SMPL_ROTATION_COUNT, SMPL_BETA_COUNT))
    out.extend(rotations.reshape(-1).tobytes())
    out.extend(betas.tobytes())
    out.extend(transl.tobytes())
    return bytes(out)


def parse_smal_payload(obj: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    pose = obj.get("pose")
    if not isinstance(pose, dict):
        return None
    smal = pose.get("smal")
    if not isinstance(smal, dict):
        return None
    global_orient = _smpl_matrix_array(smal.get("globalOrient"), 1)
    body_pose = _smpl_matrix_array(smal.get("pose"), SMAL_ROTATION_COUNT - 1)
    if global_orient is None or body_pose is None:
        return None
    try:
        betas = np.asarray(smal.get("betas"), dtype=np.float32).reshape(-1)
        transl_source = smal.get(
            "transl",
            smal.get("predCamTFull", pose.get("sourcePredCamTFull", [0.0, 0.0, 0.0])),
        )
        transl = np.asarray(transl_source, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if betas.size < SMAL_BETA_COUNT:
        return None
    if transl.size < 3:
        transl = np.zeros((3,), dtype=np.float32)
    rotations = np.concatenate(
        [global_orient.reshape(1, 3, 3), body_pose.reshape(SMAL_ROTATION_COUNT - 1, 3, 3)],
        axis=0,
    )
    return {
        "rotations": rotations.astype(np.float32),
        "betas": betas[:SMAL_BETA_COUNT].astype(np.float32),
        "transl": transl[:3].astype(np.float32),
    }


def pack_smal_payload(smal_payload: Dict[str, np.ndarray]) -> bytes:
    rotations = np.asarray(smal_payload["rotations"], dtype="<f4").reshape(SMAL_ROTATION_COUNT, 3, 3)
    betas = np.asarray(smal_payload["betas"], dtype="<f4").reshape(SMAL_BETA_COUNT)
    transl = np.asarray(smal_payload["transl"], dtype="<f4").reshape(3)
    out = bytearray()
    out.extend(struct.pack("<HHH", SMAL_BLOCK_VERSION, SMAL_ROTATION_COUNT, SMAL_BETA_COUNT))
    out.extend(rotations.reshape(-1).tobytes())
    out.extend(betas.tobytes())
    out.extend(transl.tobytes())
    return bytes(out)


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


def source_is_full_sbs(source_w: int, video_w: int, w_eye: int, left_eye_origin: str) -> bool:
    """True when the object's coordinates live on the full side-by-side
    canvas, i.e. the condition under which source_canvas_size shifts the
    origin and adjust_left_eye is meaningful."""
    return left_eye_origin == "full" and source_w == video_w and video_w == w_eye * 2


def shift_left_eye_if_full_sbs(value: float, w_eye: int, width: int, is_full_sbs: bool) -> float:
    """adjust_left_eye only for a full-SBS source canvas. On a single-eye
    source a pose keypoint extrapolated past the right edge used to be shifted
    by w_eye onto mid-frame with its confidence intact (flow-audit B6); now it
    is left where it is and the crop test downstream (meta_to_eye_point ->
    None -> vis 0) drops it exactly like a keypoint past any other edge."""
    if is_full_sbs:
        return adjust_left_eye(value, w_eye, width, "full")
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


def check_crop_fraction(
    *,
    crop_w: int,
    crop_h: int,
    full_meta_w: int,
    full_meta_h: int,
    w_eye: int,
    height: int,
    source_w: int,
    source_h: int,
    align128: int,
    tolerance: float = 0.005,
) -> Dict[str, float]:
    """The one invariant the source -> depth plane -> crop -> eye mapping
    relies on: the fraction of the depth plane the crop keeps must equal the
    fraction of the source frame the eye keeps, per axis. --align128 1
    satisfies it at 1280x720 by coincidence (512/576 == 640/720) and
    silently stretches every other size (1920x1080: 1.067x); an uncropped
    eye with --align128 1 fails it too. Raises SystemExit on violation so a
    mismatch introduced by any stage is a hard failure, never a stretch."""
    frac_w_meta = float(crop_w) / float(full_meta_w)
    frac_h_meta = float(crop_h) / float(full_meta_h)
    frac_w_eye = float(w_eye) / float(source_w)
    frac_h_eye = float(height) / float(source_h)
    result = {
        "crop_w_fraction": frac_w_meta,
        "crop_h_fraction": frac_h_meta,
        "eye_w_fraction": frac_w_eye,
        "eye_h_fraction": frac_h_eye,
    }
    if abs(frac_w_meta - frac_w_eye) <= tolerance and abs(frac_h_meta - frac_h_eye) <= tolerance:
        return result
    who = (
        f"build_bundle_svb.py --align128 {align128} cropped the depth plane to "
        f"{crop_w}x{crop_h} of {full_meta_w}x{full_meta_h}"
        if (crop_w != full_meta_w or crop_h != full_meta_h)
        else "build_bundle_svb.py kept the whole depth plane"
    )
    eye_note = (
        f"the stereo stage kept an eye of {w_eye}x{height} out of a {source_w}x{source_h} source"
        if (w_eye != source_w or height != source_h)
        else f"the eye {w_eye}x{height} is the full {source_w}x{source_h} source frame"
    )
    raise SystemExit(
        "Crop fraction mismatch: the metadata plane and the eye video do not cover the "
        f"same part of the frame. {who}, i.e. {frac_w_meta:.4f} x {frac_h_meta:.4f} of it, "
        f"while {eye_note}, i.e. {frac_w_eye:.4f} x {frac_h_eye:.4f}. Every anchor and bbox "
        f"would be stretched by {frac_w_eye / frac_w_meta:.3f} x {frac_h_eye / frac_h_meta:.3f} "
        "with no warning (the 2026-09-16 car job lost the bottom 80 rows this way). "
        "The supported configuration is --align128 0 with a full-frame eye "
        "(StereoCrafter/inpainting_inference_padded.py); with --align128 1 the eye "
        "must be cropped to the same 128-multiple fraction, which only 1280x720 does."
    )


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


def source_image_size(
    obj: Dict[str, Any], fallback_w: int, fallback_h: int
) -> Tuple[int, int]:
    """Return the coordinate canvas used by a SAM2-derived object."""
    sam2 = obj.get("sam2")
    segmentation = sam2.get("segmentation") if isinstance(sam2, dict) else None
    size = segmentation.get("size") if isinstance(segmentation, dict) else None
    if isinstance(size, (list, tuple)) and len(size) >= 2:
        try:
            height = int(size[0])
            width = int(size[1])
            if width > 0 and height > 0:
                return width, height
        except (TypeError, ValueError):
            pass
    return fallback_w, fallback_h


def source_canvas_size(
    source_w: int,
    source_h: int,
    video_w: int,
    w_eye: int,
    left_eye_origin: str,
) -> Tuple[int, int]:
    """Return the left-eye canvas after an optional full-SBS x-origin shift."""
    if left_eye_origin == "full" and source_w == video_w and video_w == w_eye * 2:
        return w_eye, source_h
    return source_w, source_h


def source_to_metadata_point(
    u: float,
    v: float,
    source_w: int,
    source_h: int,
    meta_w: int,
    meta_h: int,
) -> Tuple[float, float]:
    if source_w <= 0 or source_h <= 0:
        return u, v
    return u * float(meta_w) / float(source_w), v * float(meta_h) / float(source_h)


def source_to_metadata_bbox(
    bbox: Tuple[float, float, float, float],
    source_w: int,
    source_h: int,
    meta_w: int,
    meta_h: int,
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_meta, y_meta = source_to_metadata_point(x, y, source_w, source_h, meta_w, meta_h)
    w_meta = w * float(meta_w) / float(source_w) if source_w > 0 else w
    h_meta = h * float(meta_h) / float(source_h) if source_h > 0 else h
    return x_meta, y_meta, w_meta, h_meta


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


def sample_depth_mask_median(
    depth_frame: np.ndarray,
    segmentation: Any,
    source_w: int,
    source_h: int,
    full_meta_w: int,
    full_meta_h: int,
    crop_x0: int,
    crop_y0: int,
    stats_out: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Median disparity over the whole SAM2 mask instead of a 7x7 window.

    D-004 (2026-09-09): the person anchor is one point at the pelvis keypoint, so
    when the subject's own forearm swings across it the 7x7 window jumps between
    the hip surface and the arm surface -- 0.09 disparity in a single frame, which
    the EMA then smears into a half-second ramp Unity sees as "the person suddenly
    got bigger". Taking the median over the segmented body makes the value depend
    on the whole silhouette instead of which limb happens to cover one pixel.

    This is NOT a fix for D-004 itself: the median tracks true distance no better
    than the point sample did (measured: corr with bbox height 0.560 -> 0.496). It
    only removes the false jumps.

    Assumes the mask's source frame is a single eye (source_w == eye width). A
    side-by-side source would need adjust_left_eye applied per pixel, which is not
    implemented -- callers must not enable this for such inputs.
    """
    if depth_frame is None or segmentation is None:
        return None
    if source_w <= 0 or source_h <= 0 or full_meta_w <= 0 or full_meta_h <= 0:
        return None
    mask = decode_mask(segmentation, source_h, source_w)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    # depth_frame is already cropped to (crop_h, crop_w); the mask is in source
    # pixels of the uncropped frame. Go source -> full meta -> crop, and DROP the
    # pixels outside the crop rather than clipping them -- clipping squeezes the
    # rows below the eye view onto the bottom edge, which drags background
    # disparity into the median (the first attempt did exactly that: frame 224's
    # person read 0.50 instead of 0.66).
    h, w = depth_frame.shape
    xi = np.rint(xs * (float(full_meta_w) / float(source_w))).astype(np.int64) - int(crop_x0)
    yi = np.rint(ys * (float(full_meta_h) / float(source_h))).astype(np.int64) - int(crop_y0)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if not np.any(inside):
        if stats_out is not None:
            stats_out["valid_count"] = 0.0
        return None
    values = depth_frame[yi[inside], xi[inside]].astype(np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        if stats_out is not None:
            stats_out["valid_count"] = 0.0
        return None
    median = float(np.median(values))
    q1, q3 = (float(v) for v in np.percentile(values, [25.0, 75.0]))
    p10, p90 = (float(v) for v in np.percentile(values, [10.0, 90.0]))
    if stats_out is not None:
        stats_out.update(
            {
                "valid_count": float(values.size),
                "median": median,
                "iqr": q3 - q1,
                "mad": float(np.median(np.abs(values - median))),
                "p10": p10,
                "p90": p90,
            }
        )
    return median


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


def disparity_to_camera_z(disparity: float) -> float:
    """Flip depth_npz's normalized disparity (0.0=far, 1.0=near) into the
    bundle's camera-Z convention (larger=farther), matching animal_camera_root
    (which already flips via project_animal_keypoints_to_camera.py) and
    camera_xyz_from_uv_depth's projection math. Clamped away from exactly 0.0
    so a near-1.0 disparity sample can't fail downstream '>0' validity checks.
    """
    return max(1.0 - float(disparity), 1e-4)


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
    stats_out: Optional[Dict[str, float]] = None,
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
        if stats_out is not None:
            stats_out["valid_count"] = 0.0
        return None
    values = flat[mask]
    median = float(np.median(values))
    p10 = float(np.percentile(values, 10))
    p25 = float(np.percentile(values, 25))
    p75 = float(np.percentile(values, 75))
    p90 = float(np.percentile(values, 90))
    iqr = p75 - p25
    mad = float(np.median(np.abs(values - median)))
    if stats_out is not None:
        stats_out["valid_count"] = float(values.size)
        stats_out["median"] = median
        stats_out["p10"] = p10
        stats_out["p25"] = p25
        stats_out["p75"] = p75
        stats_out["p90"] = p90
        stats_out["iqr"] = iqr
        stats_out["mad"] = mad
    return median


def sample_depth_robust(
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
    min_valid: int,
    prev_band: float,
    prev_min_valid: int,
    prev_min_frac: float,
    tau_iqr: float,
    tau_range: float,
    tau_mad: float,
    tau_jump: float,
    conf_n: float,
    conf_th: float,
    conf_margin: float,
    prev_z: Optional[float],
) -> Tuple[Optional[float], Dict[str, float], Optional[str]]:
    stats: Dict[str, float] = {}
    z = sample_depth_median(
        depth_frame,
        u_eye,
        v_eye,
        sample_k,
        meta_w,
        meta_h,
        eye_w,
        eye_h,
        crop_x0,
        crop_y0,
        crop_w,
        crop_h,
        stats_out=stats,
    )
    if z is None:
        return None, stats, "no_valid_depth"
    valid_count = int(round(stats.get("valid_count", 0.0)))
    if valid_count < max(1, int(min_valid)):
        return None, stats, "gate_min_valid"
    stats["filtered_count"] = 0.0
    stats["filtered_frac"] = 0.0
    if prev_z is not None and math.isfinite(prev_z) and prev_z > 0.0 and float(prev_band) > 0.0:
        # Reuse sampled patch values by re-running median sampler would be costly; sample once here.
        # Pull values again from depth patch for band filtering.
        # This keeps behavior deterministic with current coordinate mapping.
        values_stats: Dict[str, float] = {}
        _ = sample_depth_median(
            depth_frame,
            u_eye,
            v_eye,
            sample_k,
            meta_w,
            meta_h,
            eye_w,
            eye_h,
            crop_x0,
            crop_y0,
            crop_w,
            crop_h,
            stats_out=values_stats,
        )
        if "valid_count" in values_stats and values_stats["valid_count"] > 0:
            # Need actual values for filtering; reconstruct from patch directly.
            meta_uv = eye_to_meta_point(u_eye, v_eye, crop_x0, crop_y0, crop_w, crop_h, eye_w, eye_h)
            if meta_uv is not None:
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
                vals = window.reshape(-1)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                filtered = vals[np.abs(vals - float(prev_z)) <= float(prev_band)]
                stats["filtered_count"] = float(filtered.size)
                filt_frac = float(filtered.size) / float(valid_count) if valid_count > 0 else 0.0
                stats["filtered_frac"] = filt_frac
                if (
                    filtered.size >= max(1, int(prev_min_valid))
                    and filt_frac >= max(0.0, float(prev_min_frac))
                ):
                    return float(np.median(filtered)), stats, "prev_band"
    iqr = stats.get("iqr", 0.0)
    depth_range = stats.get("p90", z) - stats.get("p10", z)
    iqr_bad = float(tau_iqr) > 0.0 and iqr > float(tau_iqr)
    mad_bad = float(tau_mad) > 0.0 and stats.get("mad", 0.0) > float(tau_mad)
    if iqr_bad and mad_bad:
        return None, stats, "gate_iqr_mad"
    # Legacy gate: keep as fallback and require both wide spread and high MAD.
    if float(tau_range) > 0.0 and float(tau_mad) > 0.0:
        if depth_range > float(tau_range) and mad_bad:
            return None, stats, "gate_range_and_mad"
    if (
        prev_z is not None
        and math.isfinite(prev_z)
        and prev_z > 0.0
        and abs(z - prev_z) > float(tau_jump)
        and conf_n < (float(conf_th) + float(conf_margin))
    ):
        return None, stats, "gate_jump_low_conf"
    return z, stats, None


def load_shots_for_bundle(path: Optional[str], total_frames: int) -> List[Tuple[int, int]]:
    """Loads a [start, end) shot-boundary list, or a single whole-video shot if
    no path is given. Same list-of-pairs JSON convention as
    scripts/run_rose_inpaint_with_shots.py / scripts/run_background_plate_inpaint.py."""
    if not path:
        return [(0, total_frames)]
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    shots = sorted((int(s), int(e)) for s, e in data)
    if not shots:
        raise ValueError(f"Empty shot list: {path}")
    if shots[0][0] != 0 or shots[-1][1] != total_frames:
        raise ValueError(
            f"Shots must cover [0, {total_frames}); got [{shots[0][0]}, {shots[-1][1]}) from {path}"
        )
    for (_, prev_end), (next_start, _) in zip(shots, shots[1:]):
        if prev_end != next_start:
            raise ValueError(f"Gap/overlap between shots at frame {prev_end}/{next_start} in {path}")
    return shots


def _decode_frame_exclude_mask(
    frame_entry: Any,
    full_meta_w: int,
    full_meta_h: int,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    exclude = np.zeros((crop_h, crop_w), dtype=bool)
    for raw_obj in extract_objects(frame_entry):
        raw_sam2 = raw_obj.get("sam2")
        raw_sam2 = raw_sam2 if isinstance(raw_sam2, dict) else {}
        segmentation = (
            raw_obj.get("segmentation")
            or raw_obj.get("mask")
            or raw_sam2.get("segmentation")
            or raw_sam2.get("mask")
        )
        if segmentation is None:
            continue
        source_w, source_h = source_image_size(raw_obj, full_meta_w, full_meta_h)
        mask = decode_mask(segmentation, source_h, source_w)
        if mask is None:
            continue
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        u = xs.astype(np.float64) * (float(full_meta_w) / float(source_w))
        v = ys.astype(np.float64) * (float(full_meta_h) / float(source_h))
        inside = (
            (u >= float(crop_x0))
            & (u < float(crop_x0 + crop_w))
            & (v >= float(crop_y0))
            & (v < float(crop_y0 + crop_h))
        )
        if not np.any(inside):
            continue
        ui = np.clip(np.round(u[inside] - crop_x0).astype(np.int64), 0, crop_w - 1)
        vi = np.clip(np.round(v[inside] - crop_y0).astype(np.int64), 0, crop_h - 1)
        exclude[vi, ui] = True
    return exclude


def compute_background_disparity_series(
    depth: np.ndarray,
    frames: List[Any],
    full_meta_w: int,
    full_meta_h: int,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    min_bg_px: int,
    shots: List[Tuple[int, int]],
    max_occlusion_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame median raw disparity of a *fixed-per-shot* set of background
    pixels: those never covered by any tracked object's mask at any frame
    within that shot. DepthCrafter's raw output drifts in absolute scale over
    a few seconds even where the real depth cannot change (verified on a
    provably-static background patch -- see
    docs/bundle-shared/D-004-anchor-z-accuracy.md). This series is that drift
    signal, used by
    build_background_drift_correction to remove it from tracked objects'
    sampled disparity.

    Fixing the pixel set per shot (rather than "whatever's left after this
    frame's mask" each frame) matters: a naive per-frame background sample
    mixes in genuine scene-depth changes whenever the excluded area's size or
    position shifts (e.g. a large object sweeping across a frame with real
    background depth structure -- near grass vs. far treeline -- exposes a
    different depth mix each frame).

    Returns (background_disparity, shot_safe_mask). shot_safe_mask is a
    per-frame bool: False for every frame in a shot where some frame's
    tracked-object occlusion exceeded max_occlusion_frac of the crop area.
    Measured evidence for this gate (mask-based occlusion, not bbox area):
    on FINNAL_HUMAN (where the correction measurably improved D-004's R^2
    metric) peak occlusion never exceeded 5.4%; on FINNAL_TRAIN (where the
    same correction measurably worsened it) peak occlusion reached 20.2%.
    This is a correlated proxy, not a proven
    causal mechanism -- treat it as a safety gate, not a certainty. Frames in
    an unsafe shot still get a background_disparity value (for visibility in
    the sidecar) but build_background_drift_correction must zero their
    correction.

    NaN where a shot's stable set (or, as a per-frame fallback when that set
    is too small, that frame's own non-excluded pixels) has fewer than
    min_bg_px pixels; caller fills those from the nearest earlier trusted
    frame.
    """
    n = depth.shape[0]
    result = np.full(n, np.nan, dtype=np.float64)
    shot_safe = np.ones(n, dtype=bool)
    shot_ranges = shots if shots else [(0, n)]
    min_bg_px = max(1, int(min_bg_px))
    crop_area = float(crop_w * crop_h)
    for start, end in shot_ranges:
        end = min(end, n, len(frames))
        if start >= end:
            continue
        frame_exclude_masks: Dict[int, np.ndarray] = {}
        union_exclude = np.zeros((crop_h, crop_w), dtype=bool)
        max_occlusion = 0.0
        for frame_idx in range(start, end):
            mask = _decode_frame_exclude_mask(
                frames[frame_idx], full_meta_w, full_meta_h, crop_x0, crop_y0, crop_w, crop_h
            )
            frame_exclude_masks[frame_idx] = mask
            union_exclude |= mask
            max_occlusion = max(max_occlusion, np.count_nonzero(mask) / crop_area)
        is_safe = max_occlusion <= float(max_occlusion_frac)
        if not is_safe:
            shot_safe[start:end] = False
            print(
                f"Background drift correction: shot [{start},{end}) peak occlusion "
                f"{100 * max_occlusion:.1f}% > {100 * float(max_occlusion_frac):.1f}% "
                "threshold -- correction disabled for this shot."
            )
        stable_background = ~union_exclude
        stable_count = int(np.count_nonzero(stable_background))
        for frame_idx in range(start, end):
            if stable_count >= min_bg_px:
                background = depth[frame_idx][stable_background]
            else:
                background = depth[frame_idx][~frame_exclude_masks[frame_idx]]
            valid = background[np.isfinite(background) & (background > 0)]
            if valid.size >= min_bg_px:
                result[frame_idx] = float(np.median(valid))
    return result, shot_safe


def build_background_drift_correction(
    background_disparity: np.ndarray,
    shot_safe: np.ndarray,
    shots: List[Tuple[int, int]],
    num_frames: int,
) -> np.ndarray:
    """Turns a per-frame background disparity series into a per-frame
    correction (raw disparity units) to subtract from tracked objects' sampled
    disparity. Fills NaN frames (too few background pixels) by holding the
    nearest earlier trusted value (or the first trusted value, for a leading
    gap). Baselined per shot -- since a real camera-distance jump at a shot
    cut is legitimate (existing convention, see EMA/hold reset at
    shot_start_frames), the correction should only flatten drift *within* a
    shot, not carry a baseline across cuts.

    Frames where shot_safe is False get zero correction (see
    compute_background_disparity_series's peak-occlusion gate) -- their
    background_disparity isn't a trustworthy single-region reference, so no
    correction is safer than a wrong one.
    """
    filled = background_disparity.copy()
    last_valid: Optional[float] = None
    for i in range(filled.shape[0]):
        if np.isfinite(filled[i]):
            last_valid = float(filled[i])
        elif last_valid is not None:
            filled[i] = last_valid
    first_valid: Optional[float] = None
    for i in range(filled.shape[0]):
        if np.isfinite(filled[i]):
            first_valid = float(filled[i])
            break
    if first_valid is not None:
        for i in range(filled.shape[0]):
            if np.isfinite(filled[i]):
                break
            filled[i] = first_valid

    correction = np.zeros(num_frames, dtype=np.float64)
    shot_ranges = shots if shots else [(0, num_frames)]
    for start, end in shot_ranges:
        end = min(end, num_frames)
        if start >= end:
            continue
        segment = filled[start:end]
        valid_segment = segment[np.isfinite(segment)]
        if valid_segment.size == 0:
            continue
        baseline = float(np.median(valid_segment))
        correction[start:end] = segment - baseline
    correction[~shot_safe[:num_frames]] = 0.0
    return correction


def compute_fixed_box_disparity_series(
    depth: np.ndarray,
    box_crop_px: Tuple[int, int, int, int],
) -> np.ndarray:
    """Per-frame median raw disparity within a fixed rectangular box, given
    in crop-space pixels (x0, y0, x1, y1) of the depth plane -- NOT the eye
    pixels that tracked objects' bboxX/bboxY use in the bundle (the two
    differ by eye_w/crop_w, eye_h/crop_h). `depth` must already be the
    crop-applied array (post the `depth[:, crop_y0:crop_y0+crop_h,
    crop_x0:crop_x0+crop_w]` slice in build_bundle()), so no further scaling
    is needed: crop-space and this array's indices are 1:1.
    """
    n = depth.shape[0]
    x0, y0, x1, y1 = box_crop_px
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(depth.shape[2], x1)
    y1 = min(depth.shape[1], y1)
    result = np.full(n, np.nan, dtype=np.float64)
    if x1 <= x0 or y1 <= y0:
        return result
    patch = depth[:, y0:y1, x0:x1].astype(np.float64)
    valid = np.isfinite(patch)
    counts = valid.reshape(n, -1).sum(axis=1)
    medians = np.nanmedian(np.where(valid, patch, np.nan).reshape(n, -1), axis=1)
    medians[counts == 0] = np.nan
    result[:] = medians
    return result


def compute_shot_scale_calibration(
    near_disparity: np.ndarray,
    far_disparity: np.ndarray,
    z_near: float,
    z_far: float,
    shots: List[Tuple[int, int]],
    num_frames: int,
) -> List[Dict[str, Any]]:
    """Solves one (a, b) pair per shot for disparity = a/Z + b (Z in
    meters), from the shot-median disparity of two fixed static background
    patches assumed to sit at real-world distances z_near < z_far. See
    docs/bundle-shared/D-004-anchor-z-accuracy.md.

    Only b is trustworthy at metric scale: algebraically, with r = z_far /
    z_near, b = far - (near - far) / (r - 1) depends solely on the ratio r,
    not on the absolute value of either z_near or z_far. a carries an
    unknown absolute scale factor from wherever z_near's assumed value came
    from -- only the ratio 1 / (disparity - b) is meaningful for relative
    placement (this is what Unity's ResolvePopoutFraction needs), not a by
    itself.

    Validated on FINNAL_HUMAN against two independent real-distance
    references (person via keypoints3d bisection, ball via known handball
    diameter): coefficient of variation of the predicted/true distance
    ratio came out 0.086-0.098, versus 0.33-0.36 for the shipped
    background_drift_correction's single-reference drift removal alone. A
    per-frame time-varying a(t), b(t) fit was also tested and only reached
    0.089-0.092 -- close enough to this per-shot constant that per-frame
    isn't worth the extra complexity or the risk of any single frame's
    patch reading going bad.

    A shot with too few finite background samples gets a=None, b=None
    entries (not silently dropped), so a missing calibration is visible in
    the output rather than a gap Unity has to notice on its own.
    """
    if not (z_far > z_near > 0):
        raise ValueError(f"require 0 < z_near < z_far, got z_near={z_near}, z_far={z_far}")
    denom = 1.0 / z_near - 1.0 / z_far
    shot_ranges = shots if shots else [(0, num_frames)]
    results: List[Dict[str, Any]] = []
    for start, end in shot_ranges:
        end = min(end, num_frames)
        if start >= end:
            continue
        near_seg = near_disparity[start:end]
        far_seg = far_disparity[start:end]
        near_valid = near_seg[np.isfinite(near_seg)]
        far_valid = far_seg[np.isfinite(far_seg)]
        entry: Dict[str, Any] = {"shotStart": int(start), "shotEnd": int(end)}
        if near_valid.size == 0 or far_valid.size == 0:
            entry.update({"a": None, "b": None})
            results.append(entry)
            continue
        near_med = float(np.median(near_valid))
        far_med = float(np.median(far_valid))
        a = (near_med - far_med) / denom
        b = far_med - a / z_far
        entry.update(
            {
                "a": float(a),
                "b": float(b),
                "zNearAssumedM": float(z_near),
                "zFarAssumedM": float(z_far),
                "nearDisparityMedian": near_med,
                "farDisparityMedian": far_med,
            }
        )
        results.append(entry)
    return results


def smooth_value(prev: Optional[float], value: Optional[float], alpha: float) -> float:
    if value is None:
        if prev is None:
            return 0.0
        return prev
    if prev is None:
        return value
    return alpha * prev + (1.0 - alpha) * value


def evaluate_placement_observation(
    *,
    bbox: Tuple[float, float, float, float],
    anchor_u: float,
    anchor_v: float,
    anchor_z_raw: Optional[float],
    anchor_source: str,
    depth_stats: Dict[str, float],
    state: TrackState,
    w_eye: int,
    height: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    area = max(0.0, float(bbox_w) * float(bbox_h))
    reasons: List[str] = []
    confidence = 1.0

    edge_margin = max(0.0, float(args.placement_conf_edge_margin_px))
    edge_touch = (
        bbox_x <= edge_margin
        or bbox_y <= edge_margin
        or bbox_x + bbox_w >= float(w_eye) - edge_margin
        or bbox_y + bbox_h >= float(height) - edge_margin
    )
    if edge_touch:
        confidence -= 0.35
        reasons.append("bbox_touches_frame_edge")
        if state.placement_low_streak > 0:
            confidence -= 0.35
            reasons.append("continuing_frameout_tail")

    prev_area = state.bbox_area
    area_ratio: Optional[float] = None
    if prev_area is not None and prev_area > 0.0:
        area_ratio = area / prev_area
        if area_ratio < float(args.placement_conf_area_shrink_ratio):
            confidence -= 0.25
            reasons.append("bbox_area_shrunk")

    if anchor_source == "bbox_center_depth":
        confidence -= 0.20
        reasons.append("bbox_center_anchor_fallback")

    valid_count = int(round(depth_stats.get("valid_count", 0.0)))
    if anchor_z_raw is None or not math.isfinite(float(anchor_z_raw)) or float(anchor_z_raw) <= 0.0:
        confidence -= 0.60
        reasons.append("no_valid_anchor_depth")
    elif anchor_source not in ("animal_camera_root", "person_mask_median_depth") and valid_count < max(
        1, int(args.depth_gate_min_valid)
    ):
        confidence -= 0.20
        reasons.append("low_anchor_depth_valid_count")

    iqr = float(depth_stats.get("iqr", 0.0))
    mad = float(depth_stats.get("mad", 0.0))
    if (
        # person_mask_median_depth spreads over the whole body on purpose, so a wide
        # iqr/mad is the expected shape, not a straddled edge. This gate exists to
        # catch a *point* sample sitting on a depth discontinuity (D-004); applying
        # it to the mask median would hold the anchor on ~1% of frames for a reason
        # that no longer means anything.
        anchor_source != "person_mask_median_depth"
        and float(args.depth_gate_iqr) > 0.0
        and float(args.depth_gate_mad) > 0.0
        and iqr > float(args.depth_gate_iqr)
        and mad > float(args.depth_gate_mad)
    ):
        confidence -= 0.25
        reasons.append("unstable_anchor_depth_patch")

    if (
        state.last_good_anchor_u is not None
        and state.last_good_anchor_v is not None
        and math.isfinite(float(anchor_u))
        and math.isfinite(float(anchor_v))
    ):
        du = float(anchor_u) - float(state.last_good_anchor_u)
        dv = float(anchor_v) - float(state.last_good_anchor_v)
        anchor_jump_px = math.sqrt(du * du + dv * dv)
        if anchor_jump_px > float(args.placement_conf_anchor_jump_px):
            confidence -= 0.20
            reasons.append("anchor_uv_jump")
    else:
        anchor_jump_px = None

    if (
        # placement_conf_depth_jump is a step in normalized disparity; the
        # animal root is AniMer camera-space Z on its own scale, already
        # median-filtered and smoothed by project_animal_keypoints_to_camera.py.
        anchor_source != "animal_camera_root"
        and state.last_good_anchor_z is not None
        and anchor_z_raw is not None
        and math.isfinite(float(anchor_z_raw))
        and abs(float(anchor_z_raw) - float(state.last_good_anchor_z)) > float(args.placement_conf_depth_jump)
    ):
        confidence -= 0.20
        reasons.append("anchor_depth_jump")

    confidence = max(0.0, min(1.0, confidence))
    threshold = float(args.placement_conf_hold_threshold)
    status = "high" if confidence >= threshold else "low"
    return {
        "confidence": confidence,
        "status": status,
        "reasons": reasons,
        "area": area,
        "areaRatioFromPrevious": area_ratio,
        "anchorJumpPx": anchor_jump_px,
        "edgeTouch": edge_touch,
        "depthStats": {
            "validCount": valid_count,
            "median": depth_stats.get("median"),
            "iqr": depth_stats.get("iqr"),
            "mad": depth_stats.get("mad"),
            "p10": depth_stats.get("p10"),
            "p90": depth_stats.get("p90"),
        },
    }


def frame_shot_indices(shots: Sequence[Tuple[int, int]], num_frames: int) -> np.ndarray:
    """shot index per frame from the [start, end) list load_shots_for_bundle
    validated (contiguous, covering [0, num_frames))."""
    out = np.zeros(num_frames, dtype=np.int64)
    for idx, (start, end) in enumerate(shots):
        out[max(0, start) : min(num_frames, end)] = idx
    return out


def reset_track_state_if_discontinuous(
    state: TrackState, frame_idx: int, frame_shot: np.ndarray
) -> Optional[str]:
    """Cold-start the track unless it was seen on the previous frame of the
    same shot. Returns why it was reset ("cold_start", "gap", "shot") or None.

    A hard cut makes a real camera-distance jump legitimate, not sensor
    noise -- carrying pre-cut EMA/last-good-anchor state across it would
    smear or gate-hold a stale placement into the new shot. The same holds
    for a track that re-enters after a gap: flow-audit B1 measured train
    track 3 coming back at raw z 0.388 but used z 0.759 (0.8*stale + 0.2*raw)
    and a hold gate comparing against a last-good from 400 frames earlier.
    B2: a track absent on the cut frame itself and back k frames later must
    also not keep pre-cut state, hence the shot-index comparison rather than
    a "frame_idx is a shot start" test.
    """
    last = state.last_seen_frame
    if last is None:
        reason = "cold_start"
    elif last != frame_idx - 1:
        reason = "gap"
    elif int(frame_shot[last]) != int(frame_shot[frame_idx]):
        reason = "shot"
    else:
        return None
    state.reset()
    return reason


def resolve_track_anchor(
    *,
    state: TrackState,
    placement_eval: Dict[str, Any],
    anchor_u: float,
    anchor_v: float,
    anchor_z_raw: Optional[float],
    anchor_source: str,
    frame_idx: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Turn one raw observation into the anchor written to meta.bin (hold,
    EMA) and advance the track state. `skipped` is True when there is nothing
    to write: no raw depth, no EMA value and no last-good anchor (flow-audit
    B7 -- the old code wrote anchor_z=0.0 and let the EMA ramp up from it)."""
    placement_held = False
    hold_source = None
    anchor_z: Optional[float] = None
    if (
        placement_eval["status"] == "low"
        and state.last_good_anchor_u is not None
        and state.last_good_anchor_v is not None
        and state.last_good_anchor_z is not None
    ):
        anchor_u = float(state.last_good_anchor_u)
        anchor_v = float(state.last_good_anchor_v)
        anchor_z = float(state.last_good_anchor_z)
        anchor_source = "held_previous_high_conf"
        placement_held = True
        hold_source = state.last_good_anchor_source
    elif anchor_z_raw is None and state.anchor_z is None:
        # Cold start without depth: state.anchor_z stays None so the next
        # frame with a real sample starts the EMA from that sample.
        if state.last_good_anchor_z is not None:
            anchor_u = float(state.last_good_anchor_u)
            anchor_v = float(state.last_good_anchor_v)
            anchor_z = float(state.last_good_anchor_z)
            anchor_source = "held_previous_high_conf"
            placement_held = True
            hold_source = state.last_good_anchor_source
        else:
            state.bbox_area = float(placement_eval["area"])
            state.placement_low_streak += 1
            state.last_seen_frame = frame_idx
            return {
                "skipped": True,
                "anchor_u": float(anchor_u),
                "anchor_v": float(anchor_v),
                "anchor_z": None,
                "anchor_source": anchor_source,
                "placement_held": False,
                "hold_source": None,
            }
    else:
        if anchor_source == "animal_camera_root":
            anchor_z = float(anchor_z_raw) if anchor_z_raw is not None else smooth_value(state.anchor_z, None, args.ema_alpha)
        else:
            anchor_z = smooth_value(state.anchor_z, anchor_z_raw, args.ema_alpha)
        state.anchor_z = anchor_z
        if placement_eval["status"] == "high":
            state.last_good_anchor_u = float(anchor_u)
            state.last_good_anchor_v = float(anchor_v)
            state.last_good_anchor_z = float(anchor_z)
            state.last_good_anchor_source = str(anchor_source)
    state.bbox_area = float(placement_eval["area"])
    if placement_eval["status"] == "low":
        state.placement_low_streak += 1
    else:
        state.placement_low_streak = 0
    state.last_seen_frame = frame_idx
    return {
        "skipped": False,
        "anchor_u": float(anchor_u),
        "anchor_v": float(anchor_v),
        "anchor_z": float(anchor_z),
        "anchor_source": anchor_source,
        "placement_held": placement_held,
        "hold_source": hold_source,
    }


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


def camera_xyz_from_uv_depth(
    u: float,
    v: float,
    z: float,
    w_eye: int,
    height: int,
    fovx_deg: float,
) -> np.ndarray:
    if z <= 0.0 or w_eye <= 0 or height <= 0:
        return np.zeros(3, dtype=np.float32)
    fovx_rad = math.radians(fovx_deg)
    fx = 1.0 / math.tan(fovx_rad / 2.0)
    fy = fx * (float(w_eye) / float(height))
    x_ndc = (float(u) / float(w_eye) - 0.5) * 2.0
    y_ndc = (0.5 - float(v) / float(height)) * 2.0
    return np.asarray([x_ndc * z / fx, y_ndc * z / fy, z], dtype=np.float32)


def pose_source_axes(pose: Any) -> Optional[str]:
    """AXES_Y_DOWN / AXES_Y_UP_LEGACY from pose.cameraAxes or
    pose.coordinateSystem (whichever names the axes), None when unlabelled."""
    if not isinstance(pose, dict):
        return None
    for key in ("cameraAxes", "coordinateSystem"):
        text = str(pose.get(key) or "").lower()
        if "y_down" in text:
            return AXES_Y_DOWN
        if "y_up" in text:
            return AXES_Y_UP_LEGACY
    return None


def pose_is_camera_absolute(pose: Any) -> bool:
    """Does pose.keypoints3d carry an absolute camera-space root (the animal
    lift in project_animal_keypoints_to_camera.py) rather than engine-local
    joints? Read from coordinateSystem, or from the lift's own keys when a
    sidecar uses coordinateSystem for the axis label instead."""
    if not isinstance(pose, dict):
        return False
    if "camera_xyz_absolute" in str(pose.get("coordinateSystem") or "").lower():
        return True
    bbox3d = pose.get("bbox3d")
    if isinstance(bbox3d, dict) and "absolute" in str(bbox3d.get("coord_system") or "").lower():
        return True
    return pose.get("skeletonRoot3d") is not None and pose.get("rootSource") is not None


def resolve_pose_flip_y(args: argparse.Namespace, source_axes: Optional[str]) -> bool:
    """Whether pose.keypoints3d Y is negated before it becomes meta.bin joints
    (BUNDLE_JOINT_AXES). In auto mode every label leads to a flip -- y_down
    is the OpenCV frame the engines emit, and the legacy y_up label was put
    on those same engine-native offsets -- but the reason is recorded per
    label rather than assumed (see the AXES_* comment)."""
    override = str(args.pose_keypoints3d_flip_y).lower()
    if override in ("0", "1"):
        return override == "1"
    return source_axes in (AXES_Y_DOWN, AXES_Y_UP_LEGACY, None)


def camera_xyz_to_pixel_ydown(
    xyz: Sequence[float], width: int, height: int, fovx_deg: float
) -> Optional[Tuple[float, float]]:
    """Project an OpenCV-frame (+y down) camera point onto a width x height
    canvas with horizontal FOV fovx_deg -- the inverse of the lift in
    project_animal_keypoints_to_camera.py. Same formula as
    scripts/bundle_common/geometry.py camera_xyz_to_pixel; kept local so the
    builder imports nothing that may not exist yet."""
    x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)) or z <= 0.0:
        return None
    if width <= 0 or height <= 0:
        return None
    tan_x = math.tan(math.radians(float(fovx_deg)) * 0.5)
    tan_y = tan_x * (float(height) / float(width))
    half_w = float(width) * 0.5
    half_h = float(height) * 0.5
    return half_w + (x / (tan_x * z)) * half_w, half_h + (y / (tan_y * z)) * half_h


def pose_joint_scale(args: argparse.Namespace, coordinate_system: Optional[str], engine: Optional[str]) -> float:
    text = f"{coordinate_system or ''} {engine or ''}".lower()
    if "metrabs" in text:
        return float(args.metrabs_joint_scale)
    if "animer" in text or "smal" in text:
        return float(args.animer_joint_scale)
    return 1.0


def compute_pose_keypoints3d_for_bundle(
    source_joints3d: np.ndarray,
    source_valid: np.ndarray,
    root_indices: Sequence[int],
    anchor_xyz: np.ndarray,
    joints_space: str,
    joint_scale: float,
    flip_y: bool,
    prev_joints_rel: Optional[np.ndarray],
    ema_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joints = source_joints3d.astype(np.float32).copy() * float(joint_scale)
    if flip_y:
        joints[:, 1] *= -1.0
    valid = source_valid.astype(bool).copy()
    valid &= np.all(np.isfinite(joints), axis=1)
    root = compute_root(joints, valid, root_indices)
    joints_rel_raw = joints - root.reshape(1, 3)

    if prev_joints_rel is not None and prev_joints_rel.shape == joints_rel_raw.shape:
        for idx in range(joints_rel_raw.shape[0]):
            if not valid[idx]:
                joints_rel_raw[idx] = prev_joints_rel[idx]
        joints_rel = float(ema_alpha) * prev_joints_rel + (1.0 - float(ema_alpha)) * joints_rel_raw
    else:
        joints_rel = joints_rel_raw

    if joints_space == "camera_xyz_absolute":
        joints_out = joints_rel + anchor_xyz.reshape(1, 3)
    else:
        joints_out = joints_rel
    return joints_out.astype(np.float32), joints_rel.astype(np.float32), valid, root.astype(np.float32)


def _compute_joints3d_and_root(
    keypoints: List[Tuple[float, float, float]],
    depth_frame: np.ndarray,
    w_eye: int,
    height: int,
    fovx_deg: float,
    sample_k: int,
    conf_th: float,
    meta_w: int,
    meta_h: int,
    crop_x0: int,
    crop_y0: int,
    crop_w: int,
    crop_h: int,
    root_indices: Sequence[int],
    prev_joints_abs: Optional[np.ndarray],
    depth_gate_min_valid: int,
    depth_gate_prev_band: float,
    depth_gate_prev_min_valid: int,
    depth_gate_prev_min_frac: float,
    depth_gate_use_anchor_fallback: bool,
    anchor_z_fallback: Optional[float],
    depth_gate_iqr: float,
    depth_gate_range: float,
    depth_gate_mad: float,
    depth_gate_jump: float,
    depth_gate_conf_margin: float,
    debug_depth: bool,
    frame_idx: int,
    track_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    kp_count = len(keypoints)
    joints3d = np.zeros((kp_count, 3), dtype=np.float32)
    valid = np.zeros(kp_count, dtype=bool)
    fovx_rad = math.radians(fovx_deg)
    fx = 1.0 / math.tan(fovx_rad / 2.0)
    fy = fx * (float(w_eye) / float(height))

    for idx, (u, v, conf) in enumerate(keypoints):
        conf_n = normalize_conf(conf)
        if conf_n < conf_th:
            continue
        # prev_joints_abs stores this function's own previous-frame output, so
        # its z is already in the flipped (bundle camera-Z, larger=farther)
        # domain -- see state.joints_abs assignment at the call site. Keep
        # prev_z in that domain for the "reuse previous" branch below, but
        # feed sample_depth_robust's internal prev-frame banding a raw-domain
        # value (disparity_to_camera_z is its own inverse for non-clamped
        # inputs) since that gate compares against freshly re-sampled raw
        # depth_frame values.
        prev_z = None
        if (
            prev_joints_abs is not None
            and idx < prev_joints_abs.shape[0]
            and prev_joints_abs.shape[1] >= 3
            and math.isfinite(float(prev_joints_abs[idx, 2]))
            and float(prev_joints_abs[idx, 2]) > 0.0
        ):
            prev_z = float(prev_joints_abs[idx, 2])
        prev_z_raw = disparity_to_camera_z(prev_z) if prev_z is not None else None

        z_new, depth_stats, depth_reason = sample_depth_robust(
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
            min_valid=depth_gate_min_valid,
            prev_band=depth_gate_prev_band,
            prev_min_valid=depth_gate_prev_min_valid,
            prev_min_frac=depth_gate_prev_min_frac,
            tau_iqr=depth_gate_iqr,
            tau_range=depth_gate_range,
            tau_mad=depth_gate_mad,
            tau_jump=depth_gate_jump,
            conf_n=conf_n,
            conf_th=conf_th,
            conf_margin=depth_gate_conf_margin,
            prev_z=prev_z_raw,
        )
        used_prev = False
        used_anchor = False
        if z_new is None:
            if prev_z is not None:
                # Already flipped-domain (see comment above); reuse as-is.
                z = prev_z
                used_prev = True
            elif (
                depth_gate_use_anchor_fallback
                and anchor_z_fallback is not None
                and math.isfinite(anchor_z_fallback)
                and anchor_z_fallback > 0.0
            ):
                # anchor_z_fallback is the caller's already-fixed anchor_z;
                # already flipped-domain, do not flip again.
                z = float(anchor_z_fallback)
                used_anchor = True
            else:
                z = 0.0
        else:
            # z_new is raw depth_frame domain (0.0=far, 1.0=near); flip to
            # the bundle's camera-Z convention (larger=farther).
            z = disparity_to_camera_z(z_new)
        if z <= 0.0:
            joints3d[idx] = (0.0, 0.0, 0.0)
            if debug_depth:
                print(
                    "[depth_debug] "
                    f"frame={frame_idx} trackId={track_id} kp={idx} "
                    f"u={u:.2f} v={v:.2f} conf={conf_n:.3f} "
                    f"valid_count={int(round(depth_stats.get('valid_count', 0.0)))} "
                    f"median={_fmt_float(depth_stats.get('median'))} "
                    f"p10={_fmt_float(depth_stats.get('p10'))} p25={_fmt_float(depth_stats.get('p25'))} "
                    f"p75={_fmt_float(depth_stats.get('p75'))} p90={_fmt_float(depth_stats.get('p90'))} "
                    f"iqr={_fmt_float(depth_stats.get('iqr'))} "
                    f"band={depth_gate_prev_band:.4f} filtN={int(round(depth_stats.get('filtered_count', 0.0)))} "
                    f"filtFrac={depth_stats.get('filtered_frac', 0.0):.2f} "
                    f"mad={_fmt_float(depth_stats.get('mad'))} z_new={_fmt_float(z_new)} "
                    f"z_used={_fmt_float(z)} prev_z={_fmt_float(prev_z)} "
                    f"z_new_reason={depth_reason or 'none'} "
                    f"z_used_reason={'use_anchor_z' if used_anchor else ('use_prev_z' if used_prev else 'use_zero')}"
                )
            continue
        x_ndc = (u / float(w_eye) - 0.5) * 2.0
        y_ndc = (0.5 - v / float(height)) * 2.0
        X = x_ndc * z / fx
        Y = y_ndc * z / fy
        joints3d[idx] = (X, Y, z)
        valid[idx] = True
        if debug_depth:
            used_reason = "use_anchor_z" if used_anchor else ("use_prev_z" if used_prev else "use_z_new")
            print(
                "[depth_debug] "
                f"frame={frame_idx} trackId={track_id} kp={idx} "
                f"u={u:.2f} v={v:.2f} conf={conf_n:.3f} "
                f"valid_count={int(round(depth_stats.get('valid_count', 0.0)))} "
                f"median={_fmt_float(depth_stats.get('median'))} "
                f"p10={_fmt_float(depth_stats.get('p10'))} p25={_fmt_float(depth_stats.get('p25'))} "
                f"p75={_fmt_float(depth_stats.get('p75'))} p90={_fmt_float(depth_stats.get('p90'))} "
                f"iqr={_fmt_float(depth_stats.get('iqr'))} "
                f"band={depth_gate_prev_band:.4f} filtN={int(round(depth_stats.get('filtered_count', 0.0)))} "
                f"filtFrac={depth_stats.get('filtered_frac', 0.0):.2f} "
                f"mad={_fmt_float(depth_stats.get('mad'))} z_new={_fmt_float(z_new)} "
                f"z_used={_fmt_float(z)} prev_z={_fmt_float(prev_z)} "
                f"z_new_reason={depth_reason or 'ok'} z_used_reason={used_reason}"
            )

    root = compute_root(joints3d, valid, root_indices)
    return joints3d, valid, root


def invert_camera_xyz_to_uv(
    joints_xyz: np.ndarray, w_eye: int, height: int, fovx_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    fovx_rad = math.radians(fovx_deg)
    fx = 1.0 / math.tan(fovx_rad / 2.0)
    fy = fx * (float(w_eye) / float(height))
    uv = np.zeros((joints_xyz.shape[0], 2), dtype=np.float32)
    valid = np.zeros(joints_xyz.shape[0], dtype=bool)
    for idx in range(joints_xyz.shape[0]):
        X, Y, z = float(joints_xyz[idx, 0]), float(joints_xyz[idx, 1]), float(joints_xyz[idx, 2])
        if not (math.isfinite(X) and math.isfinite(Y) and math.isfinite(z)) or z <= 0.0:
            continue
        x_ndc = X * fx / z
        y_ndc = Y * fy / z
        u = (x_ndc * 0.5 + 0.5) * float(w_eye)
        v = (0.5 - y_ndc * 0.5) * float(height)
        uv[idx] = (u, v)
        valid[idx] = True
    return uv, valid


def animal_camera_root_anchor(
    obj: Dict[str, Any],
    args: argparse.Namespace,
    w_eye: int,
    height: int,
    *,
    full_meta_w: Optional[int] = None,
    full_meta_h: Optional[int] = None,
    crop_x0: int = 0,
    crop_y0: int = 0,
    crop_w: Optional[int] = None,
    crop_h: Optional[int] = None,
) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """Return an AniMer camera-space root anchor as (u, v, z, xyz), with u/v
    in eye pixels and xyz in BUNDLE_JOINT_AXES.

    The root was lifted on the SOURCE canvas (obj source_w x source_h, the
    work video) by project_animal_keypoints_to_camera.py, so it is projected
    back onto that canvas with the inverse of that lift and then taken
    source -> depth plane -> crop -> eye like every other source pixel. When
    the depth-plane/crop arguments are omitted the source canvas is taken to
    be the eye (tests, or a full-frame eye). With --pose_keypoints3d_flip_y
    0/1 the pre-2026-09-17 path is reproduced exactly (whole-array flip,
    y-up re-projection straight onto the eye canvas).
    """
    if int(obj.get("category_id", TYPE_OTHER)) != TYPE_ANIMAL:
        return None
    if not bool(obj.get("source_is_camera_absolute", False)):
        coord_system = str(obj.get("source_coord_system") or "").lower()
        if "camera_xyz_absolute" not in coord_system:
            return None
    source_joints3d = obj.get("source_joints3d")
    if source_joints3d is None:
        return None
    try:
        joints = np.asarray(source_joints3d, dtype=np.float32).copy()
    except Exception:
        return None
    if joints.ndim != 2 or joints.shape[1] < 3:
        return None
    joints = joints[:, :3] * pose_joint_scale(
        args,
        obj.get("source_coord_system"),
        obj.get("source_engine"),
    )
    source_valid = obj.get("source_joints_valid")
    if source_valid is None:
        valid = np.ones((joints.shape[0],), dtype=bool)
    else:
        valid = np.asarray(source_valid, dtype=bool).reshape(-1)
        if valid.shape[0] != joints.shape[0]:
            valid = np.ones((joints.shape[0],), dtype=bool)
    valid &= np.all(np.isfinite(joints), axis=1)
    if not np.any(valid):
        return None

    override = str(args.pose_keypoints3d_flip_y).lower()
    if override in ("0", "1"):
        if override == "1":
            joints[:, 1] *= -1.0
        root = compute_root(joints, valid, obj.get("root_indices", []))
        if not np.all(np.isfinite(root)) or float(root[2]) <= 0.0:
            return None
        uv, uv_valid = invert_camera_xyz_to_uv(root.reshape(1, 3), w_eye, height, args.fovx_deg)
        if not bool(uv_valid[0]):
            return None
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        if u < 0.0 or u >= float(w_eye) or v < 0.0 or v >= float(height):
            return None
        return u, v, float(root[2]), root.astype(np.float32)

    root = compute_root(joints, valid, obj.get("root_indices", []))
    if not np.all(np.isfinite(root)) or float(root[2]) <= 0.0:
        return None
    source_axes = obj.get("source_axes")
    root_ydown = root.astype(np.float64).copy()
    if source_axes == AXES_Y_UP_LEGACY:
        # Legacy label: the root itself really is +y up (see AXES_* comment).
        root_ydown[1] *= -1.0
    source_w = int(obj.get("source_w") or w_eye)
    source_h = int(obj.get("source_h") or height)
    uv_source = camera_xyz_to_pixel_ydown(root_ydown, source_w, source_h, args.fovx_deg)
    if uv_source is None:
        return None
    u_src, v_src = uv_source
    if full_meta_w is None or full_meta_h is None or crop_w is None or crop_h is None:
        u, v = u_src, v_src
    else:
        u_meta, v_meta = source_to_metadata_point(u_src, v_src, source_w, source_h, full_meta_w, full_meta_h)
        uv_eye = meta_to_eye_point(u_meta, v_meta, crop_x0, crop_y0, crop_w, crop_h, w_eye, height)
        if uv_eye is None:
            return None
        u, v = uv_eye
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    if u < 0.0 or u >= float(w_eye) or v < 0.0 or v >= float(height):
        return None
    root_bundle = root_ydown.copy()
    root_bundle[1] *= -1.0  # BUNDLE_JOINT_AXES is +y up
    return float(u), float(v), float(root[2]), root_bundle.astype(np.float32)


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
    prev_joints_abs: Optional[np.ndarray],
    depth_gate_min_valid: int,
    depth_gate_prev_band: float,
    depth_gate_prev_min_valid: int,
    depth_gate_prev_min_frac: float,
    depth_gate_use_anchor_fallback: bool,
    anchor_z_fallback: Optional[float],
    depth_gate_iqr: float,
    depth_gate_range: float,
    depth_gate_mad: float,
    depth_gate_jump: float,
    depth_gate_conf_margin: float,
    debug_depth: bool,
    frame_idx: int,
    track_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kp_count = len(keypoints)
    if prev_joints_rel is not None and prev_joints_rel.shape[0] != kp_count:
        prev_joints_rel = None
    joints3d, valid, root = _compute_joints3d_and_root(
        keypoints=keypoints,
        depth_frame=depth_frame,
        w_eye=w_eye,
        height=height,
        fovx_deg=fovx_deg,
        sample_k=sample_k,
        conf_th=conf_th,
        meta_w=meta_w,
        meta_h=meta_h,
        crop_x0=crop_x0,
        crop_y0=crop_y0,
        crop_w=crop_w,
        crop_h=crop_h,
        root_indices=root_indices,
        prev_joints_abs=prev_joints_abs,
        depth_gate_min_valid=depth_gate_min_valid,
        depth_gate_prev_band=depth_gate_prev_band,
        depth_gate_prev_min_valid=depth_gate_prev_min_valid,
        depth_gate_prev_min_frac=depth_gate_prev_min_frac,
        depth_gate_use_anchor_fallback=depth_gate_use_anchor_fallback,
        anchor_z_fallback=anchor_z_fallback,
        depth_gate_iqr=depth_gate_iqr,
        depth_gate_range=depth_gate_range,
        depth_gate_mad=depth_gate_mad,
        depth_gate_jump=depth_gate_jump,
        depth_gate_conf_margin=depth_gate_conf_margin,
        debug_depth=debug_depth,
        frame_idx=frame_idx,
        track_id=track_id,
    )
    joints_rel_raw = joints3d - root

    if prev_joints_rel is not None:
        for idx in range(kp_count):
            if not valid[idx]:
                joints_rel_raw[idx] = prev_joints_rel[idx]
        joints_rel = ema_alpha * prev_joints_rel + (1.0 - ema_alpha) * joints_rel_raw
    else:
        joints_rel = joints_rel_raw

    return joints_rel, joints3d, valid, root


def choose_compression(name: str) -> Tuple[int, Any, str]:
    if name == "none":
        return 0, lambda data: data, "none"
    if name == "lz4":
        if lz4f is not None:
            return 2, lambda data: lz4f.compress(data), "lz4"
        print(
            "Warning: --frame_compress lz4 requested but python-lz4 is not installed in "
            "this env; falling back to zlib (manifest frame_compress will say zlib)."
        )
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
        anchor_source = obj.get("anchor_source")
        cat_name = obj.get("category_name") or ""
        base = (
            f"[obj {obj_idx}] tid={obj.get('track_id')} "
            f"cat={obj.get('category_id')}({cat_name}) "
            f"bbox=({bbox_x:.1f},{bbox_y:.1f},{bbox_w:.1f},{bbox_h:.1f}) "
            f"anchor={anchor_str} z_raw={z_raw} z_ema={z_ema} z_q={z_q} "
            f"anchor_source={anchor_source} "
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
                if flags & FLAG_SKELETON:
                    kp_count = kp_counts.get(int(category_id), 0)
                    pos += kp_count * 3 * 2
                    pos += kp_count
                if flags & FLAG_SMPL:
                    if pos + 6 > len(payload):
                        break
                    _smpl_version, rot_count, beta_count = struct.unpack_from("<HHH", payload, pos)
                    pos += 6 + int(rot_count) * 9 * 4 + int(beta_count) * 4 + 3 * 4
                if flags & FLAG_SMAL:
                    if pos + 6 > len(payload):
                        break
                    _smal_version, rot_count, beta_count = struct.unpack_from("<HHH", payload, pos)
                    pos += 6 + int(rot_count) * 9 * 4 + int(beta_count) * 4 + 3 * 4
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

    depth_npz = load_depth_npz(args.depth_npz)
    depth = depth_npz.depth
    depth_disp_min, depth_disp_max = depth_npz.disp_min, depth_npz.disp_max
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

    shots = load_shots_for_bundle(args.shots_json, num_frames)
    frame_shot = frame_shot_indices(shots, num_frames)

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

    background_disparity_series = np.full(num_frames, np.nan, dtype=np.float64)
    background_drift_correction = np.zeros(num_frames, dtype=np.float64)
    if args.background_drift_correction:
        background_disparity_series, background_shot_safe = compute_background_disparity_series(
            depth,
            frames,
            full_meta_w,
            full_meta_h,
            crop_x0,
            crop_y0,
            crop_w,
            crop_h,
            args.background_drift_min_px,
            shots,
            args.background_drift_max_occlusion_frac,
        )
        background_drift_correction = build_background_drift_correction(
            background_disparity_series, background_shot_safe, shots, num_frames
        )
        print(
            "Background drift correction: "
            f"disparity median={np.nanmedian(background_disparity_series):.4f} "
            f"stdev={np.nanstd(background_disparity_series):.4f} "
            f"correction range=[{background_drift_correction.min():.4f}, "
            f"{background_drift_correction.max():.4f}]"
        )

    depth_scale_calibration: Optional[Dict[str, Any]] = None
    if args.depth_scale_calibration:
        if not (
            args.depth_scale_near_box
            and args.depth_scale_far_box
            and args.depth_scale_z_near is not None
            and args.depth_scale_z_far is not None
        ):
            raise ValueError(
                "--depth_scale_calibration 1 requires --depth_scale_near_box, "
                "--depth_scale_far_box, --depth_scale_z_near and --depth_scale_z_far."
            )

        def _parse_box(s: str) -> Tuple[int, int, int, int]:
            parts = [int(v.strip()) for v in s.split(",")]
            if len(parts) != 4:
                raise ValueError(f"expected 'x0,y0,x1,y1', got {s!r}")
            return parts[0], parts[1], parts[2], parts[3]

        near_box = _parse_box(args.depth_scale_near_box)
        far_box = _parse_box(args.depth_scale_far_box)
        near_series = compute_fixed_box_disparity_series(depth, near_box)
        far_series = compute_fixed_box_disparity_series(depth, far_box)
        shot_calibrations = compute_shot_scale_calibration(
            near_series,
            far_series,
            args.depth_scale_z_near,
            args.depth_scale_z_far,
            shots,
            num_frames,
        )
        depth_scale_calibration = {
            "schema": "master_project.depth_scale_calibration.v1",
            "meaning": (
                "Per-shot affine-disparity calibration solving disparity = "
                "a/Z + b (Z in meters) from two fixed static background "
                "reference patches at assumed real-world distances "
                "zNearAssumedM < zFarAssumedM. See "
                "docs/bundle-shared/D-004-anchor-z-accuracy.md. Only b is "
                "trustworthy at "
                "metric scale -- it depends solely on the ratio "
                "zFarAssumedM/zNearAssumedM, not on either patch's absolute "
                "distance. a carries an unknown absolute scale factor; only "
                "the ratio 1/(disparity - b) is meaningful for relative "
                "placement (matches Unity's ResolvePopoutFraction use), not "
                "a by itself. Apply BEFORE any popout/placement transform, "
                "not inside it. A shot entry with a=null/b=null means too "
                "few finite background samples were available in that shot "
                "-- do not substitute a neighboring shot's values."
            ),
            "near_box_crop_px": list(near_box),
            "far_box_crop_px": list(far_box),
            "shots": shot_calibrations,
        }
        for entry in shot_calibrations:
            if entry.get("a") is None:
                print(
                    f"Depth scale calibration: shot [{entry['shotStart']},"
                    f"{entry['shotEnd']}) -- insufficient background samples, "
                    "a/b left null."
                )
            else:
                print(
                    f"Depth scale calibration: shot [{entry['shotStart']},"
                    f"{entry['shotEnd']}) a={entry['a']:.4f} b={entry['b']:.4f}"
                )

    track_state: Dict[int, TrackState] = {}
    track_id_map: Dict[Any, int] = {}
    next_track_id = 1
    prev_objects: List[Dict[str, Any]] = []
    placement_observation_frames: List[Dict[str, Any]] = []
    crop_fraction: Optional[Dict[str, float]] = None
    track_reset_counts: Dict[str, int] = {"cold_start": 0, "gap": 0, "shot": 0}
    skipped_no_depth = 0
    pose_axes_seen: Dict[str, int] = {}
    pose_flip_y_seen: Dict[str, int] = {}

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
                    raw_source_w, raw_source_h = source_image_size(raw_obj, full_meta_w, full_meta_h)
                    is_full_sbs = source_is_full_sbs(raw_source_w, width, w_eye, args.left_eye_origin)
                    source_w, source_h = source_canvas_size(
                        raw_source_w,
                        raw_source_h,
                        width,
                        w_eye,
                        args.left_eye_origin,
                    )
                    if crop_fraction is None:
                        crop_fraction = check_crop_fraction(
                            crop_w=crop_w,
                            crop_h=crop_h,
                            full_meta_w=full_meta_w,
                            full_meta_h=full_meta_h,
                            w_eye=w_eye,
                            height=height,
                            source_w=source_w,
                            source_h=source_h,
                            align128=int(args.align128),
                        )
                    x = shift_left_eye_if_full_sbs(x, w_eye, width, is_full_sbs)
                    x, y, w, h = source_to_metadata_bbox(
                        (x, y, w, h),
                        source_w,
                        source_h,
                        full_meta_w,
                        full_meta_h,
                    )
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
                    # Keypoints are measured on the source canvas, so the pixel test
                    # uses source_w/source_h (flow-audit B5); the sidecar's explicit
                    # keypoints2dUnits wins over the magnitude heuristic when present.
                    kp2d_units = pose_keypoints2d_units(raw_obj)
                    if kp2d_units == "pixels":
                        kp_in_pixels = keypoints is not None
                    elif kp2d_units == "crop_normalized":
                        kp_in_pixels = False
                    else:
                        kp_in_pixels = keypoints_look_like_pixels(keypoints, kp_vis, source_w, source_h)
                    if keypoints is not None and not kp_in_pixels:
                        denorm_keypoints = (
                            denormalize_pose_keypoints2d_from_source_box(
                                raw_obj,
                                keypoints,
                                kp_vis,
                                source_w,
                                source_h,
                            )
                            if category_id == TYPE_PERSON
                            else None
                        )
                        if denorm_keypoints is not None:
                            keypoints = denorm_keypoints
                        else:
                            keypoints = None
                            kp_vis = []
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
                            u = shift_left_eye_if_full_sbs(u, w_eye, width, is_full_sbs)
                            u, v = source_to_metadata_point(
                                u,
                                v,
                                source_w,
                                source_h,
                                full_meta_w,
                                full_meta_h,
                            )
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
                    source_joints3d, source_joints_valid, source_coord_system, source_engine, source_keypoint_format = (
                        parse_pose_keypoints3d(raw_obj, expected_kp)
                    )
                    raw_pose = raw_obj.get("pose")
                    source_axes = pose_source_axes(raw_pose)
                    source_is_camera_absolute = pose_is_camera_absolute(raw_pose)
                    smpl_payload = parse_smpl_payload(raw_obj)
                    smal_payload = parse_smal_payload(raw_obj)
                    use_pose_keypoints3d = (
                        source_joints3d is not None
                        and expected_kp > 0
                        and args.joints_source in ("auto", "pose_keypoints3d")
                    )
                    if args.joints_source == "pose_keypoints3d" and expected_kp > 0 and not use_pose_keypoints3d:
                        ann_id = raw_obj.get("id", raw_obj.get("trackId", raw_obj.get("track_id", "unknown")))
                        print(
                            f"Warning: frame {frame_idx} object {ann_id} requested pose_keypoints3d "
                            "but pose.keypoints3d is missing; falling back to 2D keypoints + depth."
                        )
                    raw_track_id = None
                    for key in ["track_id", "trackId", "track", "instance_id", "instanceId", "object_id", "id"]:
                        if key in raw_obj:
                            raw_track_id = raw_obj[key]
                            break
                    track_id, next_track_id = normalize_track_id(raw_track_id, track_id_map, next_track_id)
                    raw_sam2 = raw_obj.get("sam2")
                    raw_sam2 = raw_sam2 if isinstance(raw_sam2, dict) else {}
                    segmentation = (
                        raw_obj.get("segmentation")
                        or raw_obj.get("mask")
                        or raw_sam2.get("segmentation")
                        or raw_sam2.get("mask")
                    )
                    has_pose3d_skeleton = use_pose_keypoints3d and source_joints3d is not None
                    has_2d_depth_skeleton = expected_kp > 0 and kp_given > 0 and keypoints is not None
                    has_skeleton = bool(expected_kp > 0 and (has_pose3d_skeleton or has_2d_depth_skeleton))
                    if keypoints is None:
                        kp_vis = []
                    objects.append(
                        {
                            "track_id": track_id,
                            "category_id": category_id,
                            "bbox": (x, y, w, h),
                            "source_w": source_w,
                            "source_h": source_h,
                            "keypoints": keypoints,
                            "kp_vis": kp_vis,
                            "kp_expected": expected_kp,
                            "root_indices": cat_spec.get("root_indices", []),
                            "anchor_indices": cat_spec.get("anchor_indices", []),
                            "has_skeleton": has_skeleton,
                            "segmentation": segmentation,
                            "source_joints3d": source_joints3d if use_pose_keypoints3d else None,
                            "source_joints_valid": source_joints_valid if use_pose_keypoints3d else None,
                            "source_coord_system": source_coord_system,
                            "source_axes": source_axes,
                            "source_is_camera_absolute": source_is_camera_absolute,
                            "source_engine": source_engine,
                            "source_keypoint_format": source_keypoint_format,
                            "is_full_sbs": is_full_sbs,
                            "smpl_payload": smpl_payload,
                            "smal_payload": smal_payload,
                        }
                    )

                objects, next_track_id = assign_track_ids(objects, prev_objects, next_track_id)
                prev_objects = objects

                # Object records go into `payload`; the count is prepended after the
                # loop because an object without any usable depth is skipped (B7).
                payload = bytearray()
                written_objects = 0
                depth_frame = depth[frame_idx]
                frame_placement_observations: List[Dict[str, Any]] = []

                for obj in objects:
                    track_id = int(obj["track_id"])
                    category_id = int(obj["category_id"])
                    bbox_x, bbox_y, bbox_w, bbox_h = obj["bbox"]
                    if obj.get("source_joints3d") is not None:
                        axes_key = str(obj.get("source_axes") or "unlabelled")
                        pose_axes_seen[axes_key] = pose_axes_seen.get(axes_key, 0) + 1

                    anchor_source = "depth_sample"
                    animal_root_anchor = animal_camera_root_anchor(
                        obj,
                        args,
                        w_eye,
                        height,
                        full_meta_w=full_meta_w,
                        full_meta_h=full_meta_h,
                        crop_x0=crop_x0,
                        crop_y0=crop_y0,
                        crop_w=crop_w,
                        crop_h=crop_h,
                    )
                    anchor_depth_stats: Dict[str, float] = {}
                    if animal_root_anchor is not None:
                        anchor_u, anchor_v, anchor_z_raw, _animal_root_xyz = animal_root_anchor
                        anchor_source = "animal_camera_root"
                        anchor_depth_stats = {
                            "valid_count": 1.0,
                            "median": float(anchor_z_raw),
                            "iqr": 0.0,
                            "mad": 0.0,
                            "p10": float(anchor_z_raw),
                            "p90": float(anchor_z_raw),
                        }
                    else:
                        anchor_uv = None
                        anchor_uv_from_mask = False
                        source_w = int(obj.get("source_w", full_meta_w))
                        source_h = int(obj.get("source_h", full_meta_h))
                        if category_id in (TYPE_PERSON, TYPE_ANIMAL):
                            anchor_uv = compute_anchor_from_keypoints(
                                obj.get("keypoints"),
                                obj.get("kp_vis") or [],
                                obj.get("anchor_indices") or [],
                            )
                        if anchor_uv is None:
                            anchor_uv_source = compute_mask_centroid(
                                obj["segmentation"],
                                source_h,
                                source_w,
                            )
                            if anchor_uv_source is not None:
                                anchor_uv_from_mask = True
                                anchor_u_meta, anchor_v_meta = anchor_uv_source
                                anchor_u_meta = shift_left_eye_if_full_sbs(
                                    anchor_u_meta, w_eye, width, bool(obj.get("is_full_sbs"))
                                )
                                anchor_u_meta, anchor_v_meta = source_to_metadata_point(
                                    anchor_u_meta,
                                    anchor_v_meta,
                                    source_w,
                                    source_h,
                                    full_meta_w,
                                    full_meta_h,
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
                            anchor_source = "bbox_center_depth"
                        else:
                            anchor_u, anchor_v = anchor_uv
                        anchor_u, anchor_v = clamp_point(anchor_u, anchor_v, w_eye, height)

                        if anchor_source != "bbox_center_depth":
                            # anchor_source describes the uv basis here (mask centroid vs
                            # keypoint midpoint); z is always window-sampled below. See
                            # docs/bundle-shared/archive/D-005-depth-sampling-window.md Q2 -- Unity couldn't tell
                            # from the old single "depth_sample" label whether mask-based
                            # anchoring was actually in effect for `other` tracks.
                            anchor_source = (
                                "mask_centroid_depth_sample" if anchor_uv_from_mask else "keypoint_depth_sample"
                            )
                        anchor_z_raw = None
                        if (
                            args.person_anchor_mask_median
                            and category_id == TYPE_PERSON
                            and obj.get("segmentation") is not None
                        ):
                            # D-004: the pelvis point sample flips between the hip and
                            # the subject's own forearm. The mask median can't, because
                            # one limb is a small fraction of the silhouette. uv is left
                            # alone -- only z changes, so the diff against the previous
                            # build is one quantity.
                            mask_stats: Dict[str, float] = {}
                            anchor_z_raw = sample_depth_mask_median(
                                depth_frame,
                                obj["segmentation"],
                                source_w,
                                source_h,
                                full_meta_w,
                                full_meta_h,
                                crop_x0,
                                crop_y0,
                                stats_out=mask_stats,
                            )
                            if anchor_z_raw is not None:
                                anchor_depth_stats = mask_stats
                                anchor_source = "person_mask_median_depth"
                        if anchor_z_raw is None:
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
                                stats_out=anchor_depth_stats,
                            )
                        if anchor_z_raw is not None:
                            # Remove DepthCrafter's few-seconds-scale absolute disparity
                            # drift before it becomes anchor_z (see
                            # docs/bundle-shared/D-004-anchor-z-accuracy.md). Not applied to
                            # animal_camera_root, which doesn't go through this branch.
                            anchor_z_raw = anchor_z_raw - background_drift_correction[frame_idx]
                            anchor_z_raw = disparity_to_camera_z(anchor_z_raw)

                    state = track_state.setdefault(track_id, TrackState())
                    reset_reason = reset_track_state_if_discontinuous(state, frame_idx, frame_shot)
                    if reset_reason is not None:
                        track_reset_counts[reset_reason] = track_reset_counts.get(reset_reason, 0) + 1
                    placement_eval = evaluate_placement_observation(
                        bbox=(bbox_x, bbox_y, bbox_w, bbox_h),
                        anchor_u=anchor_u,
                        anchor_v=anchor_v,
                        anchor_z_raw=anchor_z_raw,
                        anchor_source=anchor_source,
                        depth_stats=anchor_depth_stats,
                        state=state,
                        w_eye=w_eye,
                        height=height,
                        args=args,
                    )
                    raw_anchor_u = float(anchor_u)
                    raw_anchor_v = float(anchor_v)
                    raw_anchor_z = float(anchor_z_raw) if anchor_z_raw is not None else None
                    raw_anchor_source = str(anchor_source)
                    resolved = resolve_track_anchor(
                        state=state,
                        placement_eval=placement_eval,
                        anchor_u=anchor_u,
                        anchor_v=anchor_v,
                        anchor_z_raw=anchor_z_raw,
                        anchor_source=anchor_source,
                        frame_idx=frame_idx,
                        args=args,
                    )
                    if resolved["skipped"]:
                        skipped_no_depth += 1
                        print(
                            f"Warning: frame {frame_idx} track {track_id} has no usable anchor depth "
                            f"({raw_anchor_source}) and no previous anchor to carry; object not "
                            "written to meta.bin for this frame."
                        )
                        frame_placement_observations.append(
                            {
                                "trackId": track_id,
                                "categoryId": category_id,
                                "category": cat_name_map.get(category_id, f"cat_{category_id}"),
                                "bbox": [float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h)],
                                "rawAnchor": {
                                    "u": raw_anchor_u,
                                    "v": raw_anchor_v,
                                    "z": None,
                                    "source": raw_anchor_source,
                                },
                                "usedAnchor": None,
                                "skipped": True,
                                "skipReason": "no_anchor_depth_at_cold_start",
                                "placementConfidence": float(placement_eval["confidence"]),
                                "placementStatus": placement_eval["status"],
                                "placementHeld": False,
                                "holdSource": None,
                                "reasons": placement_eval["reasons"],
                                "resetReason": reset_reason,
                            }
                        )
                        continue
                    anchor_u = resolved["anchor_u"]
                    anchor_v = resolved["anchor_v"]
                    anchor_z = resolved["anchor_z"]
                    anchor_source = resolved["anchor_source"]
                    placement_held = resolved["placement_held"]
                    hold_source = resolved["hold_source"]

                    anchor_z_q = int(round(float(anchor_z) / args.quant_pos_scale))
                    if not -32768 <= anchor_z_q <= 32767:
                        # Silent clamping would pin every far object to the same
                        # step -- the exact failure D-008 reported, but worse.
                        raise SystemExit(
                            f"anchor_z={anchor_z:.6f} at frame {frame_idx} track {track_id} "
                            f"needs anchor_z_q={anchor_z_q}, which overflows int16 at "
                            f"--quant_pos_scale {args.quant_pos_scale}. Re-run with a "
                            f"larger scale (at least {abs(anchor_z) / 32767:.8f}). "
                            f"Depth-sampled anchors stay under 1.0; animal_camera_root "
                            f"anchors are camera-space Z and can be much larger."
                        )
                    anchor_scale_q = 65535
                    debug_depth = int(args.debug_frame) == frame_idx
                    frame_placement_observations.append(
                        {
                            "trackId": track_id,
                            "categoryId": category_id,
                            "category": cat_name_map.get(category_id, f"cat_{category_id}"),
                            "bbox": [
                                float(bbox_x),
                                float(bbox_y),
                                float(bbox_w),
                                float(bbox_h),
                            ],
                            "rawAnchor": {
                                "u": raw_anchor_u,
                                "v": raw_anchor_v,
                                "z": raw_anchor_z,
                                "source": raw_anchor_source,
                            },
                            "usedAnchor": {
                                "u": float(anchor_u),
                                "v": float(anchor_v),
                                "z": float(anchor_z),
                                "source": anchor_source,
                            },
                            "placementConfidence": float(placement_eval["confidence"]),
                            "placementStatus": placement_eval["status"],
                            "placementHeld": placement_held,
                            "holdSource": hold_source,
                            "reasons": placement_eval["reasons"],
                            "resetReason": reset_reason,
                            "areaRatioFromPrevious": placement_eval["areaRatioFromPrevious"],
                            "anchorJumpPx": placement_eval["anchorJumpPx"],
                            "edgeTouch": placement_eval["edgeTouch"],
                            "depthStats": placement_eval["depthStats"],
                        }
                    )

                    keypoints = obj["keypoints"]
                    kp_vis = obj.get("kp_vis") or []
                    kp_expected = int(obj.get("kp_expected", 0))
                    has_skeleton = bool(obj.get("has_skeleton", False))
                    smpl_payload = obj.get("smpl_payload")
                    smal_payload = obj.get("smal_payload")
                    flags = 0
                    if has_skeleton:
                        flags |= FLAG_SKELETON
                    if smpl_payload is not None:
                        flags |= FLAG_SMPL
                    if smal_payload is not None:
                        flags |= FLAG_SMAL
                    joints_rel_q_local = None
                    encoded_skeleton = False

                    written_objects += 1
                    payload.extend(struct.pack("<I", track_id))
                    payload.extend(struct.pack("<B", category_id))
                    payload.extend(struct.pack("<B", flags))
                    # Round the corners, then derive w/h, so x0+w is the rounded x1
                    # (rounding x and w independently could be off by one).
                    bbox_x0_q = int(round(bbox_x))
                    bbox_y0_q = int(round(bbox_y))
                    bbox_w_q = max(0, int(round(bbox_x + bbox_w)) - bbox_x0_q)
                    bbox_h_q = max(0, int(round(bbox_y + bbox_h)) - bbox_y0_q)
                    payload.extend(struct.pack("<HHHH", bbox_x0_q, bbox_y0_q, bbox_w_q, bbox_h_q))
                    payload.extend(struct.pack("<HH", int(round(anchor_u)), int(round(anchor_v))))
                    payload.extend(struct.pack("<h", anchor_z_q))
                    payload.extend(struct.pack("<H", anchor_scale_q))
                    payload.extend(struct.pack("<hhhh", *ROT_Q))

                    if has_skeleton and obj.get("source_joints3d") is not None:
                        source_joints3d = obj["source_joints3d"]
                        source_valid = obj.get("source_joints_valid")
                        if source_valid is None:
                            source_valid = np.ones((int(source_joints3d.shape[0]),), dtype=bool)
                        joint_scale = pose_joint_scale(
                            args,
                            obj.get("source_coord_system"),
                            obj.get("source_engine"),
                        )
                        anchor_xyz = camera_xyz_from_uv_depth(
                            anchor_u,
                            anchor_v,
                            anchor_z,
                            w_eye,
                            height,
                            args.fovx_deg,
                        )
                        if state.kp_count != kp_expected:
                            state.joints_rel = None
                            state.joints_abs = None
                            state.kp_count = kp_expected
                        flip_y = resolve_pose_flip_y(args, obj.get("source_axes"))
                        flip_key = f"{obj.get('source_axes') or 'unlabelled'}->flip_y={int(flip_y)}"
                        pose_flip_y_seen[flip_key] = pose_flip_y_seen.get(flip_key, 0) + 1
                        joints_out, joints_rel, joints_valid, root_raw = compute_pose_keypoints3d_for_bundle(
                            source_joints3d=source_joints3d,
                            source_valid=source_valid,
                            root_indices=obj.get("root_indices", []),
                            anchor_xyz=anchor_xyz,
                            joints_space=args.joints_space,
                            joint_scale=joint_scale,
                            flip_y=flip_y,
                            prev_joints_rel=state.joints_rel,
                            ema_alpha=args.ema_alpha if args.joints_space == "camera_xyz_root_relative" else 0.0,
                        )
                        state.joints_rel = joints_rel
                        state.joints_abs = joints_rel + anchor_xyz.reshape(1, 3)
                        if len(kp_vis) != kp_expected:
                            kp_vis = [
                                1 if kp_idx < int(joints_valid.shape[0]) and bool(joints_valid[kp_idx]) else 0
                                for kp_idx in range(kp_expected)
                            ]
                        for kp_idx in range(min(kp_expected, int(joints_valid.shape[0]))):
                            if not bool(joints_valid[kp_idx]):
                                kp_vis[kp_idx] = 0
                        joints_rel_q = quantize_array_int16(joints_out, args.quant_joint_scale)
                        joints_rel_q_local = joints_rel_q
                        payload.extend(
                            struct.pack("<" + "h" * (kp_expected * 3), *joints_rel_q.reshape(-1).tolist())
                        )
                        payload.extend(struct.pack("<" + "B" * kp_expected, *kp_vis))
                        encoded_skeleton = True
                        if debug_depth:
                            print(
                                f"[pose3d_source_debug] frame={frame_idx} trackId={track_id} "
                                f"source={obj.get('source_engine')}/{obj.get('source_coord_system')} "
                                f"joint_scale={joint_scale:.6f} anchor_xyz=({anchor_xyz[0]:.4f},{anchor_xyz[1]:.4f},{anchor_xyz[2]:.4f}) "
                                f"root=({root_raw[0]:.4f},{root_raw[1]:.4f},{root_raw[2]:.4f}) "
                                f"valid={int(np.count_nonzero(joints_valid))}/{kp_expected} "
                                f"joints_space={args.joints_space} "
                                f"pose_keypoints3d_flip_y={int(flip_y)} ({args.pose_keypoints3d_flip_y})"
                            )

                    if has_skeleton and keypoints is not None and not encoded_skeleton:
                        root_subtracted = args.joints_space == "camera_xyz_root_relative"
                        joints_smoothing = "none"
                        if state.kp_count != kp_expected:
                            state.joints_rel = None
                            state.joints_abs = None
                            state.kp_count = kp_expected
                        prev_joints_abs = state.joints_abs

                        if root_subtracted:
                            prev_joints_rel = state.joints_rel
                            joints_out, joints3d_raw, joints3d_valid, root_raw = compute_joints_rel(
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
                                prev_joints_abs=prev_joints_abs,
                                depth_gate_min_valid=args.depth_gate_min_valid,
                                depth_gate_prev_band=args.depth_gate_prev_band,
                                depth_gate_prev_min_valid=args.depth_gate_prev_min_valid,
                                depth_gate_prev_min_frac=args.depth_gate_prev_min_frac,
                                depth_gate_use_anchor_fallback=bool(args.depth_gate_use_anchor_fallback),
                                anchor_z_fallback=anchor_z,
                                depth_gate_iqr=args.depth_gate_iqr,
                                depth_gate_range=args.depth_gate_range,
                                depth_gate_mad=args.depth_gate_mad,
                                depth_gate_jump=args.depth_gate_jump,
                                depth_gate_conf_margin=args.depth_gate_conf_margin,
                                debug_depth=debug_depth,
                                frame_idx=frame_idx,
                                track_id=track_id,
                            )
                            state.joints_rel = joints_out
                            state.joints_abs = joints3d_raw
                            joints_smoothing = f"ema_alpha={args.ema_alpha}"
                        else:
                            joints3d_raw, joints3d_valid, root_raw = _compute_joints3d_and_root(
                                keypoints=keypoints,
                                depth_frame=depth_frame,
                                w_eye=w_eye,
                                height=height,
                                fovx_deg=args.fovx_deg,
                                sample_k=args.sample_k,
                                conf_th=args.conf_th,
                                meta_w=meta_w,
                                meta_h=meta_h,
                                crop_x0=crop_x0,
                                crop_y0=crop_y0,
                                crop_w=crop_w,
                                crop_h=crop_h,
                                root_indices=obj.get("root_indices", []),
                                prev_joints_abs=prev_joints_abs,
                                depth_gate_min_valid=args.depth_gate_min_valid,
                                depth_gate_prev_band=args.depth_gate_prev_band,
                                depth_gate_prev_min_valid=args.depth_gate_prev_min_valid,
                                depth_gate_prev_min_frac=args.depth_gate_prev_min_frac,
                                depth_gate_use_anchor_fallback=bool(args.depth_gate_use_anchor_fallback),
                                anchor_z_fallback=anchor_z,
                                depth_gate_iqr=args.depth_gate_iqr,
                                depth_gate_range=args.depth_gate_range,
                                depth_gate_mad=args.depth_gate_mad,
                                depth_gate_jump=args.depth_gate_jump,
                                depth_gate_conf_margin=args.depth_gate_conf_margin,
                                debug_depth=debug_depth,
                                frame_idx=frame_idx,
                                track_id=track_id,
                            )
                            joints_out = joints3d_raw
                            state.joints_rel = None
                            state.joints_abs = joints3d_raw

                        if debug_depth:
                            joints_cam = joints_out
                            joints_mask = np.all(np.isfinite(joints_cam), axis=1)
                            kp_uv = np.array([(float(u), float(v)) for u, v, _ in keypoints], dtype=np.float32)
                            debug_kp_count = kp_expected
                            if kp_uv.shape[0] != debug_kp_count:
                                debug_kp_count = int(kp_uv.shape[0])
                            kp_mask = np.array(
                                [
                                    idx < len(kp_vis) and kp_vis[idx] > 0 and math.isfinite(float(kp_uv[idx, 0])) and math.isfinite(float(kp_uv[idx, 1]))
                                    for idx in range(debug_kp_count)
                                ],
                                dtype=bool,
                            )
                            if joints_mask.shape[0] != debug_kp_count:
                                joints_mask = joints_mask[:debug_kp_count]
                                joints3d_valid = joints3d_valid[:debug_kp_count]
                            joints_abs = joints_cam + root_raw.reshape(1, 3) if root_subtracted else joints_cam
                            uv_inv, uv_inv_valid = invert_camera_xyz_to_uv(joints_abs, w_eye, height, args.fovx_deg)
                            inv_mask = kp_mask & joints3d_valid & uv_inv_valid

                            def mm(arr: np.ndarray, mask: np.ndarray, col: int) -> Tuple[Optional[float], Optional[float]]:
                                if arr.size == 0 or mask.size == 0 or not np.any(mask):
                                    return None, None
                                vals = arr[mask, col]
                                if vals.size == 0:
                                    return None, None
                                return float(np.min(vals)), float(np.max(vals))

                            u_min, u_max = mm(kp_uv, kp_mask, 0)
                            v_min, v_max = mm(kp_uv, kp_mask, 1)
                            jx_min, jx_max = mm(joints_cam, joints_mask, 0)
                            jy_min, jy_max = mm(joints_cam, joints_mask, 1)
                            jz_min, jz_max = mm(joints_cam, joints_mask, 2)
                            ui_min, ui_max = mm(uv_inv, inv_mask, 0)
                            vi_min, vi_max = mm(uv_inv, inv_mask, 1)
                            err_mean = None
                            err_max = None
                            if np.any(inv_mask):
                                delta = uv_inv[inv_mask] - kp_uv[inv_mask]
                                err = np.sqrt(np.sum(delta * delta, axis=1))
                                if err.size > 0:
                                    err_mean = float(np.mean(err))
                                    err_max = float(np.max(err))
                            print(
                                f"[joints_debug] frame={frame_idx} trackId={track_id} kpCount={debug_kp_count} "
                                f"joints_space={args.joints_space} root_subtracted={1 if root_subtracted else 0} "
                                f"anchor=({anchor_u:.3f},{anchor_v:.3f}) bbox=({bbox_w:.3f},{bbox_h:.3f}) "
                                f"uv[min,max]=({_fmt_float(u_min)},{_fmt_float(u_max)})x({_fmt_float(v_min)},{_fmt_float(v_max)}) "
                                f"jointsCam[min,max]=x({_fmt_float(jx_min)},{_fmt_float(jx_max)}) "
                                f"y({_fmt_float(jy_min)},{_fmt_float(jy_max)}) z({_fmt_float(jz_min)},{_fmt_float(jz_max)}) "
                                f"uv_inv[min,max]=({_fmt_float(ui_min)},{_fmt_float(ui_max)})x({_fmt_float(vi_min)},{_fmt_float(vi_max)}) "
                                f"uv_err[mean,max]=({_fmt_float(err_mean)},{_fmt_float(err_max)}) nInv={int(np.count_nonzero(inv_mask))} "
                                f"smoothing={joints_smoothing}"
                            )
                        if len(kp_vis) != kp_expected:
                            kp_vis = (kp_vis + [0] * kp_expected)[:kp_expected]
                        valid_n = min(kp_expected, int(joints_out.shape[0]))
                        for kp_idx in range(valid_n):
                            if (
                                float(joints_out[kp_idx, 0]) == 0.0
                                and float(joints_out[kp_idx, 1]) == 0.0
                                and float(joints_out[kp_idx, 2]) == 0.0
                            ):
                                kp_vis[kp_idx] = 0
                        joints_rel_q = quantize_array_int16(joints_out, args.quant_joint_scale)
                        joints_rel_q_local = joints_rel_q
                        payload.extend(
                            struct.pack("<" + "h" * (kp_expected * 3), *joints_rel_q.reshape(-1).tolist())
                        )
                        payload.extend(struct.pack("<" + "B" * kp_expected, *kp_vis))
                    if smpl_payload is not None:
                        payload.extend(pack_smpl_payload(smpl_payload))
                    if smal_payload is not None:
                        payload.extend(pack_smal_payload(smal_payload))
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
                                "anchor_source": anchor_source,
                                "has_skeleton": has_skeleton,
                                "kp_count": kp_expected,
                                "vis_count": vis_count,
                                "joints_q_min": joints_q_min,
                                "joints_q_max": joints_q_max,
                            }
                        )

                bg_disp_raw = background_disparity_series[frame_idx]
                placement_observation_frames.append(
                    {
                        "frameIndex": frame_idx,
                        "backgroundDisparity": (
                            float(bg_disp_raw) if np.isfinite(bg_disp_raw) else None
                        ),
                        "backgroundDriftCorrection": float(background_drift_correction[frame_idx]),
                        "objects": frame_placement_observations,
                    }
                )
                payload = bytearray(struct.pack("<H", written_objects)) + payload
                compressed = compress_fn(bytes(payload))
                offset = meta_f.tell()
                if debug_frame:
                    log_frame_summary(
                        frame_idx,
                        offset,
                        len(payload),
                        len(compressed),
                        written_objects,
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

        print(
            "Track state resets: "
            f"cold_start={track_reset_counts.get('cold_start', 0)} "
            f"gap={track_reset_counts.get('gap', 0)} shot={track_reset_counts.get('shot', 0)}; "
            f"objects skipped for lack of depth: {skipped_no_depth}; "
            f"pose axes seen: {pose_axes_seen or 'none'}"
        )

        joints_smoothing_manifest = (
            f"ema_alpha={args.ema_alpha}"
            if args.joints_space == "camera_xyz_root_relative"
            else "none"
        )
        fovx_rad = math.radians(args.fovx_deg)
        fx_norm = 1.0 / math.tan(fovx_rad / 2.0)
        fy_norm = fx_norm * (float(w_eye) / float(height))
        cx = float(w_eye) * 0.5
        cy = float(height) * 0.5
        # Every CLI argument, paths reduced to basenames, so a bundle can be
        # rebuilt without consulting recommended_bundles.json (flow-audit B8).
        path_args = {"video_mp4", "depth_npz", "metadata_json", "out_bundle", "shots_json", "dump_manifest"}
        build_args = {
            key: (os.path.basename(str(value)) if key in path_args and value else value)
            for key, value in vars(args).items()
        }
        manifest = {
            "width": width,
            "height": height,
            "eye_w": w_eye,
            "eye_h": height,
            "meta_w": meta_w,
            "meta_h": meta_h,
            "depth_w": full_meta_w,
            "depth_h": full_meta_h,
            "crop_x0": crop_x0,
            "crop_y0": crop_y0,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "crop_fraction": crop_fraction,
            "align128": args.align128,
            "crop_mode": args.crop_mode,
            "fps": fps,
            "num_frames": num_frames,
            "left_eye_origin": args.left_eye_origin,
            "fovx_deg": args.fovx_deg,
            "quant_pos_scale": args.quant_pos_scale,
            "quant_joint_scale": args.quant_joint_scale,
            "joints_space": args.joints_space,
            "joints_source": args.joints_source,
            "depth_policy": {
                "schema": "master_project.depth_policy.v1",
                "convention": "depthcrafter_normalized_disparity",
                "near_far_direction": "depth_npz values: 0.0=far/back, 1.0=near/front",
                "depth_npz_format": depth_npz.format,
                "disp_min": depth_disp_min,
                "disp_max": depth_disp_max,
                "normalization": "global_minmax_pre_clip",
                "meaning": (
                    "disp_min/disp_max are the pre-normalization min/max DepthCrafter "
                    "produced for this clip, before the global min-max normalization "
                    "that maps depth_npz into [0, 1]. Recover the pre-normalization "
                    "value with disp_raw = disp_norm * (disp_max - disp_min) + disp_min, "
                    "then treat 1 / disp_raw as a relative (not metric) depth ordering "
                    "signal -- DepthCrafter is affine-invariant, so disp_raw ~= a/Z + b "
                    "for unknown per-clip a, b; disp_min/disp_max do not calibrate a, b. "
                    "disp_min/disp_max can be null for depth_npz files produced before "
                    "this field was added."
                ),
                "relation_to_anchor_z": (
                    "meta.bin anchor_z is this same normalized disparity, flipped: "
                    "anchor_z = max(1 - disp_norm, 1e-4), larger=farther, matching "
                    "camera_xyz_from_uv_depth. So anchor_z is linear in DISPARITY, not "
                    "in depth -- the far field is compressed into a narrow band by "
                    "construction and anchor_z is not metres. Do not apply the "
                    "disp_min/disp_max recovery formula to anchor_z; DepthCrafter is "
                    "affine-invariant, so no per-clip a, b are known."
                ),
            },
            # One dict for everything Unity reads about anchor_z: what the number is
            # (definition/units/quantizer) and which quantity each category holds.
            # These used to be two literals under the same key, so the first was
            # silently dropped and requantize_bundle_anchor_z.py then replaced the
            # survivor wholesale (flow-audit B4); requantize now update()s only
            # quant_pos_scale/quant_note.
            "anchor_z_policy": {
                "schema": "master_project.anchor_z_policy.v1",
                "definition": "anchor_z = max(1 - depth_npz_normalized_disparity, 1e-4)",
                "units": (
                    "Not metres. Linear in DepthCrafter's normalized disparity, so equal "
                    "anchor_z steps are not equal distance steps and the far field is "
                    "compressed. Use it as a relative ordering/placement signal only."
                ),
                "quant_pos_scale": args.quant_pos_scale,
                "quant_note": (
                    "Read quant_pos_scale from this manifest (or the meta.bin header) -- "
                    "never hardcode it. It was reduced from 0.002 to 0.0001 so that "
                    "same-frame objects in the far field keep distinct anchor_z steps "
                    "instead of collapsing onto one. Read it from here per bundle -- a "
                    "clip whose anchors are all depth-sampled may ship a finer step. "
                    "See docs/bundle-shared/archive/D-008-anchor-z-quantization.md."
                ),
                "person": (
                    "person_mask_median_depth -- median disparity over the whole SAM2 mask"
                    if args.person_anchor_mask_median
                    else "keypoint_depth_sample -- 7x7 window at the pelvis keypoint"
                ),
                "animal": "animal_camera_root -- AniMer camera-space root Z. Does not read the depth map at all.",
                "other": "mask_centroid_depth_sample -- 7x7 window at the SAM2 mask centroid",
                "person_anchor_mask_median": int(args.person_anchor_mask_median),
                "meaning": (
                    "Which quantity anchor_z holds, per category. The three are not "
                    "interchangeable: the person/other paths are normalized disparity "
                    "(larger=farther after the flip), the animal path is camera-space Z "
                    "on AniMer's own scale. See docs/bundle-shared/D-004-anchor-z-accuracy.md."
                ),
                "sidecar_field": "rawAnchor.source in source/placement_observations.json is authoritative per frame.",
            },
            "pose_keypoints3d_policy": {
                "enabled": args.joints_source in ("auto", "pose_keypoints3d"),
                "meaning": "pose.keypoints3d is treated as skeleton shape. Animal camera_xyz_absolute roots are also used as placement anchors; depth sampling remains the fallback anchor source.",
                "metrabs_joint_scale": args.metrabs_joint_scale,
                "animer_joint_scale": args.animer_joint_scale,
                "flip_y": args.pose_keypoints3d_flip_y,
                "flip_y_meaning": (
                    "auto: the Y flip is derived per object from the sidecar's axis label "
                    "(pose.cameraAxes / pose.coordinateSystem) so that output_axes holds "
                    "for every input; 0/1: the pre-2026-09-17 blind flag."
                ),
                "source_axes_seen": pose_axes_seen,
                "flip_y_applied": pose_flip_y_seen,
                "output_axes": BUNDLE_JOINT_AXES,
                "animal_camera_root_anchor": True,
                "animal_root_reprojection": (
                    "eye_canvas_legacy"
                    if str(args.pose_keypoints3d_flip_y).lower() in ("0", "1")
                    else "source_canvas_opencv_then_source_to_eye"
                ),
            },
            "smpl_meta_policy": {
                "enabled": True,
                "flag_bit": 1,
                "block_version": SMPL_BLOCK_VERSION,
                "layout": "uint16 version, uint16 rotation_count, uint16 beta_count, float32 rotation_matrices[rotation_count][3][3], float32 betas[beta_count], float32 transl[3]",
                "rotation_count": SMPL_ROTATION_COUNT,
                "beta_count": SMPL_BETA_COUNT,
                "parameterization": "rotation_matrix",
                "rotation_order": ["global_orient", "body_pose_23"],
            },
            "smal_meta_policy": {
                "enabled": True,
                "flag_bit": 2,
                "block_version": SMAL_BLOCK_VERSION,
                "layout": "uint16 version, uint16 rotation_count, uint16 beta_count, float32 rotation_matrices[rotation_count][3][3], float32 betas[beta_count], float32 transl[3]",
                "rotation_count": SMAL_ROTATION_COUNT,
                "beta_count": SMAL_BETA_COUNT,
                "parameterization": "rotation_matrix",
                "rotation_order": ["global_orient", "pose_34"],
            },
            "placement_observation_policy": {
                "schema": "master_project.placement_observation_policy.v1",
                "runtime_behavior": (
                    "meta.bin stores the held/smoothed placement anchor. Low-confidence "
                    "observations hold the previous high-confidence anchor when available. "
                    "Per-track EMA/hold state is reset at every shot change and after any "
                    "in-shot track gap (resetReason in the sidecar). An object with no "
                    "usable depth at a cold start is not written for that frame "
                    "(skipped=true in the sidecar) instead of anchor_z=0."
                ),
                "sidecar": "source/placement_observations.json",
                "track_state_resets": track_reset_counts,
                "objects_skipped_no_depth": skipped_no_depth,
                "hold_threshold": float(args.placement_conf_hold_threshold),
                "edge_margin_px": float(args.placement_conf_edge_margin_px),
                "area_shrink_ratio": float(args.placement_conf_area_shrink_ratio),
                "anchor_jump_px": float(args.placement_conf_anchor_jump_px),
                "depth_jump": float(args.placement_conf_depth_jump),
                "confidence_meaning": "0.0 low confidence, 1.0 high confidence for placement update only.",
            },
            "shots": [[int(s), int(e)] for s, e in shots],
            "depth_scale_calibration": depth_scale_calibration,
            "shot_boundary_policy": {
                "schema": "master_project.shot_boundary_policy.v1",
                "meaning": (
                    "Each [start, end) range in shots is one continuous camera take with "
                    "no hard cut inside it, in bundle frame indices. Camera distance/size "
                    "can legitimately differ between shots for the same trackId -- it is "
                    "not object motion. The bundle build resets per-track EMA smoothing "
                    "and placement-hold gating at each shot's first frame, so anchor_u/"
                    "anchor_v/anchor_z and root-relative skeleton joints are a fresh, "
                    "ungated observation there instead of a blend with the previous shot."
                ),
                "unity_guidance": (
                    "Do not interpolate or spring position/scale across a shot boundary "
                    "for the same trackId; snap to the new shot's first-frame anchor "
                    "instead. SMPL/SMAL betas are estimated independently per frame and "
                    "are not reset or smoothed by shot -- their frame-to-frame noise is a "
                    "separate, generally-noisy signal unrelated to shot boundaries."
                ),
                "single_shot_default": args.shots_json is None,
            },
            "camera_axes": BUNDLE_JOINT_AXES,
            "uv_origin": "top_left",
            "joints_quant_scale": args.quant_joint_scale,
            "smoothing": joints_smoothing_manifest,
            "fx_norm": fx_norm,
            "fy_norm": fy_norm,
            "cx": cx,
            "cy": cy,
            "frame_compress": compress_name,
            "frame_compress_requested": args.frame_compress,
            "video_transcode": transcode_info,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "inputs": {
                "video_mp4": os.path.basename(args.video_mp4),
                "video_mp4_bundle_source": os.path.basename(video_mp4_path),
                "depth_npz": os.path.basename(args.depth_npz),
                "metadata_json": os.path.basename(args.metadata_json),
            },
            "build_args": build_args,
        }
        print(
            "[manifest_verify] "
            f"joints_space={manifest['joints_space']} "
            f"fx_norm={manifest['fx_norm']:.6f} fy_norm={manifest['fy_norm']:.6f} "
            f"eye=({manifest['eye_w']},{manifest['eye_h']}) "
            f"joints_quant_scale={manifest['joints_quant_scale']}"
        )
        if args.dump_manifest:
            dump_path = os.path.abspath(args.dump_manifest)
            dump_dir = os.path.dirname(dump_path)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            with open(dump_path, "w", encoding="utf-8") as dump_f:
                json.dump(manifest, dump_f, indent=2)
                dump_f.write("\n")

        placement_observations = {
            "schema": "master_project.placement_observations.v1",
            "policy": manifest["placement_observation_policy"],
            "frames": placement_observation_frames,
            "summary": {
                "numFrames": len(placement_observation_frames),
                "numObjects": sum(len(frame.get("objects", [])) for frame in placement_observation_frames),
                "numHeldPlacements": sum(
                    1
                    for frame in placement_observation_frames
                    for obj in frame.get("objects", [])
                    if obj.get("placementHeld")
                ),
                "numLowConfidence": sum(
                    1
                    for frame in placement_observation_frames
                    for obj in frame.get("objects", [])
                    if obj.get("placementStatus") == "low"
                ),
            },
        }

        with zipfile.ZipFile(args.out_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(video_mp4_path, arcname="video.mp4", compress_type=zipfile.ZIP_STORED)
            zf.write(meta_path, arcname="meta.bin", compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr(
                "source/placement_observations.json",
                json.dumps(placement_observations, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
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
                    "level_requested": args.transcode_level,
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
