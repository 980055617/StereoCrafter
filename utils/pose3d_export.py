import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - optional dependency
    mask_utils = None


LOGGER = logging.getLogger("pose3d_export")


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def load_camera_intrinsics(path: str, *, width: int, height: int) -> CameraIntrinsics:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fx = float(payload["fx"])
    fy = float(payload.get("fy", fx))
    cx = float(payload.get("cx", (width - 1) / 2.0))
    cy = float(payload.get("cy", (height - 1) / 2.0))
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=int(payload.get("width", width)),
        height=int(payload.get("height", height)),
    )


def backproject_pixel(
    u: float,
    v: float,
    z: float,
    intrinsics: Optional[CameraIntrinsics],
    *,
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float, float]:
    if intrinsics is None:
        denom_x = max((frame_width - 1) / 2.0, 1.0)
        denom_y = max((frame_height - 1) / 2.0, 1.0)
        x = (u - (frame_width - 1) / 2.0) / denom_x * z
        y = (v - (frame_height - 1) / 2.0) / denom_y * z
        return float(x), float(y), float(z)

    x = (u - intrinsics.cx) / intrinsics.fx * z
    y = (v - intrinsics.cy) / intrinsics.fy * z
    return float(x), float(y), float(z)


def pixel_to_ray(
    u: float,
    v: float,
    intrinsics: Optional[CameraIntrinsics],
    *,
    frame_width: int,
    frame_height: int,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to a ray direction in the camera frame (normalized image coords).

    Returns (x_ray, y_ray) where:
      - with intrinsics: x_ray=(u-cx)/fx, y_ray=(v-cy)/fy
      - without intrinsics: normalized by image half-size (unitless)

    This is often more stable than scaling X,Y by an arbitrary/relative depth Z.
    """
    if intrinsics is None:
        denom_x = max((frame_width - 1) / 2.0, 1.0)
        denom_y = max((frame_height - 1) / 2.0, 1.0)
        x_ray = (u - (frame_width - 1) / 2.0) / denom_x
        y_ray = (v - (frame_height - 1) / 2.0) / denom_y
        return float(x_ray), float(y_ray)
    return float((u - intrinsics.cx) / intrinsics.fx), float((v - intrinsics.cy) / intrinsics.fy)


def depth_patch_values(depth_frame: np.ndarray, u: float, v: float, radius: int) -> np.ndarray:
    height, width = depth_frame.shape[:2]
    x = int(round(u))
    y = int(round(v))
    x0 = max(x - radius, 0)
    x1 = min(x + radius + 1, width)
    y0 = max(y - radius, 0)
    y1 = min(y + radius + 1, height)
    patch = depth_frame[y0:y1, x0:x1]
    if patch.size == 0:
        return np.empty((0,), dtype=np.float32)
    return patch.reshape(-1).astype(np.float32, copy=False)


def depth_stats(values: np.ndarray) -> Optional[Dict[str, float]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def bbox_to_corners_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("COCO bbox must be [x, y, w, h]")
    x, y, w, h = map(float, bbox)
    return x, y, x + w, y + h


def decode_coco_rle(segmentation: Any) -> Optional[np.ndarray]:
    """
    Decode COCO RLE segmentation dict into a boolean mask.

    Uses pycocotools if available; otherwise falls back to a pure-Python decoder for
    compressed RLE strings (the common COCO JSON representation).
    """
    if segmentation is None:
        return None
    if not isinstance(segmentation, dict):
        return None
    if "counts" not in segmentation or "size" not in segmentation:
        return None

    if mask_utils is not None:
        decoded = mask_utils.decode(segmentation)
        if decoded is None:
            return None
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return decoded.astype(bool)

    size = segmentation.get("size")
    counts = segmentation.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        return None

    rle: List[int] = []
    if isinstance(counts, list):
        rle = [int(v) for v in counts]
    elif isinstance(counts, str):
        # Port of pycocotools' compressed RLE decoder.
        s = counts
        p = 0
        while p < len(s):
            x = 0
            k = 0
            more = 1
            while more:
                c = ord(s[p]) - 48
                p += 1
                x |= (c & 0x1F) << (5 * k)
                more = c & 0x20
                k += 1
                if not more and (c & 0x10):
                    x |= -1 << (5 * k)
            if len(rle) > 1:
                x += rle[-2]
            rle.append(int(x))
    else:
        return None

    total = height * width
    flat = np.zeros((total,), dtype=np.uint8)
    idx = 0
    val = 0
    for rl in rle:
        rl = int(rl)
        if rl <= 0:
            val ^= 1
            continue
        end = idx + rl
        if idx < total:
            flat[idx:min(end, total)] = val
        idx = end
        val ^= 1

    # COCO RLE is in column-major (Fortran) order.
    mask = flat.reshape((height, width), order="F")
    return mask.astype(bool)


def compute_3d_aabb(points_xyz: np.ndarray) -> Optional[Dict[str, Any]]:
    if points_xyz.size == 0:
        return None
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    points_xyz = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if points_xyz.size == 0:
        return None
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    size = maxs - mins
    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()
    corners = [
        [float(x0), float(y0), float(z0)],
        [float(x1), float(y0), float(z0)],
        [float(x1), float(y1), float(z0)],
        [float(x0), float(y1), float(z0)],
        [float(x0), float(y0), float(z1)],
        [float(x1), float(y0), float(z1)],
        [float(x1), float(y1), float(z1)],
        [float(x0), float(y1), float(z1)],
    ]
    return {
        "min": [float(v) for v in mins.tolist()],
        "max": [float(v) for v in maxs.tolist()],
        "center": [float(v) for v in center.tolist()],
        "size": [float(v) for v in size.tolist()],
        "corners": corners,
    }


def _rebuild_corners_from_center_size(center: Sequence[float], size: Sequence[float]) -> Dict[str, Any]:
    center = np.asarray(center, dtype=np.float32)
    size = np.asarray(size, dtype=np.float32)
    half = size / 2.0
    mins = center - half
    maxs = center + half
    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()
    corners = [
        [float(x0), float(y0), float(z0)],
        [float(x1), float(y0), float(z0)],
        [float(x1), float(y1), float(z0)],
        [float(x0), float(y1), float(z0)],
        [float(x0), float(y0), float(z1)],
        [float(x1), float(y0), float(z1)],
        [float(x1), float(y1), float(z1)],
        [float(x0), float(y1), float(z1)],
    ]
    return {
        "min": [float(v) for v in mins.tolist()],
        "max": [float(v) for v in maxs.tolist()],
        "center": [float(v) for v in center.tolist()],
        "size": [float(v) for v in size.tolist()],
        "corners": corners,
    }


def compute_ray_aabb(points_xyrz: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Compute an axis-aligned box in (x_ray, y_ray, z) space.

    This produces a cuboid whose projection is stable even when depth is relative/normalized,
    unlike XYZ boxes that can balloon due to mixing X,Y extrema across different depths.
    """
    if points_xyrz.size == 0:
        return None
    points_xyrz = np.asarray(points_xyrz, dtype=np.float32).reshape(-1, 3)
    points_xyrz = points_xyrz[np.isfinite(points_xyrz).all(axis=1)]
    if points_xyrz.size == 0:
        return None
    mins = points_xyrz.min(axis=0)
    maxs = points_xyrz.max(axis=0)
    center = (mins + maxs) / 2.0
    size = maxs - mins
    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()
    corners = [
        [float(x0), float(y0), float(z0)],
        [float(x1), float(y0), float(z0)],
        [float(x1), float(y1), float(z0)],
        [float(x0), float(y1), float(z0)],
        [float(x0), float(y0), float(z1)],
        [float(x1), float(y0), float(z1)],
        [float(x1), float(y1), float(z1)],
        [float(x0), float(y1), float(z1)],
    ]
    return {
        "min": [float(v) for v in mins.tolist()],
        "max": [float(v) for v in maxs.tolist()],
        "center": [float(v) for v in center.tolist()],
        "size": [float(v) for v in size.tolist()],
        "corners": corners,
    }


def mask_depth_to_pointcloud_aabb(
    *,
    mask: np.ndarray,
    depth_frame: np.ndarray,
    intrinsics: Optional[CameraIntrinsics],
    max_points: int = 8000,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if mask is None:
        return None
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return None

    height, width = depth_frame.shape[:2]
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None

    total = ys.size
    max_points = int(max(max_points, 1))
    if total > max_points:
        step = max(total // max_points, 1)
        idx = np.arange(0, total, step, dtype=np.int64)[:max_points]
        ys = ys[idx]
        xs = xs[idx]

    z = depth_frame[ys, xs].astype(np.float32, copy=False)
    valid = np.isfinite(z)
    if z_min is not None:
        valid &= z >= float(z_min)
    if z_max is not None:
        valid &= z <= float(z_max)
    if not np.any(valid):
        return None

    ys = ys[valid].astype(np.float32, copy=False)
    xs = xs[valid].astype(np.float32, copy=False)
    z = z[valid]

    if intrinsics is None:
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        denom_x = max((width - 1) / 2.0, 1.0)
        denom_y = max((height - 1) / 2.0, 1.0)
        x = (xs - cx) / denom_x * z
        y = (ys - cy) / denom_y * z
    else:
        x = (xs - intrinsics.cx) / intrinsics.fx * z
        y = (ys - intrinsics.cy) / intrinsics.fy * z

    points = np.stack([x, y, z], axis=1)
    return compute_3d_aabb(points)


def mask_depth_to_ray_aabb(
    *,
    mask: np.ndarray,
    depth_frame: np.ndarray,
    intrinsics: Optional[CameraIntrinsics],
    max_points: int = 8000,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if mask is None:
        return None
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return None

    height, width = depth_frame.shape[:2]
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None

    total = ys.size
    max_points = int(max(max_points, 1))
    if total > max_points:
        step = max(total // max_points, 1)
        idx = np.arange(0, total, step, dtype=np.int64)[:max_points]
        ys = ys[idx]
        xs = xs[idx]

    z = depth_frame[ys, xs].astype(np.float32, copy=False)
    valid = np.isfinite(z)
    if z_min is not None:
        valid &= z >= float(z_min)
    if z_max is not None:
        valid &= z <= float(z_max)
    if not np.any(valid):
        return None

    ys = ys[valid].astype(np.float32, copy=False)
    xs = xs[valid].astype(np.float32, copy=False)
    z = z[valid]

    if intrinsics is None:
        denom_x = max((width - 1) / 2.0, 1.0)
        denom_y = max((height - 1) / 2.0, 1.0)
        x_ray = (xs - (width - 1) / 2.0) / denom_x
        y_ray = (ys - (height - 1) / 2.0) / denom_y
    else:
        x_ray = (xs - intrinsics.cx) / intrinsics.fx
        y_ray = (ys - intrinsics.cy) / intrinsics.fy

    points = np.stack([x_ray, y_ray, z], axis=1)
    return compute_ray_aabb(points)


def export_pose_annotations_3d(
    pose_annotations_path: str,
    depth: np.ndarray,
    output_path: str,
    *,
    intrinsics_path: Optional[str] = None,
    keypoint_depth_radius: int = 2,
    min_keypoints_for_3d: int = 4,
    prefer_mask_pointcloud: bool = True,
    mask_pointcloud_max_points: int = 8000,
    bbox3d_space: str = "xyz",
    ema_bbox_alpha: float = 0.0,
    debug: bool = False,
) -> str:
    """
    Lift COCO-style 2D pose/segmentation annotations to pseudo-3D boxes using predicted depth.

    Notes:
    - Depth is assumed to be aligned to the annotation frames (same resolution and frame indices).
    - If intrinsics_path is not provided, projection uses a normalized camera model (unitless).
    - Depth is treated as relative (often normalized) unless you provide metric depth elsewhere.
    """
    with open(pose_annotations_path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images: List[Dict[str, Any]] = coco.get("images", [])
    annotations: List[Dict[str, Any]] = coco.get("annotations", [])
    categories: List[Dict[str, Any]] = coco.get("categories", [])

    image_id_to_frame_id: Dict[int, int] = {}
    image_id_to_size: Dict[int, Tuple[int, int]] = {}
    for image in images:
        image_id = int(image.get("id"))
        frame_id = int(image.get("frame_id", image_id))
        height = int(image.get("height"))
        width = int(image.get("width"))
        image_id_to_frame_id[image_id] = frame_id
        image_id_to_size[image_id] = (height, width)

    category_id_to_name = {int(cat.get("id")): str(cat.get("name")) for cat in categories}

    depth = np.asarray(depth)
    if depth.ndim != 3:
        raise ValueError("depth must have shape [T, H, W]")
    depth_t, depth_h, depth_w = depth.shape

    intrinsics = None
    if intrinsics_path:
        intrinsics = load_camera_intrinsics(intrinsics_path, width=depth_w, height=depth_h)

    out_coco = dict(coco)
    out_annotations: List[Dict[str, Any]] = []

    if bbox3d_space not in {"xyz", "ray"}:
        raise ValueError("bbox3d_space must be one of {'xyz','ray'}")
    coord_system = (
        ("camera" if intrinsics is not None else "normalized_camera")
        if bbox3d_space == "xyz"
        else ("ray_camera" if intrinsics is not None else "ray_normalized")
    )

    for anno in annotations:
        anno = dict(anno)
        image_id = int(anno.get("image_id"))
        frame_id = image_id_to_frame_id.get(image_id)
        if frame_id is None or frame_id < 0 or frame_id >= depth_t:
            anno["bbox3d"] = None
            anno["bbox3d_method"] = "missing_depth"
            out_annotations.append(anno)
            continue

        depth_frame = depth[frame_id]
        img_h, img_w = image_id_to_size.get(image_id, (depth_h, depth_w))
        if (img_h, img_w) != (depth_h, depth_w):
            LOGGER.warning(
                "Annotation image size (%dx%d) differs from depth (%dx%d) for image_id=%s. "
                "Proceeding but indexing may be off.",
                img_w,
                img_h,
                depth_w,
                depth_h,
                image_id,
            )

        category_name = category_id_to_name.get(int(anno.get("category_id", -1)), "unknown")
        keypoints = anno.get("keypoints") or []

        points_xyz: List[Tuple[float, float, float]] = []
        depth_values: Any = []
        method: Optional[str] = None
        mask: Optional[np.ndarray] = None

        valid_kp_count = 0  # Count keypoints that pass v>0 and coordinate checks.
        if category_name in {"person", "animal"} and isinstance(keypoints, list) and len(keypoints) >= 3:
            triplets = [keypoints[i : i + 3] for i in range(0, len(keypoints), 3)]
            sampled = []
            for x, y, v in triplets:
                if v is None or float(v) <= 0:
                    continue
                if x is None or y is None:
                    continue
                x_f = float(x)
                y_f = float(y)
                if x_f <= 0 or y_f <= 0:
                    continue
                valid_kp_count += 1
                patch = depth_patch_values(depth_frame, x_f, y_f, radius=keypoint_depth_radius)
                if patch.size == 0:
                    continue
                z = float(np.median(patch))
                sampled.append(z)
                if bbox3d_space == "xyz":
                    points_xyz.append(
                        backproject_pixel(
                            x_f,
                            y_f,
                            z,
                            intrinsics,
                            frame_width=depth_w,
                            frame_height=depth_h,
                        )
                    )
                else:
                    x_ray, y_ray = pixel_to_ray(
                        x_f,
                        y_f,
                        intrinsics,
                        frame_width=depth_w,
                        frame_height=depth_h,
                    )
                    points_xyz.append((x_ray, y_ray, z))
            if len(points_xyz) >= min_keypoints_for_3d:
                sampled_np = np.asarray(sampled, dtype=np.float32)
                if sampled_np.size > 0:
                    med = float(np.median(sampled_np))
                    std = float(np.std(sampled_np))
                    k_sigma = 1.0
                    lower = med - k_sigma * std
                    upper = med + k_sigma * std
                    if lower > upper:
                        lower, upper = upper, lower
                    if debug:
                        LOGGER.info(
                            "Keypoint depth stats image_id=%s track_id=%s cat=%s: median=%.4f std=%.4f lower=%.4f upper=%.4f (k=%.1f, n=%d)",
                            image_id,
                            anno.get("track_id"),
                            category_name,
                            med,
                            std,
                            lower,
                            upper,
                            k_sigma,
                            sampled_np.size,
                        )
                    # clamp z but keep x,y
                    sampled_np = np.clip(sampled_np, lower, upper)
                    points_xyz = [
                        (pt[0], pt[1], float(np.clip(pt[2], lower, upper)))
                        for pt in points_xyz
                    ]
                method = "keypoints"
                depth_values = sampled_np
            else:
                LOGGER.info(
                    "Fallback to %s for image_id=%s track_id=%s: valid keypoints=%d < min_keypoints_for_3d=%d",
                    "mask/bbox",
                    image_id,
                    anno.get("track_id"),
                    valid_kp_count,
                    min_keypoints_for_3d,
                )

        if method is None:
            segmentation = anno.get("segmentation")
            mask = decode_coco_rle(segmentation)
            if mask is not None and mask.shape != depth_frame.shape:
                # Resize segmentation mask to depth size if needed.
                import cv2  # local import to keep this module lightweight

                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (depth_w, depth_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if mask is not None:
                # Restrict mask to its 2D bbox to avoid stray pixels from RLE decoding artifacts.
                x0, y0, x1, y1 = bbox_to_corners_xyxy(anno.get("bbox", [0, 0, 0, 0]))
                x0i = int(max(min(round(x0), depth_w - 1), 0))
                x1i = int(max(min(round(x1), depth_w), 0))
                y0i = int(max(min(round(y0), depth_h - 1), 0))
                y1i = int(max(min(round(y1), depth_h), 0))
                cropped = np.zeros_like(mask, dtype=bool)
                if x1i > x0i and y1i > y0i:
                    cropped[y0i:y1i, x0i:x1i] = mask[y0i:y1i, x0i:x1i]
                mask = cropped

            if mask is not None and mask.any():
                depth_values = depth_frame[mask].astype(np.float32, copy=False)
                method = "mask"
            else:
                x0, y0, x1, y1 = bbox_to_corners_xyxy(anno.get("bbox", [0, 0, 0, 0]))
                x0i = int(max(min(round(x0), depth_w - 1), 0))
                x1i = int(max(min(round(x1), depth_w), 0))
                y0i = int(max(min(round(y0), depth_h - 1), 0))
                y1i = int(max(min(round(y1), depth_h), 0))
                if x1i > x0i and y1i > y0i:
                    depth_values = depth_frame[y0i:y1i, x0i:x1i].reshape(-1).astype(np.float32, copy=False)
                else:
                    depth_values = np.empty((0,), dtype=np.float32)
                method = "bbox"

        stats = depth_stats(np.asarray(depth_values))
        if stats is None:
            anno["bbox3d"] = None
            anno["bbox3d_method"] = f"{method}_no_depth"
            out_annotations.append(anno)
            continue

        z_near = stats["p10"]
        z_far = stats["p90"]
        if z_far < z_near:
            z_near, z_far = z_far, z_near
        z_thickness = max(z_far - z_near, 1e-6)

        if method == "keypoints" and points_xyz:
            points_np = np.asarray(points_xyz, dtype=np.float32)
            aabb = compute_3d_aabb(points_np) if bbox3d_space == "xyz" else compute_ray_aabb(points_np)
        elif method == "mask" and prefer_mask_pointcloud and mask is not None:
            if bbox3d_space == "xyz":
                aabb = mask_depth_to_pointcloud_aabb(
                    mask=mask,
                    depth_frame=depth_frame,
                    intrinsics=intrinsics,
                    max_points=mask_pointcloud_max_points,
                    # For "other" we keep the full depth range; for human/animal we guard with p10/p90.
                    z_min=None if category_name == "other" else stats["p10"],
                    z_max=None if category_name == "other" else stats["p90"],
                )
            else:
                aabb = mask_depth_to_ray_aabb(
                    mask=mask,
                    depth_frame=depth_frame,
                    intrinsics=intrinsics,
                    max_points=mask_pointcloud_max_points,
                    z_min=None if category_name == "other" else stats["p10"],
                    z_max=None if category_name == "other" else stats["p90"],
                )
            if aabb is None:
                method = "bbox"
        else:
            aabb = None

        if aabb is None:
            x0, y0, x1, y1 = bbox_to_corners_xyxy(anno.get("bbox", [0, 0, 0, 0]))
            corners_uv = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            projected = []
            for z in (z_near, z_far):
                for u, v in corners_uv:
                    if bbox3d_space == "xyz":
                        projected.append(
                            backproject_pixel(
                                float(u),
                                float(v),
                                float(z),
                                intrinsics,
                                frame_width=depth_w,
                                frame_height=depth_h,
                            )
                        )
                    else:
                        x_ray, y_ray = pixel_to_ray(
                            float(u),
                            float(v),
                            intrinsics,
                            frame_width=depth_w,
                            frame_height=depth_h,
                        )
                        projected.append((x_ray, y_ray, float(z)))
            projected_np = np.asarray(projected, dtype=np.float32)
            aabb = compute_3d_aabb(projected_np) if bbox3d_space == "xyz" else compute_ray_aabb(projected_np)

        if aabb is None:
            anno["bbox3d"] = None
            anno["bbox3d_method"] = f"{method}_failed"
            out_annotations.append(anno)
            continue

        anno["bbox3d"] = {
            **aabb,
            "coord_system": coord_system,
            "depth_unit": "relative",
            "depth_stats": stats,
            "depth_thickness": float(z_thickness),
        }
        anno["bbox3d_method"] = method
        out_annotations.append(anno)

    if ema_bbox_alpha > 0.0:
        ema_bbox_alpha = float(np.clip(ema_bbox_alpha, 0.0, 1.0))
        track_prev: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for anno in sorted(out_annotations, key=lambda a: image_id_to_frame_id.get(int(a.get("image_id", -1)), -1)):
            tid = anno.get("track_id")
            bbox3d = anno.get("bbox3d")
            method = anno.get("bbox3d_method")
            if tid is None or not isinstance(bbox3d, dict):
                continue
            if "center" not in bbox3d or "size" not in bbox3d:
                continue
            curr_center = np.asarray(bbox3d["center"], dtype=np.float32)
            curr_size = np.asarray(bbox3d["size"], dtype=np.float32)
            prev = track_prev.get(int(tid))
            # If we lost keypoints/mask and had a previous box, snap to previous to avoid large jumps.
            if prev is not None and method != "keypoints":
                prev_c, prev_s = prev
                curr_center = prev_c
                curr_size = prev_s
            if prev is None:
                sm_center, sm_size = curr_center, curr_size
            else:
                prev_c, prev_s = prev
                sm_center = ema_bbox_alpha * curr_center + (1.0 - ema_bbox_alpha) * prev_c
                sm_size = ema_bbox_alpha * curr_size + (1.0 - ema_bbox_alpha) * prev_s
            bbox3d["center"] = sm_center.tolist()
            bbox3d["size"] = sm_size.tolist()
            rebuilt = _rebuild_corners_from_center_size(sm_center, sm_size)
            bbox3d["min"] = rebuilt["min"]
            bbox3d["max"] = rebuilt["max"]
            bbox3d["corners"] = rebuilt["corners"]
            track_prev[int(tid)] = (sm_center, sm_size)

    out_coco["annotations"] = out_annotations
    out_coco["pose_3d_meta"] = {
        "created_at": datetime.now().isoformat(),
        "coord_system": coord_system,
        "depth_unit": "relative",
        "intrinsics": None
        if intrinsics is None
        else {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "width": intrinsics.width,
            "height": intrinsics.height,
        },
        "source_pose_annotations": pose_annotations_path,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(out_coco, handle, ensure_ascii=False)

    LOGGER.info("Wrote 3D pose annotations to %s", output_path)
    return output_path
