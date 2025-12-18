import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fire import Fire

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - optional dependency
    mask_utils = None


@dataclass(frozen=True)
class CocoIndex:
    images: Dict[int, Dict[str, Any]]
    annotations_by_image_id: Dict[int, List[Dict[str, Any]]]
    category_id_to_name: Dict[int, str]
    skeleton_by_category_id: Dict[int, List[Tuple[int, int]]]


def _build_index(coco: Dict[str, Any]) -> CocoIndex:
    images = {int(img["id"]): img for img in coco.get("images", [])}

    annotations_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for anno in coco.get("annotations", []):
        annotations_by_image_id.setdefault(int(anno["image_id"]), []).append(anno)

    category_id_to_name = {
        int(cat["id"]): str(cat.get("name", cat.get("supercategory", "unknown")))
        for cat in coco.get("categories", [])
    }
    skeleton_by_category_id: Dict[int, List[Tuple[int, int]]] = {}
    for cat in coco.get("categories", []):
        cat_id = int(cat["id"])
        skel = []
        for edge in cat.get("skeleton", []):
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            skel.append((int(edge[0]), int(edge[1])))
        skeleton_by_category_id[cat_id] = skel

    return CocoIndex(
        images=images,
        annotations_by_image_id=annotations_by_image_id,
        category_id_to_name=category_id_to_name,
        skeleton_by_category_id=skeleton_by_category_id,
    )


def _load_intrinsics_from_meta(coco: Dict[str, Any]) -> Optional[Dict[str, float]]:
    meta = coco.get("pose_3d_meta") or {}
    intr = meta.get("intrinsics")
    if not isinstance(intr, dict):
        return None
    required = ("fx", "fy", "cx", "cy")
    if not all(k in intr for k in required):
        return None
    return {k: float(intr[k]) for k in required}


def _decode_rle(segmentation: Any) -> Optional[np.ndarray]:
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

    return flat.reshape((height, width), order="F").astype(bool)


def _color_from_track(track_id: Optional[int]) -> Tuple[int, int, int]:
    if track_id is None:
        return (0, 255, 255)
    seed = int(track_id) * 2654435761 % (2**32)
    rng = np.random.default_rng(seed)
    bgr = (rng.integers(40, 255), rng.integers(40, 255), rng.integers(40, 255))
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_bbox(frame_bgr: np.ndarray, bbox: Sequence[float], color: Tuple[int, int, int]) -> None:
    if not bbox or len(bbox) != 4:
        return
    x, y, w, h = bbox
    x0 = int(round(float(x)))
    y0 = int(round(float(y)))
    x1 = int(round(float(x) + float(w)))
    y1 = int(round(float(y) + float(h)))
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)


def _draw_mask_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    if mask is None or not mask.any():
        return
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = np.zeros_like(frame_bgr, dtype=np.uint8)
    overlay[:, :] = np.array(color, dtype=np.uint8)
    frame_bgr[mask] = cv2.addWeighted(frame_bgr[mask], 1.0 - alpha, overlay[mask], alpha, 0.0)


def _draw_keypoints(
    frame_bgr: np.ndarray,
    keypoints: Sequence[float],
    skeleton: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    if not keypoints or len(keypoints) < 3:
        return
    pts: Dict[int, Tuple[int, int]] = {}
    for i in range(0, len(keypoints), 3):
        idx = i // 3
        x, y, v = keypoints[i : i + 3]
        if v is None or float(v) <= 0:
            continue
        x_i = int(round(float(x)))
        y_i = int(round(float(y)))
        pts[idx] = (x_i, y_i)
        cv2.circle(frame_bgr, (x_i, y_i), 3, color, -1, lineType=cv2.LINE_AA)

    for a, b in skeleton:
        # skeleton indices in COCO are 1-based; keypoint array indices are 0-based
        a0 = a - 1
        b0 = b - 1
        if a0 in pts and b0 in pts:
            cv2.line(frame_bgr, pts[a0], pts[b0], color, 2, lineType=cv2.LINE_AA)


def _label_text(
    anno: Dict[str, Any],
    category_name: str,
    show_bbox3d: bool,
) -> str:
    base = f"{category_name}"
    if "track_id" in anno:
        base += f" t{anno.get('track_id')}"
    if not show_bbox3d:
        return base
    bbox3d = anno.get("bbox3d")
    if not isinstance(bbox3d, dict):
        return base
    stats = bbox3d.get("depth_stats") or {}
    if isinstance(stats, dict) and "median" in stats:
        base += f" z~{stats['median']:.3f}"
    method = anno.get("bbox3d_method")
    if method:
        base += f" ({method})"
    return base


def _put_label(frame_bgr: np.ndarray, x: int, y: int, text: str, color: Tuple[int, int, int]) -> None:
    x = int(np.clip(x, 0, frame_bgr.shape[1] - 1))
    y = int(np.clip(y, 0, frame_bgr.shape[0] - 1))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x0 = x
    y0 = max(y - th - 6, 0)
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + tw + 6, y0 + th + 6), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (x0 + 3, y0 + th + 3), font, scale, color, thickness, cv2.LINE_AA)


def _iter_frame_indices(total: int, start: int, end: int, stride: int) -> Iterable[int]:
    start = max(int(start), 0)
    end = total if end < 0 else min(int(end), total)
    stride = max(int(stride), 1)
    return range(start, end, stride)


def _project_point_xyz_to_uv(
    point_xyz: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
    coord_system: str,
    intrinsics: Optional[Dict[str, float]],
) -> Optional[Tuple[int, int]]:
    x, y, z = map(float, point_xyz)
    if not np.isfinite([x, y, z]).all():
        return None

    if coord_system == "camera" and intrinsics is not None:
        if z <= 1e-8:
            return None
        u = intrinsics["fx"] * (x / z) + intrinsics["cx"]
        v = intrinsics["fy"] * (y / z) + intrinsics["cy"]
    elif coord_system == "ray_camera" and intrinsics is not None:
        u = intrinsics["fx"] * x + intrinsics["cx"]
        v = intrinsics["fy"] * y + intrinsics["cy"]
    else:
        # normalized / ray-normalized fallbacks
        cx = (frame_width - 1) / 2.0
        cy = (frame_height - 1) / 2.0
        denom_x = max((frame_width - 1) / 2.0, 1.0)
        denom_y = max((frame_height - 1) / 2.0, 1.0)
        if coord_system == "ray_normalized":
            u = x * denom_x + cx
            v = y * denom_y + cy
        else:
            # Matches the normalized_camera backprojection:
            # x = (u - cx) / denom_x * z, so u = (x / z) * denom_x + cx
            if z <= 1e-8:
                return None
            u = (x / z) * denom_x + cx
            v = (y / z) * denom_y + cy

    u_i = int(round(u))
    v_i = int(round(v))
    if u_i < 0 or v_i < 0 or u_i >= frame_width or v_i >= frame_height:
        # Allow slightly out-of-frame projections; clip to draw edges reaching border.
        u_i = int(np.clip(u_i, 0, frame_width - 1))
        v_i = int(np.clip(v_i, 0, frame_height - 1))
    return u_i, v_i


def _draw_bbox3d_cuboid(
    frame_bgr: np.ndarray,
    corners_xyz: Sequence[Sequence[float]],
    *,
    coord_system: str,
    intrinsics: Optional[Dict[str, float]],
    color: Tuple[int, int, int],
    thickness: int = 2,
    draw_front_face: bool = False,
    draw_back_face: bool = True,
) -> None:
    if not corners_xyz or len(corners_xyz) != 8:
        return

    projected: List[Optional[Tuple[int, int]]] = []
    for corner in corners_xyz:
        projected.append(
            _project_point_xyz_to_uv(
                corner,
                frame_width=frame_bgr.shape[1],
                frame_height=frame_bgr.shape[0],
                coord_system=coord_system,
                intrinsics=intrinsics,
            )
        )

    edges = []
    if draw_front_face:
        edges.extend([(0, 1), (1, 2), (2, 3), (3, 0)])
    if draw_back_face:
        edges.extend([(4, 5), (5, 6), (6, 7), (7, 4)])
    edges.extend([(0, 4), (1, 5), (2, 6), (3, 7)])
    for a, b in edges:
        pa = projected[a]
        pb = projected[b]
        if pa is None or pb is None:
            continue
        cv2.line(frame_bgr, pa, pb, color, thickness, lineType=cv2.LINE_AA)


def main(
    video_path: str,
    pose3d_json_path: Optional[str] = None,
    *,
    window_name: str = "pose3d_debug",
    start: int = 0,
    end: int = -1,
    stride: int = 1,
    display_scale: float = 1.0,
    show_masks: bool = True,
    show_keypoints: bool = True,
    show_bbox: bool = False,
    show_bbox3d: bool = True,
    show_bbox3d_cuboid: bool = False,
    cuboid_draw_front_face: bool = False,
    cuboid_draw_back_face: bool = False,
    mask_alpha: float = 0.35,
    output_video_path: Optional[str] = None,
    output_fps: float = -1.0,
    quiet: bool = False,
) -> None:
    """
    Visualize COCO pose annotations (and optional bbox3d fields) on top of a video.

    - pose3d_json_path: COCO-style JSON. If it includes `bbox3d`, depth stats are displayed in the label.
    - output_video_path: if set, writes an annotated MP4 (no UI required).
    """
    base_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if pose3d_json_path is None:
        pose3d_json_path = os.path.join(base_dir, f"{base_name}_3d_pose.json")
    if output_video_path is None:
        output_video_path = os.path.join(base_dir, f"{base_name}_3d_pose.mp4")

    with open(pose3d_json_path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    index = _build_index(coco)
    intrinsics = _load_intrinsics_from_meta(coco)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        # Some codecs don't report frame count reliably; fall back to iterating until EOF.
        total_frames = 10**9

    writer = None
    if output_video_path:
        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
        write_fps = native_fps if output_fps is None or output_fps <= 0 else float(output_fps)
        out_w = int(round(frame_w * display_scale))
        out_h = int(round(frame_h * display_scale))
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            write_fps,
            (out_w, out_h),
        )

    if not quiet:
        print(
            "Controls: SPACE=pause/resume, q/ESC=quit | "
            f"frames={total_frames if total_frames < 10**8 else 'unknown'} fps={native_fps:.2f}"
        )

    paused = False
    frame_indices = _iter_frame_indices(total_frames, start, end, stride)

    for frame_idx in frame_indices:
        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                break

        # Map video frame index -> COCO image_id by matching frame_id.
        # This assumes pose JSON uses `images[].frame_id` corresponding to video frame indices.
        image_id = None
        # Fast path: when ids are contiguous and frame_id == id
        if frame_idx in index.images and int(index.images[frame_idx].get("frame_id", frame_idx)) == frame_idx:
            image_id = frame_idx
        else:
            for iid, img in index.images.items():
                if int(img.get("frame_id", iid)) == frame_idx:
                    image_id = iid
                    break

        annos = index.annotations_by_image_id.get(int(image_id), []) if image_id is not None else []

        for anno in annos:
            cat_id = int(anno.get("category_id", -1))
            cat_name = index.category_id_to_name.get(cat_id, f"id={cat_id}")
            color = _color_from_track(anno.get("track_id"))

            if show_masks:
                mask = _decode_rle(anno.get("segmentation"))
                if mask is None and anno.get("segmentation") is not None and mask_utils is None:
                    # Only warn once per run by using a sentinel attribute.
                    if not getattr(main, "_warned_no_pycoco", False):
                        setattr(main, "_warned_no_pycoco", True)
                        if not quiet:
                            print("WARN: pycocotools not installed; masks will be skipped.")
                if mask is not None and mask.shape[:2] != frame_bgr.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                _draw_mask_overlay(frame_bgr, mask, color=color, alpha=mask_alpha)

            if show_bbox:
                _draw_bbox(frame_bgr, anno.get("bbox", []), color=color)

            if show_keypoints and isinstance(anno.get("keypoints"), list) and anno.get("num_keypoints", 0) > 0:
                skeleton = index.skeleton_by_category_id.get(cat_id, [])
                _draw_keypoints(frame_bgr, anno.get("keypoints", []), skeleton=skeleton, color=color)

            bbox = anno.get("bbox", [0, 0, 0, 0])
            x, y = int(round(float(bbox[0]))), int(round(float(bbox[1])))
            _put_label(frame_bgr, x, y, _label_text(anno, cat_name, show_bbox3d=show_bbox3d), color=color)

            if show_bbox3d_cuboid:
                bbox3d = anno.get("bbox3d")
                if isinstance(bbox3d, dict) and isinstance(bbox3d.get("corners"), list):
                    coord_system = str(bbox3d.get("coord_system") or "normalized_camera")
                    _draw_bbox3d_cuboid(
                        frame_bgr,
                        bbox3d.get("corners"),
                        coord_system=coord_system,
                        intrinsics=intrinsics,
                        color=color,
                        draw_front_face=cuboid_draw_front_face,
                        draw_back_face=cuboid_draw_back_face,
                    )

        if display_scale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(round(frame_w * display_scale)), int(round(frame_h * display_scale))),
                interpolation=cv2.INTER_AREA if display_scale < 1.0 else cv2.INTER_LINEAR,
            )

        if writer is not None:
            writer.write(frame_bgr)
        else:
            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
            if key == ord(" "):
                paused = not paused

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Wrote annotated video to {output_video_path}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Fire(main)
