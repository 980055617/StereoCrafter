# =============================================
# File: /workspace/stereocraft/scripts/reconstruct_splatting_from_depth_video.py
# ---------------------------------------------
# 目的: 深度から2x2スプラッティング復元
# =============================================

import os
from typing import Optional

import cv2
import numpy as np
import torch
from fire import Fire

from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth
from utils.pose3d_export import export_pose_annotations_3d, decode_coco_rle
from depth_splatting_inference import ForwardWarpStereo, _to_float32_unit_range


def _load_depth_array(depth_path: str) -> np.ndarray:
    if depth_path.endswith(".npz"):
        payload = np.load(depth_path)
        if "depth" not in payload:
            raise KeyError(f"{depth_path} does not contain key 'depth'")
        return np.asarray(payload["depth"])
    return np.asarray(np.load(depth_path))


def _build_frame_union_masks(
    pose_annotations_path: str,
    target_height: int,
    target_width: int,
) -> dict[int, np.ndarray]:
    with open(pose_annotations_path, "r", encoding="utf-8") as handle:
        coco = __import__("json").load(handle)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    image_id_to_frame = {int(img["id"]): int(img.get("frame_id", img["id"])) for img in images}

    frame_masks: dict[int, np.ndarray] = {}
    for anno in annotations:
        image_id = int(anno.get("image_id"))
        frame_id = image_id_to_frame.get(image_id)
        if frame_id is None:
            continue
        mask = decode_coco_rle(anno.get("segmentation"))
        if mask is None:
            continue
        bbox = anno.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
            x0 = max(int(x), 0)
            y0 = max(int(y), 0)
            x1 = min(int(x + w), mask.shape[1])
            y1 = min(int(y + h), mask.shape[0])
            cropped = np.zeros_like(mask, dtype=bool)
            cropped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            mask = cropped
        if mask.shape != (target_height, target_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if frame_id not in frame_masks:
            frame_masks[frame_id] = mask
        else:
            frame_masks[frame_id] |= mask
    return frame_masks


def reconstruct_2x2(
    left_video_path: str,
    depth_path: Optional[str],
    output_2x2_video: Optional[str],
    *,
    max_disp: float = 20.0,
    batch_size: int = 8,
    pose_annotations_path: Optional[str] = None,
    pose_3d_output_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    ema_depth_alpha: float = 0.0,
    ema_bbox_alpha: float = 0.0,
    depth_only: bool = False,
    debug: bool = False,
) -> str:
    if debug:
        import logging

        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
        )

    cap = cv2.VideoCapture(left_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {left_video_path}")

    base_dir = os.path.dirname(left_video_path)
    base_name = os.path.splitext(os.path.basename(left_video_path))[0]

    if depth_path is None:
        depth_path = os.path.join(base_dir, f"{base_name}_depth.npz")
    depth_array = _load_depth_array(depth_path)
    if ema_depth_alpha > 0.0:
        ema_depth_alpha = float(np.clip(ema_depth_alpha, 0.0, 1.0))
        smoothed = np.empty_like(depth_array, dtype=np.float32)
        prev = None
        for i, d in enumerate(depth_array):
            d = d.astype(np.float32, copy=False)
            if prev is None:
                prev = d
            else:
                prev = ema_depth_alpha * d + (1.0 - ema_depth_alpha) * prev
            smoothed[i] = prev
        depth_array = smoothed
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ret, first = cap.read()
    if not ret:
        raise ValueError("Failed to read first frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    h, w = first.shape[:2]
    if output_2x2_video is None:
        output_2x2_video = os.path.join(base_dir, f"{base_name}_2x2_video.mp4")
    writer = cv2.VideoWriter(
        output_2x2_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h * 2),
    )

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()
    left_frames = []
    depth_list = []
    frame_ids: list[int] = []
    frame_union_masks: dict[int, np.ndarray] = {}
    if not depth_only:
        if pose_annotations_path is None:
            pose_annotations_path = os.path.join(base_dir, f"{base_name}_pose_annotations.json")
        if os.path.exists(pose_annotations_path):
            frame_union_masks = _build_frame_union_masks(
                pose_annotations_path,
                target_height=h,
                target_width=w,
            )

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(depth_array):
            break
        left_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_list.append(depth_array[idx])
        frame_ids.append(idx)
        idx += 1

        if len(left_frames) >= batch_size or idx == frame_count or idx == len(depth_array):
            batch_frames = _to_float32_unit_range(np.asarray(left_frames))
            batch_depth = np.asarray(depth_list, dtype=np.float32)

            left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
            disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()
            disp_map = disp_map * 2.0 - 1.0
            disp_map = disp_map * max_disp

            with torch.no_grad():
                right_video, occlusion_mask = stereo_projector(left_video, disp_map)

            right_video = right_video.cpu().permute(0, 2, 3, 1).numpy()
            occlusion_mask = (
                occlusion_mask.cpu().permute(0, 2, 3, 1).numpy().repeat(3, axis=-1)
            )
            np.clip(right_video, 0.0, 1.0, out=right_video)
            np.clip(occlusion_mask, 0.0, 1.0, out=occlusion_mask)

            depth_vis = vis_sequence_depth(batch_depth)
            depth_vis = np.multiply(depth_vis, 255.0, out=depth_vis).astype(np.uint8)
            left_frames_uint8 = np.multiply(batch_frames, 255.0, out=batch_frames).astype(np.uint8)
            right_video_uint8 = np.multiply(right_video, 255.0, out=right_video).astype(np.uint8)
            occlusion_mask_uint8 = np.multiply(occlusion_mask, 255.0, out=occlusion_mask).astype(np.uint8)

            for j in range(len(left_frames_uint8)):
                if frame_union_masks and j < len(frame_ids):
                    fid = frame_ids[j]
                    fm = frame_union_masks.get(fid)
                    if fm is not None:
                        if fm.shape != occlusion_mask_uint8[j].shape[:2]:
                            fm = cv2.resize(
                                fm.astype(np.uint8),
                                (occlusion_mask_uint8[j].shape[1], occlusion_mask_uint8[j].shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        # Left-bottom: occlusion mask in white
                        occlusion_mask_uint8[j][fm] = 255
                        # Right-bottom: black-out the masked region in right view
                        right_video_uint8[j][fm] = 0

                top = np.concatenate([left_frames_uint8[j], depth_vis[j]], axis=1)
                bottom = np.concatenate([occlusion_mask_uint8[j], right_video_uint8[j]], axis=1)
                grid = np.concatenate([top, bottom], axis=0)
                grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
                writer.write(grid_bgr)

            left_frames.clear()
            depth_list.clear()
            frame_ids.clear()
            torch.cuda.empty_cache()
    cap.release()
    writer.release()

    if not depth_only:
        if pose_3d_output_path is None:
            pose_3d_output_path = os.path.join(base_dir, f"{base_name}_3d_pose.json")

        export_pose_annotations_3d(
            pose_annotations_path,
            depth_array,
            pose_3d_output_path,
            intrinsics_path=intrinsics_path,
            ema_bbox_alpha=ema_bbox_alpha,
            debug=debug,
        )

    return output_2x2_video


def main(
    left_video_path: str,
    depth_path: Optional[str] = None,
    output_2x2_video: Optional[str] = None,
    *,
    max_disp: float = 20.0,
    batch_size: int = 8,
    pose_annotations_path: Optional[str] = None,
    pose_3d_output_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    ema_depth_alpha: float = 0.0,
    ema_bbox_alpha: float = 0.0,
    depth_only: bool = False,
    debug: bool = False,
) -> str:
    return reconstruct_2x2(
        left_video_path=left_video_path,
        depth_path=depth_path,
        output_2x2_video=output_2x2_video,
        max_disp=max_disp,
        batch_size=batch_size,
        pose_annotations_path=pose_annotations_path,
        pose_3d_output_path=pose_3d_output_path,
        intrinsics_path=intrinsics_path,
        ema_depth_alpha=ema_depth_alpha,
        ema_bbox_alpha=ema_bbox_alpha,
        depth_only=depth_only,
        debug=debug,
    )


if __name__ == "__main__":
    Fire(main)
