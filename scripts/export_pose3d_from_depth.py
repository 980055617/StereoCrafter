import os
from typing import Optional

import numpy as np
from fire import Fire

from utils.pose3d_export import export_pose_annotations_3d


def _load_depth(depth_path: str) -> np.ndarray:
    if depth_path.endswith(".npz"):
        payload = np.load(depth_path)
        if "depth" not in payload:
            raise KeyError(f"{depth_path} does not contain key 'depth'")
        return np.asarray(payload["depth"])
    return np.asarray(np.load(depth_path))


def main(
    pose_annotations_path: str,
    depth_path: str,
    *,
    output_path: Optional[str] = None,
    intrinsics_path: Optional[str] = None,
    keypoint_depth_radius: int = 2,
    min_keypoints_for_3d: int = 4,
    prefer_mask_pointcloud: bool = True,
    mask_pointcloud_max_points: int = 8000,
    bbox3d_space: str = "xyz",
) -> str:
    """
    Export pseudo-3D cuboid bboxes (bbox3d) from a depth map + COCO pose annotations.

    Typical flow:
    1) Run depth_splatting_inference.py with --save_depth True to produce a .npz depth file.
    2) Run this script with that depth .npz and pose_annotations.json.

    depth_path supports:
    - .npz with key 'depth' (DepthCrafter save format in this repo)
    - .npy containing [T,H,W]
    """
    depth = _load_depth(depth_path)
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape [T,H,W], got {depth.shape}")

    if output_path is None:
        base = os.path.splitext(os.path.basename(depth_path))[0]
        output_path = os.path.join(os.path.dirname(depth_path) or ".", base + "_pose3d.json")

    return export_pose_annotations_3d(
        pose_annotations_path,
        depth,
        output_path,
        intrinsics_path=intrinsics_path,
        keypoint_depth_radius=keypoint_depth_radius,
        min_keypoints_for_3d=min_keypoints_for_3d,
        prefer_mask_pointcloud=prefer_mask_pointcloud,
        mask_pointcloud_max_points=mask_pointcloud_max_points,
        bbox3d_space=bbox3d_space,
    )


if __name__ == "__main__":
    Fire(main)
