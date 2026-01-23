#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
from typing import Dict, Optional

import numpy as np

import depth_splatting_inference as depth_splatting_new
import depth_splatting_inference_origin as depth_splatting_origin

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)




def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _compare_arrays(
    array_a: np.ndarray,
    array_b: np.ndarray,
) -> Dict[str, float]:
    if array_a.shape != array_b.shape:
        raise ValueError(f"Depth shape mismatch: {array_a.shape} vs {array_b.shape}")
    diff = array_a.astype(np.float32) - array_b.astype(np.float32)
    return {
        "mse_mean": float(np.mean(diff ** 2)),
        "mae_mean": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def _npz_path_for_video(output_video_path: str) -> str:
    base_dir = os.path.dirname(output_video_path)
    base_name = os.path.splitext(os.path.basename(output_video_path))[0]
    return os.path.join(base_dir, f"{base_name}.npz")


def _maybe_empty_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run(
    input_video_path: str,
    output_dir: Optional[str] = None,
    metrics_path: Optional[str] = None,
) -> None:
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_root = temp_dir.name
    else:
        output_root = output_dir
        _ensure_dir(output_root)

    origin_out = os.path.join(output_root, f"{base_name}_origin.mp4")
    new_out = os.path.join(output_root, f"{base_name}_new.mp4")

    origin_demo = depth_splatting_origin.DepthCrafterDemo(
        unet_path="./weights/DepthCrafter",
        pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
    )
    origin_depth, _ = origin_demo.infer(
        input_video_path=input_video_path,
        output_video_path=origin_out,
        save_depth=True,
    )
    _maybe_empty_cache()

    new_demo = depth_splatting_new.DepthCrafterDemo(
        unet_path="./weights/DepthCrafter",
        pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
    )
    new_depth, _ = new_demo.infer(
        input_video_path=input_video_path,
        output_video_path=new_out,
        save_depth=True,
    )
    origin_npz = _npz_path_for_video(origin_out)
    new_npz = _npz_path_for_video(new_out)
    depth_metrics = _compare_arrays(
        np.load(origin_npz)["depth"],
        np.load(new_npz)["depth"],
    )
    del origin_depth, new_depth
    _maybe_empty_cache()

    report = {
        "depth_metrics": depth_metrics,
        "paths": {"origin": origin_out, "new": new_out},
    }
    if metrics_path:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    if temp_dir is not None:
        temp_dir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run both depth_splatting_inference scripts and compare outputs."
    )
    parser.add_argument("--input_video_path", default="./video_data/test/test.mp4")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--metrics_path", default=None)

    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
