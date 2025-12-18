"""Batch runner to inpaint the held-out split and save videos.

Usage example:
  python scripts/batch_inference_origin.py --config config/train.json
    --pre_trained_path weights/stable-video-diffusion-img2vid-xt-1-1/
    --unet_path weights/StereoCrafter/
    --output_dir video_data/model_test_output/origin

データ分割:
- train.json の dataset_split_ratios / dataset_split_seed を使い、train_glob を分割して
  dataset_split_group (デフォルト: "test") の動画だけを処理します。
"""

import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import gc

import torch
from decord import VideoReader, cpu
from fire import Fire

from inpainting_inference_origin import main as run_single_inference


def _load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object.")
    return data


def _split_video_paths(
    all_video_paths: list[str],
    ratios: Sequence[float] | None,
    seed: int,
) -> dict[str, list[str]]:
    split_map: dict[str, list[str]] = {k: [] for k in ("train", "val", "test")}
    if not ratios:
        split_map["train"] = list(all_video_paths)
        return split_map

    ratios = [float(v) for v in ratios]
    if len(ratios) != 3:
        raise ValueError("dataset_split_ratios must contain exactly three values: [train, val, test].")
    if any(v < 0 for v in ratios):
        raise ValueError("dataset_split_ratios cannot contain negative values.")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("dataset_split_ratios must sum to a positive value.")

    shuffled = list(all_video_paths)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    normalized = [r / ratio_sum for r in ratios]
    raw_counts = [n * total for n in normalized]
    counts = [math.floor(v) for v in raw_counts]
    remainder = total - sum(counts)
    if remainder > 0:
        fractional_order = sorted(
            range(len(raw_counts)),
            key=lambda idx: (raw_counts[idx] - counts[idx]),
            reverse=True,
        )
        for idx in fractional_order[:remainder]:
            counts[idx] += 1

    keys = ("train", "val", "test")
    cursor = 0
    for key, count in zip(keys, counts):
        if count > 0:
            split_map[key] = shuffled[cursor : cursor + count]
        cursor += count
    return split_map


def run(
    config: str = "config/train.json",
    pre_trained_path: str | None = None,
    unet_path: str | None = None,
    output_dir: str = "video_data/model_test_output/origin",
    dataset_split_group: str = "test",
    dataset_split_seed: int | None = None,
    dataset_split_ratios: Sequence[float] | None = None,
    frames_chunk: int = 23,
    overlap: int = 3,
    tile_num: int = 1,
    decode_chunk_size: int = 2,
    auto_tile: bool = True,
) -> None:
    cfg = _load_config(config)
    train_glob = cfg.get("train_glob", "/workspace/stereocraft/video_data/train/*.mp4")
    ratios = dataset_split_ratios if dataset_split_ratios is not None else cfg.get("dataset_split_ratios", None)
    seed = dataset_split_seed if dataset_split_seed is not None else int(cfg.get("dataset_split_seed", 7))
    split_group = (dataset_split_group or cfg.get("dataset_split_group", "test") or "test").strip().lower()

    pre_trained = pre_trained_path or cfg.get("pre_trained_path")
    unet = unet_path or cfg.get("unet_path")
    if not pre_trained or not unet:
        raise ValueError("pre_trained_path and unet_path must be provided (either via args or config).")

    all_videos = sorted(glob.glob(train_glob))
    if not all_videos:
        raise FileNotFoundError(f"No videos found for pattern: {train_glob}")

    split_map = _split_video_paths(all_videos, ratios, seed)
    if split_group not in split_map:
        raise ValueError(f"Unknown dataset_split_group '{split_group}'. Expected one of train/val/test.")

    target_videos = split_map[split_group]
    if not target_videos:
        raise ValueError(f"No videos in split '{split_group}'. Check ratios/seed.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Running inference on {len(target_videos)} videos in split '{split_group}' -> {output_dir}")

    for idx, video_path in enumerate(target_videos, start=1):
        stem = Path(video_path).stem
        tile_use = tile_num
        if auto_tile:
            vr = VideoReader(video_path, ctx=cpu(0))
            if len(vr) == 0:
                print(f"[WARN] No frames in {video_path}, skipping.")
                continue
            h, w, _ = vr[0].shape
            if max(h, w) >= 4000:
                tile_use = 4
            elif max(h, w) >= 2100:
                tile_use = 2
            else:
                tile_use = max(1, tile_num)
            print(f"[{idx}/{len(target_videos)}] {stem} (auto tile_num={tile_use}, res={h}x{w})")
        else:
            print(f"[{idx}/{len(target_videos)}] {stem} (tile_num={tile_use})")
        out_path = Path(output_dir) / f"{(stem[:-6] if stem.endswith('_train') else stem)}.mp4"
        if out_path.exists():
            print(f"  [SKIP] exists: {out_path}")
            continue
        run_single_inference(
            pre_trained_path=pre_trained,
            unet_path=unet,
            input_video_path=video_path,
            save_dir=output_dir,
            frames_chunk=frames_chunk,
            overlap=overlap,
            tile_num=tile_use,
            decode_chunk_size=decode_chunk_size,
        )
        # 明示的に解放して次の動画でのフラグメンテーションを抑制
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    Fire(run)
