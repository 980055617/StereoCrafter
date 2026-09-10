#!/usr/bin/env python3
"""Profile current 0160 inpainting runtime candidates.

This runner is intentionally narrow: it compares the current origin reference,
full gated/residual Mamba inference, and the reproducible hybrid up-only
partial-replacement inference. It records wall time plus an approximate
nvidia-smi peak for the selected GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "diagnose_0160" / "profile_inpainting_variants_20260618",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--gpu-index", type=int, default=0, help="Physical nvidia-smi GPU index to monitor.")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["origin", "full_gated", "hybrid_up_only"],
        choices=[
            "origin",
            "reference_matched",
            "full_gated",
            "hybrid_up_only",
            "hybrid_up12",
            "hybrid_up13",
            "hybrid_up23",
            "hybrid_exclude_up3_attn1",
        ],
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def config() -> dict[str, Any]:
    return {
        "pre_trained_path": "weights/stable-video-diffusion-img2vid-xt-1-1/",
        "unet_path": "weights/StereoCrafter/",
        "input_video_path": "video_data/splatting/0160_splatting_results.mp4",
        "frames_chunk": "14",
        "overlap": "3",
        "tile_num": "1",
        "origin_tile_num": "2",
        "inference_config": "config/0160_overfit_inference_matched.json",
        "gated_mamba_state": (
            "weights/Overfit0160GatedResidualMamba/MambaCrafter_20260530_100112/"
            "train_state_final_mamba_only.pt"
        ),
    }


def add_fire_args(command: list[str], values: dict[str, str]) -> None:
    for key, value in values.items():
        command.extend([f"--{key}", str(value)])


def build_variants(args: argparse.Namespace) -> dict[str, Variant]:
    cfg = config()
    py = args.python
    origin_save = args.out_dir / "origin"
    reference_matched_save = args.out_dir / "reference_matched"
    full_save = args.out_dir / "full_gated"
    hybrid_save = args.out_dir / "hybrid_up_only"

    origin_cmd = [py, "inpainting_inference_origin_profile.py"]
    add_fire_args(
        origin_cmd,
        {
            "pre_trained_path": cfg["pre_trained_path"],
            "unet_path": cfg["unet_path"],
            "input_video_path": cfg["input_video_path"],
            "save_dir": str(origin_save),
            "frames_chunk": cfg["frames_chunk"],
            "overlap": cfg["overlap"],
            "tile_num": cfg["origin_tile_num"],
            "profile_json": str(args.out_dir / "origin_profile.json"),
        },
    )

    full_cmd = [
        py,
        "inpainting_inference.py",
        "--config",
        cfg["inference_config"],
        "--save_dir",
        str(full_save),
        "--unet_state_path",
        cfg["gated_mamba_state"],
    ]

    hybrid_cmd = [
        py,
        "inpainting_inference_hybrid_up_only.py",
        "--config",
        cfg["inference_config"],
        "--save_dir",
        str(hybrid_save),
        "--unet_state_path",
        cfg["gated_mamba_state"],
    ]
    hybrid_up12_save = args.out_dir / "hybrid_up12"
    hybrid_up12_cmd = [
        py,
        "inpainting_inference_hybrid_up_only.py",
        "--config",
        cfg["inference_config"],
        "--save_dir",
        str(hybrid_up12_save),
        "--unet_state_path",
        cfg["gated_mamba_state"],
        "--include_patterns",
        "up_blocks.1.*,up_blocks.2.*",
    ]
    hybrid_up13_save = args.out_dir / "hybrid_up13"
    hybrid_up13_cmd = [
        py,
        "inpainting_inference_hybrid_up_only.py",
        "--config",
        cfg["inference_config"],
        "--save_dir",
        str(hybrid_up13_save),
        "--unet_state_path",
        cfg["gated_mamba_state"],
        "--include_patterns",
        "up_blocks.1.*,up_blocks.3.*",
    ]
    hybrid_up23_save = args.out_dir / "hybrid_up23"
    hybrid_up23_cmd = [
        py,
        "inpainting_inference_hybrid_up_only.py",
        "--config",
        cfg["inference_config"],
        "--save_dir",
        str(hybrid_up23_save),
        "--unet_state_path",
        cfg["gated_mamba_state"],
        "--include_patterns",
        "up_blocks.2.*,up_blocks.3.*",
    ]
    hybrid_exclude_up3_attn1_save = args.out_dir / "hybrid_exclude_up3_attn1"
    hybrid_exclude_up3_attn1_cmd = [
        py,
        "inpainting_inference_hybrid_exclude_up3_attn1.py",
        "--config",
        cfg["inference_config"],
        "--save_dir",
        str(hybrid_exclude_up3_attn1_save),
        "--unet_state_path",
        cfg["gated_mamba_state"],
    ]

    return {
        "origin": Variant(
            name="origin",
            command=origin_cmd,
            notes=(
                "Origin reference path. It uses the origin script's native crop, "
                "not the Mamba matched 576x1024 crop."
            ),
        ),
        "reference_matched": Variant(
            name="reference_matched",
            command=[
                py,
                "scripts/run_reference_matched_inpainting.py",
                "--pre_trained_path",
                cfg["pre_trained_path"],
                "--unet_path",
                cfg["unet_path"],
                "--input_video_path",
                cfg["input_video_path"],
                "--save_dir",
                str(reference_matched_save),
                "--frames_chunk",
                cfg["frames_chunk"],
                "--overlap",
                cfg["overlap"],
                "--tile_num",
                cfg["tile_num"],
                "--target_height",
                "576",
                "--target_width",
                "1024",
            ],
            notes=(
                "Reference StereoCrafter pipeline through a non-reference matched-crop "
                "adapter, outputting the same 2048x576 SBS size as the Mamba candidates."
            ),
        ),
        "full_gated": Variant(
            name="full_gated",
            command=full_cmd,
            notes="Full gated/residual exported Mamba-only checkpoint.",
        ),
        "hybrid_up_only": Variant(
            name="hybrid_up_only",
            command=hybrid_cmd,
            notes=(
                "Hybrid partial replacement: use learned Mamba only for up_blocks.*, "
                "leave non-up attn1 as origin attention."
            ),
        ),
        "hybrid_up12": Variant(
            name="hybrid_up12",
            command=hybrid_up12_cmd,
            notes=(
                "Hybrid partial replacement without up_blocks.3.*: use learned Mamba "
                "for up_blocks.1.* and up_blocks.2.*, leave up_blocks.3.* and non-up "
                "attn1 as origin attention."
            ),
        ),
        "hybrid_up13": Variant(
            name="hybrid_up13",
            command=hybrid_up13_cmd,
            notes=(
                "Hybrid partial replacement without up_blocks.2.*: use learned Mamba "
                "for up_blocks.1.* and up_blocks.3.*, leave up_blocks.2.* and non-up "
                "attn1 as origin attention."
            ),
        ),
        "hybrid_up23": Variant(
            name="hybrid_up23",
            command=hybrid_up23_cmd,
            notes=(
                "Hybrid partial replacement without up_blocks.1.*: use learned Mamba "
                "for up_blocks.2.* and up_blocks.3.*, leave up_blocks.1.* and non-up "
                "attn1 as origin attention."
            ),
        ),
        "hybrid_exclude_up3_attn1": Variant(
            name="hybrid_exclude_up3_attn1",
            command=hybrid_exclude_up3_attn1_cmd,
            notes=(
                "Current best selective hybrid candidate: use learned Mamba for "
                "up_blocks.* except up_blocks.3.attentions.1.*, which stays on "
                "the reference attention path."
            ),
        ),
    }


def query_gpu_used_mib(gpu_index: int) -> int | None:
    cmd = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    first = out.splitlines()[0].strip() if out else ""
    try:
        return int(first)
    except ValueError:
        return None


def monitor_gpu(stop: threading.Event, gpu_index: int, interval: float, samples: list[dict[str, Any]]) -> None:
    while not stop.is_set():
        used = query_gpu_used_mib(gpu_index)
        samples.append({"time": datetime.now().isoformat(), "usedMiB": used})
        stop.wait(interval)


def run_variant(args: argparse.Namespace, variant: Variant) -> dict[str, Any]:
    run_dir = args.out_dir / variant.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"{variant.name}.log"
    samples_path = args.out_dir / f"{variant.name}_gpu_memory.csv"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.update(variant.env)

    baseline_used = query_gpu_used_mib(args.gpu_index)
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(stop, args.gpu_index, float(args.sample_interval), samples),
        daemon=True,
    )

    started_at = datetime.now().isoformat()
    start = time.perf_counter()
    monitor.start()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[PROFILE] start {started_at}\n")
        log_file.write("[PROFILE] command: " + " ".join(variant.command) + "\n")
        log_file.write("[PROFILE] env overrides: " + json.dumps(variant.env, sort_keys=True) + "\n\n")
        log_file.flush()
        result = subprocess.run(
            variant.command,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    stop.set()
    monitor.join(timeout=2.0)
    finished_at = datetime.now().isoformat()

    with samples_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["time", "usedMiB"])
        writer.writeheader()
        writer.writerows(samples)

    used_values = [sample["usedMiB"] for sample in samples if sample["usedMiB"] is not None]
    peak_used = max(used_values) if used_values else None
    peak_increment = None
    if peak_used is not None and baseline_used is not None:
        peak_increment = peak_used - baseline_used

    return {
        "name": variant.name,
        "status": "completed" if result.returncode == 0 else "failed",
        "returnCode": result.returncode,
        "seconds": elapsed,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "baselineGpuUsedMiB": baseline_used,
        "peakGpuUsedMiB": peak_used,
        "peakGpuIncrementMiB": peak_increment,
        "logPath": str(log_path),
        "gpuSamplesPath": str(samples_path),
        "saveDir": str(run_dir),
        "command": variant.command,
        "env": variant.env,
        "notes": variant.notes,
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "profile_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "profile_summary.csv"
    fieldnames = [
        "name",
        "status",
        "returnCode",
        "seconds",
        "baselineGpuUsedMiB",
        "peakGpuUsedMiB",
        "peakGpuIncrementMiB",
        "logPath",
        "saveDir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variants(args)
    payload: dict[str, Any] = {
        "schema": "master_project.stereocrafter.profile_0160_inpainting_variants.v1",
        "createdAt": datetime.now().isoformat(),
        "settings": {
            "gpu": args.gpu,
            "gpuIndex": args.gpu_index,
            "sampleInterval": args.sample_interval,
            "python": args.python,
            "variants": args.variants,
            "config": config(),
        },
        "results": [],
    }
    write_outputs(args.out_dir, payload)

    for name in args.variants:
        variant = variants[name]
        print(f"[PROFILE] running {name}")
        record = run_variant(args, variant)
        payload["results"].append(record)
        write_outputs(args.out_dir, payload)
        print(
            f"[PROFILE] {name}: {record['seconds']:.1f}s "
            f"peak={record['peakGpuUsedMiB']}MiB status={record['status']}"
        )
        if record["returnCode"] != 0 and not args.continue_on_error:
            raise SystemExit(record["returnCode"])

    print(f"[PROFILE] summary: {args.out_dir / 'profile_summary.json'}")


if __name__ == "__main__":
    main()
