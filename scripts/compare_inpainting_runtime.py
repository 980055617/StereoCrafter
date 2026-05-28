#!/usr/bin/env python3
"""Compare wall-clock runtime for StereoCrafter inpainting entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two StereoCrafter inpainting scripts on the same video and record "
            "their elapsed wall-clock time."
        )
    )
    parser.add_argument("input_video", type=Path, help="Input 2x2/splatting video.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional inpainting_inference.py JSON config. Common settings are also "
            "applied to origin; Mamba/model-specific settings are passed only to "
            "inpainting_inference.py."
        ),
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=["inpainting_inference.py", "inpainting_inference_origin.py"],
        help=(
            "Scripts to compare, relative to the StereoCrafter root unless absolute. "
            "Default: inpainting_inference.py inpainting_inference_origin.py"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "inpainting_runtime_compare",
        help="Directory for per-script outputs, logs, and summary JSON.",
    )
    parser.add_argument(
        "--pre-trained-path",
        default="./weights/stable-video-diffusion-img2vid-xt-1-1",
        help="Value passed as --pre_trained_path.",
    )
    parser.add_argument(
        "--unet-path",
        default="./weights/StereoCrafter",
        help="Value passed as --unet_path.",
    )
    parser.add_argument("--frames-chunk", type=int, default=23)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--tile-num", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument(
        "--gpu",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for each run.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the scripts. Default: current Python.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Run remaining scripts even if one fails.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help=(
            "Extra raw argument appended to every command. Can be repeated, e.g. "
            "--extra-arg=--decode_chunk_size --extra-arg=2. Use only args accepted "
            "by every compared script."
        ),
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return data


def apply_config_defaults(args: argparse.Namespace, config: dict) -> None:
    mapping = {
        "pre_trained_path": "pre_trained_path",
        "unet_path": "unet_path",
        "frames_chunk": "frames_chunk",
        "overlap": "overlap",
        "tile_num": "tile_num",
        "num_inference_steps": "num_inference_steps",
    }
    for attr, key in mapping.items():
        if key in config and config[key] is not None:
            setattr(args, attr, config[key])


def script_path(script: str) -> Path:
    path = Path(script)
    return path if path.is_absolute() else ROOT / path


def result_name(script: Path) -> str:
    return script.stem.replace("inpainting_inference", "inpainting").strip("_") or script.stem


def add_arg(command: list[str], name: str, value) -> None:
    if value is None:
        return
    command.extend([name, str(value)])


def build_command(args: argparse.Namespace, script: Path, save_dir: Path, config: dict) -> list[str]:
    command = [
        args.python,
        str(script),
        "--pre_trained_path",
        args.pre_trained_path,
        "--unet_path",
        args.unet_path,
        "--input_video_path",
        str(args.input_video),
        "--save_dir",
        str(save_dir),
        "--frames_chunk",
        str(args.frames_chunk),
        "--overlap",
        str(args.overlap),
        "--tile_num",
        str(args.tile_num),
    ]
    if script.name != "inpainting_inference_origin.py":
        command.extend(["--num_inference_steps", str(args.num_inference_steps)])
    if script.name == "inpainting_inference.py":
        add_arg(command, "--precision", config.get("precision"))
        add_arg(command, "--decode_chunk_size", config.get("decode_chunk_size"))
        add_arg(command, "--unet_state_path", config.get("unet_state_path"))
        add_arg(command, "--noise_seed", config.get("noise_seed"))
        add_arg(command, "--min_guidance_scale", config.get("min_guidance_scale"))
        add_arg(command, "--max_guidance_scale", config.get("max_guidance_scale"))
        add_arg(command, "--target_height", config.get("target_height"))
        add_arg(command, "--target_width", config.get("target_width"))
        add_arg(command, "--overlap_prev_weight", config.get("overlap_prev_weight"))
    command.extend(args.extra_arg)
    return command


def run_one(args: argparse.Namespace, script: Path, config: dict) -> dict:
    name = result_name(script)
    save_dir = args.out_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"{name}.log"
    command = build_command(args, script, save_dir, config)
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    started_at = datetime.now().isoformat()
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[COMPARE] start {started_at}\n")
        log_file.write("[COMPARE] command: " + " ".join(command) + "\n\n")
        log_file.flush()
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - start
    finished_at = datetime.now().isoformat()

    record = {
        "name": name,
        "script": str(script),
        "status": "completed" if result.returncode == 0 else "failed",
        "returnCode": result.returncode,
        "seconds": elapsed,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "saveDir": str(save_dir),
        "logPath": str(log_path),
        "command": command,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[COMPARE] finished {finished_at}\n")
        log_file.write(f"[COMPARE] seconds {elapsed:.3f}\n")
        log_file.write(f"[COMPARE] returnCode {result.returncode}\n")
    return record


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_config_defaults(args, config)
    args.input_video = args.input_video.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_video.exists():
        raise FileNotFoundError(f"input video not found: {args.input_video}")

    scripts = [script_path(script).resolve() for script in args.scripts]
    missing = [str(path) for path in scripts if not path.exists()]
    if missing:
        raise FileNotFoundError("script not found: " + ", ".join(missing))
    if any(path.name == "inpainting_inference_origin.py" for path in scripts):
        if int(args.num_inference_steps) != 8:
            raise ValueError(
                "inpainting_inference_origin.py hard-codes num_inference_steps=8. "
                "Use --num-inference-steps 8, or compare scripts that expose this option."
            )

    summary_path = args.out_dir / "runtime_summary.json"
    payload = {
        "schema": "master_project.stereocrafter.inpainting_runtime_compare.v1",
        "inputVideo": str(args.input_video),
        "createdAt": datetime.now().isoformat(),
        "settings": {
            "preTrainedPath": args.pre_trained_path,
            "unetPath": args.unet_path,
            "framesChunk": args.frames_chunk,
            "overlap": args.overlap,
            "tileNum": args.tile_num,
            "numInferenceSteps": args.num_inference_steps,
            "gpu": args.gpu,
            "extraArg": args.extra_arg,
            "config": str(args.config.resolve()) if args.config else None,
        },
        "results": [],
    }
    write_summary(summary_path, payload)

    for script in scripts:
        print(f"[COMPARE] running {script.name}")
        record = run_one(args, script, config)
        payload["results"].append(record)
        write_summary(summary_path, payload)
        print(
            f"[COMPARE] {record['name']}: {record['seconds']:.2f}s "
            f"status={record['status']} log={record['logPath']}"
        )
        if record["returnCode"] != 0 and not args.continue_on_error:
            raise SystemExit(record["returnCode"])

    print(f"[COMPARE] summary: {summary_path}")


if __name__ == "__main__":
    main()
