import json
import os
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
from fire import Fire

import torch
from decord import VideoReader, cpu

from transformers import CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKLTemporalDecoder,
)
from diffusers import UNetSpatioTemporalConditionModel

from utils.training_batches import chunk_frame_ranges
from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[
        :, :, :, -overlap_size:
    ] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[
        :, :, -overlap_size:, :
    ] + weight_b * b[:, :, :overlap_size, :]
    return b


def _parse_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _parse_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def parse_frame_rate(value) -> float:
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


def _run_ffprobe_json(cmd):
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found. Install ffmpeg/ffprobe to continue.")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffprobe failed: {' '.join(cmd)}\n{stderr}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to parse ffprobe output as JSON.") from exc


def ffprobe_count_frames(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,r_frame_rate,duration,time_base",
        "-of",
        "json",
        path,
    ]
    data = _run_ffprobe_json(cmd)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no streams for {path}")
    stream = streams[0]
    return {
        "nb_read_frames": _parse_int(stream.get("nb_read_frames")),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration": _parse_float(stream.get("duration")),
        "time_base": stream.get("time_base"),
    }


def ffprobe_format_info(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        path,
    ]
    data = _run_ffprobe_json(cmd)
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    return {
        "format_duration": _parse_float(fmt.get("duration")),
        "has_audio": has_audio,
    }


def log_ffprobe(label: str, count_info: dict, format_info: dict) -> None:
    print(
        f"[ffprobe] {label}: nb_read_frames={count_info.get('nb_read_frames')} "
        f"avg_frame_rate={count_info.get('avg_frame_rate')} "
        f"r_frame_rate={count_info.get('r_frame_rate')} "
        f"duration={count_info.get('duration')} time_base={count_info.get('time_base')} "
        f"format_duration={format_info.get('format_duration')} "
        f"has_audio={format_info.get('has_audio')}"
    )


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=8,
    **kargs,
):
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    tile_size = (
        int((height + tile_overlap[0] *  (tile_num - 1)) / tile_num), 
        int((width  + tile_overlap[1] * (tile_num - 1)) / tile_num)
    )
    tile_stride = (
        (tile_size[0] - tile_overlap[0]), 
        (tile_size[1] - tile_overlap[1])
        )
    
    cols = []
    for i in range(0, tile_num):
        rows = []
        for j in range(0, tile_num):

            cond_tile = cond_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]
            mask_tile = mask_frames[
                :,
                :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]

            tile = process_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs,
            ).frames[0]

            rows.append(tile)
        cols.append(rows)

    latent_stride = (
        tile_stride[0] // spatial_n_compress,
        tile_stride[1] // spatial_n_compress,
    )
    latent_overlap = (
        tile_overlap[0] // spatial_n_compress,
        tile_overlap[1] // spatial_n_compress,
    )

    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    x = torch.cat(pixels, dim=2)
    return x


def write_video_opencv(input_frames, fps, output_video_path):

    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    out = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, 
        (width, height)
    )

    for i in range(num_frames):
        out.write(input_frames[i, :, :, ::-1])

    out.release()



def main(
    input_video_path,
    pre_trained_path="./weights/stable-video-diffusion-img2vid-xt-1-1",
    unet_path="./weights/StereoCrafter",
    num_inference_steps=8,
    save_dir=None,
    frames_chunk=23,
    overlap=3,
    tile_num=1,
):
    pre_trained_path = pre_trained_path or "./weights/stable-video-diffusion-img2vid-xt-1-1"
    unet_path = unet_path or "./weights/StereoCrafter"
    save_dir = save_dir or str(Path(input_video_path).parent)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch.float16
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, 
        subfolder="vae", 
        variant="fp16", 
        torch_dtype=torch.float16
    )

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        # variant="fp16",
        torch_dtype=torch.float16
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to("cuda")

    os.makedirs(save_dir, exist_ok=True)
    video_name = input_video_path.split("/")[-1].replace(".mp4", "").replace("_splatting_results", "") + "_inpainting_results"

    input_count = ffprobe_count_frames(input_video_path)
    input_format = ffprobe_format_info(input_video_path)
    log_ffprobe("input", input_count, input_format)

    input_nb_frames = int(input_count.get("nb_read_frames") or 0)
    if input_nb_frames <= 0:
        raise ValueError(f"ffprobe did not report readable frames for {input_video_path}")

    video_reader = VideoReader(input_video_path, ctx=cpu(0))
    fps = float(video_reader.get_avg_fps() or 0.0)
    num_frames = len(video_reader)
    if num_frames == 0:
        raise ValueError(f"No frames found in video: {input_video_path}")
    if num_frames != input_nb_frames:
        raise ValueError(
            f"Input frame count mismatch: decord={num_frames} ffprobe={input_nb_frames}"
        )
    if fps <= 0:
        fps = parse_frame_rate(input_count.get("avg_frame_rate"))
        if fps <= 0:
            fps = parse_frame_rate(input_count.get("r_frame_rate"))
    if fps <= 0:
        raise ValueError("Failed to determine FPS from input video.")

    first = video_reader[0].asnumpy()
    height_raw, width_raw = first.shape[0] // 2, first.shape[1] // 2
    height = height_raw // 128 * 128
    width = width_raw // 128 * 128

    def _load_chunk(start: int, end: int):
        indices = list(range(start, end))
        batch = video_reader.get_batch(indices).asnumpy()  # [T,H,W,C]
        frames = torch.tensor(batch).permute(0, 3, 1, 2).float()  # [T,C,H,W]
        left = frames[:, :, :height_raw, :width_raw]
        mask = frames[:, :, height_raw:, :width_raw]
        warped = frames[:, :, height_raw:, width_raw:]

        left = left[:, :, :height, :width] / 255.0
        warped = warped[:, :, :height, :width] / 255.0
        mask = mask[:, :, :height, :width] / 255.0
        mask = mask.mean(dim=1, keepdim=True)
        return left, warped, mask

    step = max(frames_chunk - overlap, 1)
    frame_ranges = list(chunk_frame_ranges(num_frames, frames_chunk, overlap)) if frames_chunk > 0 else [(0, num_frames)]

    generated_prev = None
    stem = Path(input_video_path).stem
    # "_train" が末尾についていれば落とし、数字部分のみの名前にする
    video_name = stem[:-6] if stem.endswith("_train") else stem

    # SBS をストリーミングで書き出す（元動画を上書きしないよう末尾に _3D を付与）
    width_sbs = width * 2
    output_filename = f"{video_name}_3D.mp4"
    output_path = os.path.join(save_dir, output_filename)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width_sbs, height),
    )

    written_frames = 0
    prev_end = None

    for start, end in frame_ranges:
        if end - start <= 0:
            continue
        frames_left, frames_warped, frames_mask = _load_chunk(start, end)

        input_frames_i = frames_warped.clone()
        mask_frames_i = frames_mask

        overlap_count = max(0, (prev_end - start) if prev_end is not None else 0)
        if generated_prev is not None and overlap_count > 0 and start > 0:
            ov = min(overlap_count, generated_prev.shape[0], input_frames_i.shape[0])
            if ov > 0:
                input_frames_i[:ov] = generated_prev[-ov:]

        video_latents = spatial_tiled_process(
            input_frames_i,
            mask_frames_i,
            pipeline,
            tile_num,
            spatial_n_compress=8,
            min_guidance_scale=1.01,
            max_guidance_scale=1.01,
            decode_chunk_size=8,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.0,
            num_inference_steps=num_inference_steps,
        )

        video_latents = video_latents.unsqueeze(0)
        if video_latents == torch.float16:
            pipeline.vae.to(dtype=torch.float16)

        video_frames = pipeline.decode_latents(
            video_latents,
            num_frames=video_latents.shape[1],
            decode_chunk_size=2,
        )
        video_frames = tensor2vid(video_frames, pipeline.image_processor, output_type="pil")[0]

        for j in range(len(video_frames)):
            img = video_frames[j]
            video_frames[j] = (
                torch.tensor(np.array(img)).permute(2, 0, 1).to(dtype=torch.float32)
                / 255.0
            )
        generated = torch.stack(video_frames)

        trim = 0 if start == 0 else min(overlap_count, generated.shape[0], frames_left.shape[0])
        append_gen = generated if trim == 0 else generated[trim:]
        append_left = frames_left if trim == 0 else frames_left[trim:]

        generated_prev = generated

        frames_sbs = torch.cat([append_left, append_gen], dim=3)
        frames_sbs_np = (
            (frames_sbs * 255)
            .permute(0, 2, 3, 1)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )
        for frame in frames_sbs_np:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
        written_frames += frames_sbs_np.shape[0]
        prev_end = end

    writer.release()
    output_count = ffprobe_count_frames(output_path)
    output_format = ffprobe_format_info(output_path)
    log_ffprobe("output", output_count, output_format)

    if written_frames != num_frames:
        raise ValueError(
            f"Written frame mismatch: wrote={written_frames} expected={num_frames}"
        )
    if int(output_count.get("nb_read_frames") or 0) != input_nb_frames:
        raise ValueError(
            "Output frame count mismatch: "
            f"output={output_count.get('nb_read_frames')} input={input_nb_frames}"
        )


if __name__ == "__main__":
    Fire(main)
