import gc
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video

from diffusers.training_utils import set_seed
from fire import Fire
from decord import VideoReader, cpu

from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth, dataset_res_dict

from Forward_Warp import forward_warp

# Pure-numpy chunk schedule / joint affine fit / stitching / normalisation
# (unit-tested without torch in scripts/test_depth_chunk_alignment.py).
from depth_chunking import (
    _blend_depth_overlap,  # noqa: F401  (kept importable under the old names)
    _concatenate_depth_parts,  # noqa: F401
    _depth_sample_values,  # noqa: F401
    _percentile_stats,  # noqa: F401
    _replace_tail,  # noqa: F401
    _take_tail,  # noqa: F401
    depth_chunk_ranges,
    normalize_depth_inplace,
    pair_overlaps,
    save_depth_npz_with_extras,
    solve_global_chunk_affine,
    stitch_depth_chunks,
)

_solve_global_chunk_affine = solve_global_chunk_affine
_normalize_depth_inplace = normalize_depth_inplace

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - optional dependency
    mask_utils = None


LOGGER = logging.getLogger("depth_splatting")
LOG_FILE_PATH: Optional[str] = None
VALID_CPU_OFFLOAD_MODES = {"model", "sequential"}


def _parse_cpu_offload_mode(value: Optional[str]) -> Optional[str]:
    """Normalize CLI input for cpu_offload handling."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "none"):
            return None
        if normalized in VALID_CPU_OFFLOAD_MODES:
            return normalized
    raise ValueError(
        "cpu_offload must be one of {'none', 'model', 'sequential'}"
    )


def _to_float32_unit_range(frames: np.ndarray) -> np.ndarray:
    """
    Convert frames to contiguous float32 in [0, 1] without keeping large temporary buffers.
    """
    frames = np.asarray(frames)
    if np.issubdtype(frames.dtype, np.integer):
        frames = frames.astype("float32") / 255.0
    elif frames.dtype != np.float32:
        frames = frames.astype("float32")
    return np.ascontiguousarray(frames)


def initialize_logging(output_video_path: str) -> Tuple[logging.Logger, str]:
    """
    Configure both console and file logging. The log filename is derived from the
    output video path to help correlate a run with its artifacts.
    """
    global LOG_FILE_PATH

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = os.path.splitext(os.path.basename(output_video_path))[0]
    log_file_path = os.path.join("logs", f"{base_name}_{timestamp}.log")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplicate logs when running multiple times.
    if LOGGER.handlers:
        for handler in LOGGER.handlers[:]:
            LOGGER.removeHandler(handler)

    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False
    LOG_FILE_PATH = log_file_path

    LOGGER.info("Logging initialized. Writing detailed output to %s", log_file_path)
    return LOGGER, log_file_path


def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open"):
    if dataset == "open":
        LOGGER.info("==> processing video: %s", video_path)
        vid = VideoReader(video_path, ctx=cpu(0))
        LOGGER.info("==> original video shape: %s", (len(vid), *vid.get_batch([0]).shape[1:]))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]
        original_height, original_width = height, width

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    LOGGER.info(
        "==> downsampled shape: %s, with stride: %s",
        (len(frames_idx), *vid.get_batch([0]).shape[1:]),
        stride,
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    LOGGER.info(
        "==> final processing shape (input sequence T): %s",
        (len(frames_idx), *vid.get_batch([0]).shape[1:]),
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    LOGGER.info("Captured input sequence length T=%d frames for %s", len(frames), video_path)

    return frames, fps, original_height, original_width


def iterate_video_frame_chunks(
    video_path: str,
    process_length: int,
    target_fps: int,
    max_res: int,
    dataset: str,
    chunk_size: int,
    chunk_overlap: int,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer when using chunked processing.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if dataset == "open":
        base_reader = VideoReader(video_path, ctx=cpu(0))
        LOGGER.info("==> processing video: %s", video_path)
        LOGGER.info("==> original video shape: %s", (len(base_reader), *base_reader.get_batch([0]).shape[1:]))
        original_height, original_width = base_reader.get_batch([0]).shape[1:3]
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        base_reader = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = base_reader.get_batch([0]).shape[1:3]
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]

    resized_reader = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    native_fps = base_reader.get_avg_fps()
    fps = native_fps if target_fps == -1 else target_fps
    stride = round(native_fps / fps) if fps > 0 else 1
    stride = max(stride, 1)
    frames_idx = list(range(0, len(base_reader), stride))
    LOGGER.info(
        "==> downsampled shape: %s, with stride: %s",
        (len(frames_idx), *resized_reader.get_batch([0]).shape[1:]),
        stride,
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    LOGGER.info(
        "==> final processing shape (input sequence T): %s (streamed in chunks of %d)",
        (len(frames_idx), *resized_reader.get_batch([0]).shape[1:]),
        chunk_size,
    )

    if not frames_idx:
        raise ValueError("No frames available for processing. Check the video file or sampling parameters.")
    LOGGER.info("Total input sequence length T=%d frames for chunked processing", len(frames_idx))

    # D3: the last chunk is pulled back to full length instead of emitting a
    # trailing chunk that adds no new frames (see depth_chunking).
    chunk_ranges = depth_chunk_ranges(len(frames_idx), chunk_size, chunk_overlap)

    def chunk_generator():
        for start, end in chunk_ranges:
            chunk_indices = frames_idx[start:end]
            processed_frames = resized_reader.get_batch(chunk_indices).asnumpy()
            LOGGER.info(
                "Prepared chunk frames %d-%d (input sequence T=%d)",
                chunk_indices[0],
                chunk_indices[-1],
                len(chunk_indices),
            )
            yield chunk_indices, processed_frames

    metadata = {
        "original_height": original_height,
        "original_width": original_width,
        "original_fps": native_fps,
        "target_fps": fps,
        "total_frames": len(frames_idx),
        "height": height,
        "width": width,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_ranges": chunk_ranges,
    }

    return chunk_generator(), metadata


@dataclass(frozen=True)
class _FrameMaskEntry:
    color: np.ndarray
    rle: Dict[str, Any]


def _hex_to_rgb_float(color_value: Optional[str]) -> np.ndarray:
    """Convert a hex color like '#33AAFF' to normalized RGB float32."""
    fallback = np.array([1.0, 0.2, 0.2], dtype=np.float32)
    if not color_value:
        return fallback
    color_value = color_value.strip()
    if color_value.startswith("#"):
        color_value = color_value[1:]
    if len(color_value) != 6:
        return fallback
    try:
        r = int(color_value[0:2], 16)
        g = int(color_value[2:4], 16)
        b = int(color_value[4:6], 16)
    except ValueError:
        return fallback
    return np.array([r, g, b], dtype=np.float32) / 255.0


def _decode_coco_rle_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Decode a COCO RLE dict to a 2D uint8 mask in [0, 1]."""
    if mask_utils is None:
        raise ImportError(
            "pycocotools is required to decode ELR masks. "
            "Install it with `pip install pycocotools`."
        )
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(np.uint8)


def _blend_mask_with_overlay(
    base_mask: np.ndarray,
    overlay_rgb: np.ndarray,
    overlay_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend an RGB mask with an overlay using simple alpha compositing."""
    if base_mask.shape[:2] != overlay_rgb.shape[:2]:
        raise ValueError("Overlay size does not match base mask.")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay_alpha = overlay_mask.astype(np.float32) * alpha
    overlay_alpha = overlay_alpha[..., None]
    blended = base_mask * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha
    return np.clip(blended, 0.0, 1.0, out=blended)


class ElrMaskOverlay:
    """Load ELR/RLE annotations and provide per-frame overlay masks."""

    def __init__(
        self,
        *,
        frame_masks: Dict[int, List[_FrameMaskEntry]],
        source_height: int,
        source_width: int,
        target_height: int,
        target_width: int,
    ) -> None:
        self._frame_masks = frame_masks
        self.source_height = source_height
        self.source_width = source_width
        self.target_height = target_height
        self.target_width = target_width
        self._needs_resize = (source_height != target_height) or (source_width != target_width)

    @classmethod
    def from_file(
        cls,
        mask_path: str,
        *,
        target_height: int,
        target_width: int,
    ) -> "ElrMaskOverlay":
        if mask_utils is None:
            raise ImportError(
                "pycocotools is required to load ELR/RLE masks. "
                "Please install it via `pip install pycocotools`."
            )
        with open(mask_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        video_meta = payload.get("video", {})
        source_height = int(video_meta.get("height", target_height))
        source_width = int(video_meta.get("width", target_width))
        tracklets = payload.get("tracklets", [])
        frame_masks: Dict[int, List[_FrameMaskEntry]] = {}

        for tracklet in tracklets:
            color = _hex_to_rgb_float(tracklet.get("color"))
            for mask_desc in tracklet.get("masks", []):
                frame_index = mask_desc.get("frameIndex")
                rle = mask_desc.get("rle")
                if frame_index is None or rle is None or rle.get("counts") is None:
                    continue
                frame_masks.setdefault(int(frame_index), []).append(
                    _FrameMaskEntry(color=color, rle=rle)
                )

        LOGGER.info(
            "Loaded ELR mask file %s with %d annotated frames (source resolution %dx%d).",
            mask_path,
            len(frame_masks),
            source_width,
            source_height,
        )
        return cls(
            frame_masks=frame_masks,
            source_height=source_height,
            source_width=source_width,
            target_height=target_height,
            target_width=target_width,
        )

    def get_overlay(self, frame_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        entries = self._frame_masks.get(frame_index)
        if not entries:
            return None
        overlay_rgb = np.zeros(
            (self.target_height, self.target_width, 3), dtype=np.float32
        )
        overlay_mask = np.zeros((self.target_height, self.target_width), dtype=bool)
        for entry in entries:
            mask = _decode_coco_rle_mask(entry.rle)
            if self._needs_resize:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask = mask.astype(bool)
            overlay_rgb[mask] = entry.color
            overlay_mask |= mask
        return overlay_rgb, overlay_mask

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
        cpu_offload: Optional[str] = "model",
        norm_percentile: float = 0.01,
        affine_min_spread: float = 0.02,
        affine_ridge: float = 1e-5,
    ):
        # D7: final [0,1] normalisation clips at p{norm_percentile} /
        # p{100-norm_percentile}; 0 = absolute min/max (pre-2026-09-17).
        self.norm_percentile = float(norm_percentile)
        # D1: joint chunk-affine fit -- overlaps flatter than this (p90-p10)
        # are ignored, and every chunk is ridged weakly toward (1, 0).
        self.affine_min_spread = float(affine_min_spread)
        self.affine_ridge = float(affine_ridge)
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        LOGGER.info("DepthCrafter cpu_offload mode: %s", cpu_offload or "none")
        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            LOGGER.warning("Xformers is not enabled: %s", e)
        self.pipe.enable_attention_slicing()

        # Pre-normalization (depth_min, depth_max) from the most recent infer()/
        # infer_chunked() call, for callers that need to persist the disparity
        # scale that global min-max normalization otherwise discards.
        self.last_depth_min: Optional[float] = None
        self.last_depth_max: Optional[float] = None

    def _run_depth_estimation(
        self,
        frames: np.ndarray,
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int,
        overlap: int,
        track_time: bool,
        vae_chunk_size: int = 8,
    ) -> np.ndarray:
        if vae_chunk_size <= 0:
            raise ValueError("vae_chunk_size must be positive")
        frames = _to_float32_unit_range(frames)
        with torch.inference_mode():
            result = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
                decode_chunk_size=vae_chunk_size,
            ).frames[0]

        result = result.sum(-1) / result.shape[-1]
        return result.astype("float32")

    @staticmethod
    def _resize_to_original(
        depth: np.ndarray,
        original_height: int,
        original_width: int,
    ) -> np.ndarray:
        tensor_res = torch.from_numpy(depth).unsqueeze(1).float().contiguous().cuda()
        resized = F.interpolate(
            tensor_res,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.cpu().numpy()[:, 0, :, :]

    def _normalize_depth(self, depth: np.ndarray) -> Tuple[np.ndarray, float, float]:
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        return normalize_depth_inplace(depth, self.norm_percentile)

    def infer(
        self,
        input_video_path: str,
        output_video_path: str,
        process_length: int = -1,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = -1,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = False,
        vae_chunk_size: int = 8,
    ):
        set_seed(seed)

        frames, target_fps, original_height, original_width = read_video_frames(
            input_video_path,
            process_length,
            target_fps,
            max_res,
            dataset,
        )

        res = self._run_depth_estimation(
            frames,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            overlap=overlap,
            track_time=track_time,
            vae_chunk_size=vae_chunk_size,
        )
        res = self._resize_to_original(res, original_height, original_width)
        res, self.last_depth_min, self.last_depth_max = self._normalize_depth(res)
        vis = vis_sequence_depth(res)
        LOGGER.info(
            "Depth inference produced normalized sequence T=%d frames for %s",
            res.shape[0],
            input_video_path,
        )
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            os.path.dirname(output_video_path), os.path.splitext(os.path.basename(output_video_path))[0]
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_depth:
            save_depth_npz_with_extras(
                save_path + ".npz",
                res,
                self.last_depth_min,
                self.last_depth_max,
                extras={"norm_percentile": self.norm_percentile},
            )
            write_video(save_path + "_depth_vis.mp4", vis*255.0, fps=target_fps, video_codec="h264", options={"crf": "16"})

        return res, vis

    def infer_from_frames(
        self,
        frames: np.ndarray,
        original_height: int,
        original_width: int,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        track_time: bool = False,
        vae_chunk_size: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        depth = self._run_depth_estimation(
            frames,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            overlap=overlap,
            track_time=track_time,
            vae_chunk_size=vae_chunk_size,
        )
        depth = self._resize_to_original(depth, original_height, original_width)
        depth, self.last_depth_min, self.last_depth_max = self._normalize_depth(depth)
        depth_vis = vis_sequence_depth(depth)
        LOGGER.info("Depth inference (chunk) produced sequence T=%d frames", depth.shape[0])
        return depth, depth_vis

    def infer_chunked(
        self,
        input_video_path: str,
        process_length: int = -1,
        num_denoising_steps: int = 8,
        guidance_scale: float = 1.2,
        window_size: int = 70,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = -1,
        seed: int = 42,
        track_time: bool = False,
        chunk_size: int = 140,
        chunk_overlap: int = 25,
        vae_chunk_size: int = 8,
    ) -> np.ndarray:
        set_seed(seed)
        chunk_iter, metadata = iterate_video_frame_chunks(
            input_video_path,
            process_length,
            target_fps,
            max_res,
            dataset,
            chunk_size,
            chunk_overlap,
        )
        expected_total = int(metadata["total_frames"])
        chunk_ranges = list(metadata["chunk_ranges"])
        # Actual frames shared by consecutive chunks: the nominal overlap,
        # except for the pulled-back last chunk (D3) where it is wider.
        overlaps = pair_overlaps(chunk_ranges)

        raw_chunks: List[np.ndarray] = []
        for chunk_id, (chunk_indices, frames) in enumerate(chunk_iter):
            LOGGER.info(
                "Running DepthCrafter chunk %d with source frames %d-%d (T=%d)",
                chunk_id,
                int(chunk_indices[0]),
                int(chunk_indices[-1]),
                len(chunk_indices),
            )
            chunk_depth = self._run_depth_estimation(
                frames,
                num_denoising_steps=num_denoising_steps,
                guidance_scale=guidance_scale,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
                vae_chunk_size=vae_chunk_size,
            ).astype(np.float32, copy=False)
            raw_chunks.append(np.ascontiguousarray(chunk_depth))

            del frames, chunk_depth
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fit_report: List[Dict[str, float]] = []
        affine_params = solve_global_chunk_affine(
            raw_chunks,
            overlaps,
            min_spread=self.affine_min_spread,
            ridge=self.affine_ridge,
            report=fit_report,
        )
        for entry in fit_report:
            LOGGER.info(
                "Chunk overlap %d-%d: %d frames, p90-p10 spread=%.4f -> %s",
                int(entry["pair"]),
                int(entry["pair"]) + 1,
                int(entry["overlap"]),
                entry["spread"],
                "used" if entry["used"] else "FLAT, chunk inherits previous transform",
            )
        for chunk_id, (scale, offset) in enumerate(affine_params):
            LOGGER.info(
                "Chunk %d global affine fit: scale=%.6f offset=%.6f",
                chunk_id,
                scale,
                offset,
            )
            raw_chunks[chunk_id] = raw_chunks[chunk_id] * scale + offset

        def _log_stitch(chunk_id: int, new_frames: int, unique_frames: int) -> None:
            LOGGER.info(
                "Depth chunk %d accepted %d new frames; stitched T=%d/%d",
                chunk_id,
                new_frames,
                unique_frames,
                expected_total,
            )

        depth = stitch_depth_chunks(raw_chunks, overlaps, expected_total, log=_log_stitch)
        depth, self.last_depth_min, self.last_depth_max = normalize_depth_inplace(
            depth, self.norm_percentile
        )
        LOGGER.info(
            "Chunked depth inference produced globally normalized sequence T=%d at %dx%d",
            depth.shape[0],
            depth.shape[2],
            depth.shape[1],
        )
        return depth
    

class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        """
        :param im: BCHW
        :param disp: B1HW
        :return: BCHW
        detach will lead to unconverge!!
        """
        im = im.contiguous()
        disp = disp.contiguous()
        # weights_map = torch.abs(disp)
        weights_map = disp - disp.min()
        weights_map = (
            1.414
        ) ** weights_map  # using 1.414 instead of EXP for avoding numerical overflow.
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        # mask = self.fw(weights_map, flow.detach())
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map
        

def DepthSplatting(
        input_video_path, 
        output_video_path, 
        video_depth, 
        depth_vis, 
        max_disp, 
        process_length, 
        batch_size):
    '''
    Depth-Based Video Splatting Using the Video Depth.
    Args:
        input_video_path: Path to the input video.
        output_video_path: Path to the output video.
        video_depth: Video depth with shape of [T, H, W] in [0, 1].
        depth_vis: Visualized video depth with shape of [T, H, W, 3] in [0, 1].
        process_length: The length of video to process.
        batch_size: The batch size for splatting to save GPU memory. 
        elr_mask_path: Optional path to an ELR/RLE mask json to overlay on the occlusion map.
        elr_mask_alpha: Opacity for the ELR mask overlay [0-1].
    '''
    vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    original_fps = vid_reader.get_avg_fps()
    input_frames = vid_reader[:].asnumpy()

    if process_length != -1 and process_length < len(input_frames):
        input_frames = input_frames[:process_length]
        video_depth = video_depth[:process_length]
        depth_vis = depth_vis[:process_length]

    num_frames = len(input_frames)
    height, width, _ = input_frames[0].shape

    # Initialize OpenCV VideoWriter
    out = cv2.VideoWriter(
        output_video_path, 
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps, 
        (width * 2, height)
    )
    LOGGER.info(
        "Preparing to write output sequence T=%d frames to %s",
        num_frames,
        output_video_path,
    )

    depth_vis_uint8_full = (
        np.clip(depth_vis, 0.0, 1.0, out=depth_vis) * 255.0
    ).astype(np.uint8)

    for i in range(0, num_frames, batch_size):
        batch_frames = _to_float32_unit_range(input_frames[i:i+batch_size])
        batch_depth_vis = depth_vis_uint8_full[i:i+batch_size]

        batch_frames_uint8 = (np.clip(batch_frames, 0.0, 1.0) * 255.0).astype(np.uint8)

        for j in range(len(batch_frames_uint8)):
            video_grid = np.concatenate([batch_frames_uint8[j], batch_depth_vis[j]], axis=1)
            video_grid_bgr = cv2.cvtColor(video_grid, cv2.COLOR_RGB2BGR)
            out.write(video_grid_bgr)

    out.release()
    LOGGER.info("Finished writing %d frames to %s", num_frames, output_video_path)


class DepthSplattingStreamer:
    def __init__(
        self,
        output_video_path: str,
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width * 2, frame_height),
        )

    def process_chunk(
        self,
        frames_chunk: np.ndarray,
        depth_vis_chunk: np.ndarray,
        frame_indices: Optional[Sequence[int]] = None,
        chunk_id: Optional[int] = None,
    ) -> None:
        num_frames = len(frames_chunk)
        if num_frames == 0:
            return
        chunk_label = f"chunk {chunk_id}" if chunk_id is not None else "chunk"
        LOGGER.info(
            "Processing %s with input sequence T=%d frames (output target: %s)",
            chunk_label,
            num_frames,
            self.writer,
        )

        left_frames_uint8 = np.multiply(
            np.clip(_to_float32_unit_range(frames_chunk), 0.0, 1.0),
            255.0,
        ).astype(np.uint8)
        depth_vis_uint8 = np.clip(depth_vis_chunk, 0, 255).astype(np.uint8)

        for j in range(num_frames):
            video_grid = np.concatenate(
                [left_frames_uint8[j], depth_vis_uint8[j]],
                axis=1,
            )
            video_grid_bgr = cv2.cvtColor(video_grid, cv2.COLOR_RGB2BGR)
            self.writer.write(video_grid_bgr)

        LOGGER.info(
            "Finished writing %s with sequence T=%d frames to output.",
            chunk_label,
            num_frames,
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main(
    input_video_path: str,
    output_video_path: Optional[str] = None,
    unet_path: str = "./weights/DepthCrafter",
    pre_trained_path: str = "./weights/stable-video-diffusion-img2vid-xt-1-1",
    max_disp: float = 20.0,
    process_length: int = -1,
    batch_size: int = 10,
    num_denoising_steps: int = 8,
    guidance_scale: float = 1.2,
    window_size: int = 70,
    overlap: int = 25,
    max_res: int = 1024,
    dataset: str = "open",
    target_fps: int = -1,
    seed: int = 42,
    track_time: bool = False,
    save_depth: bool = True,
    chunk_size: int = -1,
    chunk_overlap: int = -1,
    vae_chunk_size: int = 8,
    cpu_offload: Optional[str] = "model",
    debug_video: bool = False,
    norm_percentile: float = 0.01,
    affine_min_spread: float = 0.02,
    affine_ridge: float = 1e-5,
):
    cpu_offload_mode = _parse_cpu_offload_mode(cpu_offload)
    base_dir = os.path.dirname(input_video_path)
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    if output_video_path is None:
        output_video_path = os.path.join(base_dir, f"{base_name}_1x2_video.mp4")
    depth_npz_path = os.path.join(base_dir, f"{base_name}_depth.npz")
    logger, log_file_path = initialize_logging(output_video_path)
    logger.info(
        "Starting depth splatting run | input: %s | output(mp4 debug?): %s | depth npz: %s | logs: %s",
        input_video_path,
        output_video_path if debug_video else "(disabled; debug_video=False)",
        depth_npz_path,
        log_file_path,
    )
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_trained_path=pre_trained_path,
        cpu_offload=cpu_offload_mode,
        norm_percentile=norm_percentile,
        affine_min_spread=affine_min_spread,
        affine_ridge=affine_ridge,
    )

    video_depth: Optional[np.ndarray] = None
    depth_vis: Optional[np.ndarray] = None

    if chunk_size > 0:
        chunk_overlap = overlap if chunk_overlap < 0 else chunk_overlap
        LOGGER.info(
            "Using chunked depth inference: chunk_size=%d chunk_overlap=%d",
            chunk_size,
            chunk_overlap,
        )
        if debug_video:
            LOGGER.warning(
                "debug_video=True is not supported in chunked depth mode; depth npz will still be saved."
            )
        video_depth = depthcrafter_demo.infer_chunked(
            input_video_path=input_video_path,
            process_length=process_length,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            overlap=overlap,
            max_res=max_res,
            dataset=dataset,
            target_fps=target_fps,
            seed=seed,
            track_time=track_time,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            vae_chunk_size=vae_chunk_size,
        )
    else:
        video_depth, depth_vis = depthcrafter_demo.infer(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            process_length=process_length,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            window_size=window_size,
            overlap=overlap,
            max_res=max_res,
            dataset=dataset,
            target_fps=target_fps,
            seed=seed,
            track_time=track_time,
            save_depth=False,  # we handle saving below
            vae_chunk_size=vae_chunk_size,
        )

    # D7: one on-disk format for both paths (uint16 fixed point via
    # depth_io) instead of float16 (chunked) / float32 (legacy). disp_min /
    # disp_max are the lo/hi of the normalisation map, i.e. the
    # p{norm_percentile}/p{100-norm_percentile} clip points when it is > 0.
    depth_npz_path = os.path.join(base_dir, f"{base_name}_depth.npz")
    save_depth_npz_with_extras(
        depth_npz_path,
        video_depth,
        depthcrafter_demo.last_depth_min,
        depthcrafter_demo.last_depth_max,
        extras={"norm_percentile": norm_percentile},
    )
    LOGGER.info(
        "Saved depth npz to %s (uint16_fixed; norm_percentile=%g; disp_min=%.6f disp_max=%.6f)",
        depth_npz_path,
        norm_percentile,
        depthcrafter_demo.last_depth_min,
        depthcrafter_demo.last_depth_max,
    )

    if debug_video and depth_vis is not None:
        DepthSplatting(
            input_video_path, 
            output_video_path, 
            video_depth, 
            depth_vis,
            max_disp,
            process_length, 
            batch_size,
        )


if __name__ == "__main__":
    Fire(main)
