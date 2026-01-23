# =============================================
# File: /workspace/stereocraft/utils/training_pipeline.py
# ---------------------------------------------
# 目的: 学習用パイプライン構築と省メモリ設定
# =============================================

"""Pipeline construction and memory tuning helpers for training."""

import gc
import inspect
import logging
import os
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from accelerate import dispatch_model
from accelerate.hooks import remove_hook_from_submodules
from accelerate.utils import infer_auto_device_map
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.utils.torch_utils import is_compiled_module
from transformers import CLIPVisionModelWithProjection

from utils.logging_utils import ensure_logging_configured

from pipelines.mamba_stereo_video_inpainting_pipeline import MambaStableVideoDiffusionInpaintingPipeline


logger = logging.getLogger(__name__)


def apply_mamba_runtime_flags(
    model: torch.nn.Module,
    *,
    use_fast_path: bool,
    autotune_warmup: bool,
) -> int:
    """Best-effort update of Mamba runtime flags on modules; returns updated module count."""
    fast_attr_names = (
        "use_fast_path",
        "mamba_use_fast_path",
        "fast_path",
        "use_mem_eff_path",
        "mamba_use_mem_eff_path",
    )
    autotune_attr_names = (
        "autotune_warmup",
        "mamba_autotune_warmup",
        "autotune",
        "mamba_autotune",
    )
    updated_modules = 0
    for module in model.modules():
        updated = False
        for attr in fast_attr_names:
            if hasattr(module, attr):
                try:
                    setattr(module, attr, bool(use_fast_path))
                    updated = True
                except Exception:
                    continue
        for attr in autotune_attr_names:
            if hasattr(module, attr):
                try:
                    setattr(module, attr, bool(autotune_warmup))
                    updated = True
                except Exception:
                    continue
        core = getattr(module, "core", None)
        if core is not None:
            for attr in fast_attr_names:
                if hasattr(core, attr):
                    try:
                        setattr(core, attr, bool(use_fast_path))
                        updated = True
                    except Exception:
                        continue
            for attr in autotune_attr_names:
                if hasattr(core, attr):
                    try:
                        setattr(core, attr, bool(autotune_warmup))
                        updated = True
                    except Exception:
                        continue
        if updated:
            updated_modules += 1
    return updated_modules


def load_inpainting_pipeline(
    pre_trained_path: str,
    unet_path: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    pipeline_device: torch.device | None = None,
) -> MambaStableVideoDiffusionInpaintingPipeline:
    """Load pipeline with frozen encoder/VAE and trainable UNet."""
    ensure_logging_configured()
    unet_subdir = "unet_diffusers"
    expected_unet_dir = os.path.join(unet_path, unet_subdir)
    logger.info(
        "Loading UNet from %s (subfolder=%s, exists=%s)",
        unet_path,
        unet_subdir,
        os.path.isdir(expected_unet_dir),
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path,
        subfolder="image_encoder",
        variant="fp16",
        torch_dtype=torch_dtype,
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path,
        subfolder="vae",
        variant="fp16",
        torch_dtype=torch_dtype,
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path,
        subfolder="unet_diffusers",
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )
    logger.info(
        "Loaded UNet OK: in_channels=%s, out_channels=%s, num_frames=%s, cross_attention_dim=%s",
        getattr(unet.config, "in_channels", None),
        getattr(unet.config, "out_channels", None),
        getattr(unet.config, "num_frames", None),
        getattr(unet.config, "cross_attention_dim", None),
    )

    image_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    unet.requires_grad_(True).train()

    pipeline = MambaStableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch_dtype,
    )
    target_device = device if pipeline_device is None else pipeline_device
    if target_device is not None:
        pipeline = pipeline.to(target_device)
    pipeline.unet.train()
    return pipeline


def maybe_shard_unet(
    pipeline: MambaStableVideoDiffusionInpaintingPipeline,
    shard_unet_across_gpus: bool,
    per_gpu_max_mem_gib: int | Sequence[int],
    manual_device_map: Dict[str, int] | None = None,
    shard_strategy: str = "balance_params",
    preflight_runner: Callable[[str], Dict[int, float]] | None = None,
    candidate_validator: Callable[[str, Dict[str, int]], bool] | None = None,
) -> bool | None:
    """Optionally shard UNet across available GPUs."""
    ensure_logging_configured()
    if not shard_unet_across_gpus:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("shard_unet_across_gpus=True but CUDA is unavailable.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("shard_unet_across_gpus=True but fewer than 2 CUDA devices are available.")

    def _device_index_from(value) -> int | None:
        """Normalize accelerate device map values to CUDA device indices."""
        if isinstance(value, int):
            return value
        if isinstance(value, torch.device):
            if value.type != "cuda":
                return None
            return value.index if value.index is not None else 0
        if isinstance(value, str):
            value = value.strip().lower()
            if not value or value.startswith("cpu") or value == "disk":
                return None
            if value.startswith("cuda"):
                parts = value.split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    return int(parts[1])
                return 0
        return None

    def _cleanup_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            for dev_idx in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(dev_idx)

    def _reset_unet_for_dispatch() -> None:
        try:
            remove_hook_from_submodules(pipeline.unet)
        except Exception:
            pass
        try:
            pipeline.unet.to("cpu")
        except Exception:
            pass
        _cleanup_cuda_memory()

    def _used_devices_from_map(map_obj: Dict[str, object]) -> List[int]:
        used = {
            idx
            for value in map_obj.values()
            for idx in (
                [_device_index_from(value)]
                if not isinstance(value, (list, tuple))
                else [_device_index_from(v) for v in value]
            )
            if idx is not None
        }
        return sorted(used)

    def get_param_bytes_by_device(model: torch.nn.Module) -> Dict[int, int]:
        bytes_by_dev: Dict[int, int] = defaultdict(int)
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.device.type != "cuda":
                continue
            dev_index = p.device.index if p.device.index is not None else 0
            bytes_by_dev[dev_index] += p.numel() * p.element_size()
        return dict(bytes_by_dev)

    def _build_max_memory() -> Dict[int, str]:
        device_count = torch.cuda.device_count()
        if isinstance(per_gpu_max_mem_gib, (list, tuple)):
            mem_values = list(per_gpu_max_mem_gib)
            if len(mem_values) < device_count:
                raise ValueError(
                    f"per_gpu_max_mem_gib needs {device_count} entries, got {len(mem_values)}"
                )
            return {i: f"{int(mem_values[i])}GiB" for i in range(device_count)}
        return {i: f"{int(per_gpu_max_mem_gib)}GiB" for i in range(device_count)}

    def _validate_device_map(map_obj: Dict[str, int]) -> None:
        module_names = {name for name, _ in pipeline.unet.named_modules()}
        missing = [name for name in map_obj if name not in module_names]
        if missing:
            raise ValueError(f"Invalid device_map keys: {missing}")

    def _prefix_param_bytes(prefix: str) -> int:
        total = 0
        for name, param in pipeline.unet.named_parameters():
            if name == prefix or name.startswith(prefix + "."):
                total += param.numel() * param.element_size()
        return total

    def _balance_device_map() -> Dict[str, int]:
        prefixes = [
            "conv_in",
            "time_proj",
            "time_embedding",
            "add_time_proj",
            "add_embedding",
            "down_blocks.0",
            "down_blocks.1",
            "down_blocks.2",
            "down_blocks.3",
            "mid_block",
            "up_blocks.0",
            "up_blocks.1",
            "up_blocks.2",
            "up_blocks.3",
            "conv_norm_out",
            "conv_act",
            "conv_out",
        ]
        module_names = {name for name, _ in pipeline.unet.named_modules()}
        items: List[Tuple[str, int]] = []
        for prefix in prefixes:
            if prefix not in module_names:
                continue
            size = _prefix_param_bytes(prefix)
            items.append((prefix, size))
        if not items:
            raise RuntimeError("Failed to build balanced device_map: no matching UNet prefixes found.")
        items.sort(key=lambda x: x[1], reverse=True)
        totals = {0: 0, 1: 0}
        mapping: Dict[str, int] = {}
        for name, size in items:
            dev = 0 if totals[0] <= totals[1] else 1
            mapping[name] = dev
            totals[dev] += size

        def _imbalance_ratio(t0: int, t1: int) -> float:
            max_val = max(t0, t1, 1)
            return abs(t0 - t1) / float(max_val)

        # Local refinement: move one block at a time to reduce imbalance.
        improved = True
        while improved:
            improved = False
            current_ratio = _imbalance_ratio(totals[0], totals[1])
            for name, size in items:
                src = mapping[name]
                dst = 1 - src
                new_totals = dict(totals)
                new_totals[src] -= size
                new_totals[dst] += size
                new_ratio = _imbalance_ratio(new_totals[0], new_totals[1])
                if new_ratio + 1e-6 < current_ratio:
                    mapping[name] = dst
                    totals = new_totals
                    improved = True
                    break
        return mapping

    def _activation_weighted_device_map() -> Dict[str, int]:
        prefixes = [
            "conv_in",
            "time_proj",
            "time_embedding",
            "add_time_proj",
            "add_embedding",
            "down_blocks.0",
            "down_blocks.1",
            "down_blocks.2",
            "down_blocks.3",
            "mid_block",
            "up_blocks.0",
            "up_blocks.1",
            "up_blocks.2",
            "up_blocks.3",
            "conv_norm_out",
            "conv_act",
            "conv_out",
        ]
        module_names = {name for name, _ in pipeline.unet.named_modules()}
        existing = [name for name in prefixes if name in module_names]
        if not existing:
            raise RuntimeError("Failed to build activation-weighted device_map: no matching UNet prefixes found.")

        fixed_gpu1 = {
            "down_blocks.1",
            "down_blocks.2",
            "up_blocks.2",
            "up_blocks.3",
            "conv_norm_out",
            "conv_act",
            "conv_out",
        }
        fixed_gpu0 = {
            "conv_in",
            "time_proj",
            "time_embedding",
            "add_time_proj",
            "add_embedding",
            "down_blocks.0",
            "down_blocks.3",
            "mid_block",
            "up_blocks.0",
            "up_blocks.1",
        }

        mapping: Dict[str, int] = {}
        totals = {0: 0, 1: 0}
        for name in existing:
            size = _prefix_param_bytes(name)
            if name in fixed_gpu1:
                mapping[name] = 1
                totals[1] += size
            elif name in fixed_gpu0:
                mapping[name] = 0
                totals[0] += size

        remaining = [name for name in existing if name not in mapping]
        remaining.sort(key=lambda x: _prefix_param_bytes(x), reverse=True)
        for name in remaining:
            size = _prefix_param_bytes(name)
            dev = 0 if totals[0] <= totals[1] else 1
            mapping[name] = dev
            totals[dev] += size
        return mapping

    def _contiguous_prefixes() -> List[str]:
        prefixes = [
            "conv_in",
            "time_proj",
            "time_embedding",
            "add_time_proj",
            "add_embedding",
            "down_blocks.0",
            "down_blocks.1",
            "down_blocks.2",
            "down_blocks.3",
            "mid_block",
            "up_blocks.0",
            "up_blocks.1",
            "up_blocks.2",
            "up_blocks.3",
            "conv_norm_out",
            "conv_act",
            "conv_out",
        ]
        module_names = {name for name, _ in pipeline.unet.named_modules()}
        return [name for name in prefixes if name in module_names]

    def _frontload_gpu1_device_map(prefixes: List[str]) -> Dict[str, int]:
        frontload_set = {
            "conv_in",
            "time_proj",
            "time_embedding",
            "add_time_proj",
            "add_embedding",
        }
        device_map: Dict[str, int] = {}
        for name in prefixes:
            if name in frontload_set or name.startswith("down_blocks."):
                device_map[name] = 1
            else:
                device_map[name] = 0
        return device_map

    def _contiguous_segments(prefixes: List[str]) -> List[List[str]]:
        time_group = {"time_proj", "time_embedding", "add_time_proj", "add_embedding"}
        segments: List[List[str]] = []
        buffer: List[str] = []
        for name in prefixes:
            if name in time_group:
                buffer.append(name)
                continue
            if buffer:
                segments.append(buffer)
                buffer = []
            segments.append([name])
        if buffer:
            segments.append(buffer)
        return segments

    def _contiguous_one_cut_candidates(segments: List[List[str]]) -> List[Tuple[str, Dict[str, int], int]]:
        candidates: List[Tuple[str, Dict[str, int], int]] = []
        if len(segments) < 2:
            return candidates
        for cut in range(1, len(segments)):
            for start_dev in (0, 1):
                device_map: Dict[str, int] = {}
                for idx, segment in enumerate(segments):
                    dev = start_dev if idx < cut else 1 - start_dev
                    for name in segment:
                        device_map[name] = dev
                tag = f"b1_{start_dev}to{1 - start_dev}_cut{cut}"
                candidates.append((tag, device_map, 1))
        return candidates

    def _contiguous_two_cut_candidates(
        segments: List[List[str]],
        prefix_sizes: Dict[str, int],
        max_per_pattern: int = 4,
    ) -> List[Tuple[str, Dict[str, int], int]]:
        candidates: List[Tuple[str, Dict[str, int], int]] = []
        if len(segments) < 3:
            return candidates
        sizes = [sum(prefix_sizes[name] for name in segment) for segment in segments]
        total = sum(sizes)
        prefix_sum = [0]
        for size in sizes:
            prefix_sum.append(prefix_sum[-1] + size)

        def _segment_size(start: int, end: int) -> int:
            return prefix_sum[end] - prefix_sum[start]

        options_010: List[Tuple[float, int, int]] = []
        options_101: List[Tuple[float, int, int]] = []
        for start in range(1, len(segments) - 1):
            for end in range(start + 1, len(segments)):
                mid_size = _segment_size(start, end)
                outer_size = total - mid_size
                imbalance = abs(outer_size - mid_size) / float(max(outer_size, mid_size, 1))
                options_010.append((imbalance, start, end))
                options_101.append((imbalance, start, end))
        options_010.sort(key=lambda x: x[0])
        options_101.sort(key=lambda x: x[0])
        for pattern, options in (("010", options_010), ("101", options_101)):
            for _, start, end in options[:max_per_pattern]:
                device_map: Dict[str, int] = {}
                for idx, segment in enumerate(segments):
                    if pattern == "010":
                        dev = 1 if start <= idx < end else 0
                    else:
                        dev = 0 if start <= idx < end else 1
                    for name in segment:
                        device_map[name] = dev
                tag = f"b2_{pattern}_{start}-{end}"
                candidates.append((tag, device_map, 2))
        return candidates

    def _is_oom_error(err: RuntimeError) -> bool:
        msg = str(err).lower()
        return "out of memory" in msg or "cuda error" in msg

    def _is_device_mismatch_error(err: RuntimeError) -> bool:
        msg = str(err).lower()
        return "expected all tensors to be on the same device" in msg or "found at least two devices" in msg

    def _evaluate_contiguous_candidates(
        candidates: List[Tuple[str, Dict[str, int], int]]
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        total = len(candidates)
        for idx, (tag, device_map, boundary_count) in enumerate(candidates, start=1):
            logger.info(
                "Contiguous search candidate %d/%d %s boundaries=%d map=%s",
                idx,
                total,
                tag,
                boundary_count,
                device_map,
            )
            try:
                _validate_device_map(device_map)
                _reset_unet_for_dispatch()
                pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
                pipeline.unet.train()
                used_devices = _used_devices_from_map(device_map)
                if len(used_devices) < 2:
                    raise RuntimeError(f"Contiguous candidate uses single device: {used_devices}")
                stats = preflight_runner(f"contiguous_{tag}") if preflight_runner else {}
                peaks = {dev: float(stats.get(dev, 0.0)) for dev in used_devices}
                peak_max = max(peaks.values()) if peaks else 0.0
                imbalance = abs(peaks.get(0, 0.0) - peaks.get(1, 0.0))
                param_bytes_by_device = get_param_bytes_by_device(pipeline.unet)
                param_gib_by_device = {
                    dev: bytes_val / float(1024**3) for dev, bytes_val in param_bytes_by_device.items()
                }
                extra_mib_by_device = {
                    dev: (bytes_val * 5.0) / float(1024**2)
                    for dev, bytes_val in param_bytes_by_device.items()
                }
                score_devices = sorted(
                    set(used_devices) | set(param_bytes_by_device.keys()) | set(peaks.keys())
                )
                score_by_device = {
                    dev: float(peaks.get(dev, 0.0)) + float(extra_mib_by_device.get(dev, 0.0))
                    for dev in score_devices
                }
                score = max(score_by_device.values()) if score_by_device else 0.0
                logger.info(
                    "Contiguous search candidate %s OK: param_gib_by_device=%s extra_mib_by_device=%s peak_mib=%s score=%.1f",
                    tag,
                    param_gib_by_device,
                    extra_mib_by_device,
                    peaks,
                    score,
                )
                results.append(
                    {
                        "tag": tag,
                        "device_map": device_map,
                        "boundary_count": boundary_count,
                        "peaks": peaks,
                        "peak_max": peak_max,
                        "imbalance": imbalance,
                        "param_gib_by_device": param_gib_by_device,
                        "extra_mib_by_device": extra_mib_by_device,
                        "score_by_device": score_by_device,
                        "score": score,
                    }
                )
            except RuntimeError as err:
                if _is_oom_error(err):
                    logger.info("Contiguous search candidate %s OOM; skipping.", tag)
                    continue
                if _is_device_mismatch_error(err):
                    logger.info("Contiguous search candidate %s device mismatch; skipping.", tag)
                    continue
                logger.warning("Contiguous search candidate %s failed: %s", tag, err)
                raise
            finally:
                _reset_unet_for_dispatch()
        return results

    try:
        if manual_device_map:
            _validate_device_map(manual_device_map)
            _reset_unet_for_dispatch()
            pipeline.unet = dispatch_model(pipeline.unet, device_map=manual_device_map)
            used_devices = _used_devices_from_map(manual_device_map)
            logger.info("Sharded UNet with manual device_map: %s", manual_device_map)
            if len(used_devices) < 2:
                raise RuntimeError(f"Manual device_map uses single device: {used_devices}")
            return

        strategy = (shard_strategy or "balance_params").strip().lower()
        if strategy in ("contiguous_search", "contiguous"):
            if preflight_runner is None:
                logger.warning(
                    "contiguous_search requested but no preflight runner; falling back to activation_weighted."
                )
                device_map = _activation_weighted_device_map()
                _validate_device_map(device_map)
                _reset_unet_for_dispatch()
                pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
                used_devices = _used_devices_from_map(device_map)
                logger.info("Sharded UNet with activation-weighted device_map: %s", device_map)
                if len(used_devices) < 2:
                    raise RuntimeError(f"Activation-weighted device_map uses single device: {used_devices}")
                return False

            prefixes = _contiguous_prefixes()
            segments = _contiguous_segments(prefixes)
            if len(segments) < 2:
                logger.warning(
                    "contiguous_search found insufficient UNet prefixes; falling back to activation_weighted."
                )
                device_map = _activation_weighted_device_map()
                _validate_device_map(device_map)
                _reset_unet_for_dispatch()
                pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
                used_devices = _used_devices_from_map(device_map)
                logger.info("Sharded UNet with activation-weighted device_map: %s", device_map)
                if len(used_devices) < 2:
                    raise RuntimeError(f"Activation-weighted device_map uses single device: {used_devices}")
                return False

            prefix_sizes = {name: _prefix_param_bytes(name) for name in prefixes}
            candidates = _contiguous_one_cut_candidates(segments)
            frontload_map = _frontload_gpu1_device_map(prefixes)
            candidates.insert(0, ("frontload_gpu1", frontload_map, 2))
            logger.info("Contiguous search: evaluating %d one-cut candidates.", len(candidates))
            results = _evaluate_contiguous_candidates(candidates)
            if not results:
                candidates = _contiguous_two_cut_candidates(segments, prefix_sizes, max_per_pattern=4)
                logger.info("Contiguous search: evaluating %d two-cut candidates.", len(candidates))
                results = _evaluate_contiguous_candidates(candidates)

            if not results:
                logger.warning("contiguous_search found no viable device_map; falling back to activation_weighted.")
                device_map = _activation_weighted_device_map()
                _validate_device_map(device_map)
                _reset_unet_for_dispatch()
                pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
                used_devices = _used_devices_from_map(device_map)
                logger.info("Sharded UNet with activation-weighted device_map: %s", device_map)
                if len(used_devices) < 2:
                    raise RuntimeError(f"Activation-weighted device_map uses single device: {used_devices}")
                return False

            results_sorted = sorted(
                results, key=lambda item: (item["score"], item["boundary_count"], item["imbalance"])
            )
            selected = None
            if candidate_validator is not None:
                for candidate in results_sorted:
                    tag = candidate["tag"]
                    device_map = candidate["device_map"]
                    _validate_device_map(device_map)
                    _reset_unet_for_dispatch()
                    pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
                    pipeline.unet.train()
                    try:
                        ok = bool(candidate_validator(tag, device_map))
                    except RuntimeError as err:
                        if _is_oom_error(err):
                            logger.info(
                                "Contiguous search candidate %s optimizer dry-run OOM; trying next.",
                                tag,
                            )
                            ok = False
                        else:
                            raise
                    if ok:
                        selected = candidate
                        logger.info("Contiguous search candidate %s accepted by optimizer dry-run.", tag)
                        break
                    _reset_unet_for_dispatch()
                if selected is None:
                    raise RuntimeError(
                        "contiguous_search found no viable device_map after optimizer dry-run."
                    )
            else:
                selected = results_sorted[0]
                chosen_map = selected["device_map"]
                _validate_device_map(chosen_map)
                _reset_unet_for_dispatch()
                pipeline.unet = dispatch_model(pipeline.unet, device_map=chosen_map)
                pipeline.unet.train()

            chosen_map = selected["device_map"]
            peaks = selected["peaks"]
            boundary_count = selected["boundary_count"]
            score = selected["score"]
            param_gib_by_device = selected.get("param_gib_by_device", {})
            extra_mib_by_device = selected.get("extra_mib_by_device", {})
            logger.info(
                "Contiguous search selected strategy=%s device_map=%s peaks_mib=%s score=%.1f extra_mib_by_device=%s param_gib_by_device=%s boundaries=%d",
                "contiguous_search",
                chosen_map,
                peaks,
                score,
                extra_mib_by_device,
                param_gib_by_device,
                boundary_count,
            )
            return True

        if strategy in ("frontload_gpu1", "gpu1_frontload", "downblocks_gpu1"):
            prefixes = _contiguous_prefixes()
            if not prefixes:
                raise RuntimeError("frontload_gpu1 device_map failed: no matching UNet prefixes found.")
            device_map = _frontload_gpu1_device_map(prefixes)
            _validate_device_map(device_map)
            _reset_unet_for_dispatch()
            pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
            used_devices = _used_devices_from_map(device_map)
            logger.info("Sharded UNet with frontload GPU1 device_map: %s", device_map)
            if len(used_devices) < 2:
                raise RuntimeError(f"frontload_gpu1 device_map uses single device: {used_devices}")
            return

        if strategy in ("activation_weighted", "activation_balance", "activation"):
            device_map = _activation_weighted_device_map()
            _validate_device_map(device_map)
            _reset_unet_for_dispatch()
            pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
            used_devices = _used_devices_from_map(device_map)
            logger.info("Sharded UNet with activation-weighted device_map: %s", device_map)
            if len(used_devices) < 2:
                raise RuntimeError(f"Activation-weighted device_map uses single device: {used_devices}")
            return

        if strategy in ("balance_params", "balanced", "balance"):
            device_map = _balance_device_map()
            _validate_device_map(device_map)
            _reset_unet_for_dispatch()
            pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
            used_devices = _used_devices_from_map(device_map)
            logger.info("Sharded UNet with balanced device_map: %s", device_map)
            if len(used_devices) < 2:
                raise RuntimeError(f"Balanced device_map uses single device: {used_devices}")
            return

        if strategy in ("two_stage_split", "manual_split", "fallback"):
            manual_map = _build_manual_unet_map(pipeline.unet)
            _validate_device_map(manual_map)
            _reset_unet_for_dispatch()
            pipeline.unet = dispatch_model(pipeline.unet, device_map=manual_map)
            used_devices = _used_devices_from_map(manual_map)
            logger.info("Sharded UNet with manual device_map: %s", manual_map)
            if len(used_devices) < 2:
                raise RuntimeError(f"Manual device_map uses single device: {used_devices}")
            return

        if strategy in ("auto", "accelerate"):
            max_memory = _build_max_memory()
            no_split = [
                "MambaSpatioTemporalAdapter",
                "MambaSpatioTemporalModel",
                "TemporalMamba",
                "TransformerSpatioTemporalModel",
            ]
            device_map = infer_auto_device_map(
                pipeline.unet,
                max_memory=max_memory,
                no_split_module_classes=no_split,
            )
            used_devices = _used_devices_from_map(device_map)
            if (len(device_map) == 1 and "" in device_map) or len(used_devices) < 2:
                logger.info("Auto sharding produced single-device map %s. Falling back to manual split...", device_map)
                manual_map = _build_manual_unet_map(pipeline.unet)
                _validate_device_map(manual_map)
                _reset_unet_for_dispatch()
                pipeline.unet = dispatch_model(pipeline.unet, device_map=manual_map)
                used_devices = _used_devices_from_map(manual_map)
                logger.info("Sharded UNet with manual device_map: %s", manual_map)
                if len(used_devices) < 2:
                    raise RuntimeError(f"Manual device_map uses single device: {used_devices}")
                return

            _reset_unet_for_dispatch()
            pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
            logger.info("Sharded UNet across GPUs. Device map uses devices: %s", used_devices)
            logger.info("UNet device_map: %s", device_map)
            return

        raise ValueError(f"Unknown unet_shard_strategy: {shard_strategy}")
    except Exception:
        logger.exception("UNet sharding failed; aborting.")
        raise


def _build_manual_unet_map(unet: torch.nn.Module) -> Dict[str, int]:
    """Construct a manual two-GPU mapping for UNet modules.

    ポリシー:
    - down_blocks を前半 GPU0 / 後半 GPU1 に分割
    - mid_block と up_blocks は GPU1 に寄せる
    """
    def _top_level_keys() -> set[str]:
        if hasattr(unet, "_modules"):
            return set(unet._modules.keys())
        return {name for name, _ in unet.named_children()}

    def _infer_block_keys(prefix: str) -> List[str]:
        block_mod = getattr(unet, prefix, None)
        if block_mod is not None and hasattr(block_mod, "_modules"):
            keys = list(block_mod._modules.keys())
            if keys:
                return keys
        keys: set[str] = set()
        for name, _ in unet.named_modules():
            if name.startswith(prefix + "."):
                rest = name[len(prefix) + 1 :]
                key = rest.split(".")[0]
                if key:
                    keys.add(key)
        return sorted(keys, key=lambda x: int(x) if x.isdigit() else x)

    def _add_if_present(map_obj: Dict[str, int], name: str, dev: int, available: set[str]) -> None:
        if name in available:
            map_obj[name] = dev

    available = _top_level_keys()
    manual_map: Dict[str, int] = {}
    for name in ("conv_in", "time_proj", "time_embedding", "add_time_proj", "add_embedding"):
        _add_if_present(manual_map, name, 0, available)

    down_keys = _infer_block_keys("down_blocks")
    for i, key in enumerate(down_keys):
        dev = 0 if i < 2 else 1
        manual_map[f"down_blocks.{key}"] = dev

    _add_if_present(manual_map, "mid_block", 1, available)

    up_keys = _infer_block_keys("up_blocks")
    for key in up_keys:
        manual_map[f"up_blocks.{key}"] = 1

    for name in ("conv_norm_out", "conv_act", "conv_out"):
        _add_if_present(manual_map, name, 1, available)

    return manual_map


def configure_unet_memory_features(
    pipeline: MambaStableVideoDiffusionInpaintingPipeline,
    enable_gradient_checkpointing: bool,
    checkpoint_use_reentrant: bool | None,
    attn_mode: str,
    ff_chunk_size: int | None = None,
    ff_chunk_dim: int = 1,
) -> None:
    """Enable gradient checkpointing and attention optimizations."""
    ensure_logging_configured()
    if enable_gradient_checkpointing:
        try:
            ckpt_reentrant = False if checkpoint_use_reentrant is None else bool(checkpoint_use_reentrant)
            enable_fn = pipeline.unet.enable_gradient_checkpointing
            sig = inspect.signature(enable_fn)
            supports_kwargs = (
                "gradient_checkpointing_kwargs" in sig.parameters
                or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
            )
            if supports_kwargs:
                enable_fn(gradient_checkpointing_kwargs={"use_reentrant": ckpt_reentrant})
                logger.info("Enabled gradient checkpointing on UNet (use_reentrant=%s)", ckpt_reentrant)
            else:
                enable_fn()
                logger.info("Enabled gradient checkpointing on UNet (no kwargs)")
        except Exception as err:
            logger.warning("Gradient checkpointing not available or kwargs failed: %s", err)
            try:
                pipeline.unet.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing on UNet (fallback default)")
            except Exception:
                pass

    attn_mode_lower = (attn_mode or "").lower()
    if attn_mode_lower in ("auto", "xformers"):
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Using xFormers memory efficient attention")
        except Exception as err:
            if attn_mode_lower == "xformers":
                logger.warning("xFormers requested but failed: %s", err)
            attn_mode_lower = "auto"
    if attn_mode_lower in ("auto", "sdp"):
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0

            pipeline.unet.set_attn_processor(AttnProcessor2_0())
            logger.info("Using PyTorch scaled dot-product attention (AttnProcessor2_0)")
        except Exception:
            pass
    try:
        pipeline.enable_attention_slicing()
        logger.info("Enabled attention slicing")
    except Exception:
        pass

    # Optional: feed-forward chunking to reduce peak activation memory
    if ff_chunk_size is not None and ff_chunk_size > 0:
        try:
            pipeline.unet.enable_forward_chunking(chunk_size=ff_chunk_size, dim=int(ff_chunk_dim))
            logger.info(
                "Enabled UNet forward chunking: chunk_size=%s, dim=%s",
                ff_chunk_size,
                ff_chunk_dim,
            )
        except Exception as err:
            logger.warning("Forward chunking not available: %s", err)


def enable_vae_memory_helpers(pipeline: MambaStableVideoDiffusionInpaintingPipeline) -> None:
    """Enable optional VAE slicing/tiling when available."""
    ensure_logging_configured()
    for fn_name in ("enable_slicing", "enable_tiling"):
        try:
            getattr(pipeline.vae, fn_name)()
            logger.info("VAE %s enabled", fn_name)
        except Exception:
            continue


def is_compiled_vae(pipeline: MambaStableVideoDiffusionInpaintingPipeline) -> bool:
    """Return whether VAE is wrapped in torch.compile."""
    return hasattr(pipeline.vae, "_orig_mod") and is_compiled_module(pipeline.vae)
