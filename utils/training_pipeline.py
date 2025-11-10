"""Pipeline construction and memory configuration helpers.

事前学習済みパイプラインの構築・UNet の複数 GPU への分散配置・
注意機構の最適化・VAE の省メモリ設定などを関数に分解しています。

依存:
- accelerate: `dispatch_model`, `infer_auto_device_map`
- diffusers/transformers: VAE, UNet, CLIP image encoder

注意:
- UNet のシャーディングは単一プロセスのモデル並列。分散学習とは併用しません。
- `no_split_module_classes` で Mamba 関連ブロックの過度な分割を避けています。
"""

import logging
from typing import Dict

import torch
from accelerate import dispatch_model
from accelerate.utils import infer_auto_device_map
from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.utils.torch_utils import is_compiled_module
from transformers import CLIPVisionModelWithProjection

from utils.logging_utils import ensure_logging_configured

from pipelines.mamba_stereo_video_inpainting_pipeline import MambaStableVideoDiffusionInpaintingPipeline


logger = logging.getLogger(__name__)


def load_inpainting_pipeline(
    pre_trained_path: str,
    unet_path: str,
    torch_dtype: torch.dtype,
    device: torch.device,
) -> MambaStableVideoDiffusionInpaintingPipeline:
    """Load the stereo inpainting pipeline with frozen VAE/encoder and trainable UNet.

    - image_encoder / vae は推論用に固定 (eval, no grad)
    - unet は学習対象 (train, requires_grad=True)
    """
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
    pipeline = pipeline.to(device)
    pipeline.unet.train()
    return pipeline


def maybe_shard_unet(
    pipeline: MambaStableVideoDiffusionInpaintingPipeline,
    shard_unet_across_gpus: bool,
    per_gpu_max_mem_gib: int,
) -> None:
    """Optionally shard UNet across available GPUs (single-process model parallel).

    自動デバイスマップが単一 GPU になる場合は簡易な手動マッピングにフォールバックします。
    """
    ensure_logging_configured()
    if not shard_unet_across_gpus or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return

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

    try:
        max_memory = {i: f"{per_gpu_max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
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
        used_devices = sorted(
            {
                idx
                for value in device_map.values()
                for idx in ([_device_index_from(value)] if not isinstance(value, (list, tuple)) else [_device_index_from(v) for v in value])
                if idx is not None
            }
        )
        if (len(device_map) == 1 and "" in device_map) or len(used_devices) < 2:
            ensure_logging_configured()
            logger.info("Auto sharding produced single-device map %s. Falling back to manual split...", device_map)
            manual_map = _build_manual_unet_map(pipeline.unet)
            pipeline.unet = dispatch_model(pipeline.unet, device_map=manual_map)
            logger.info("Sharded UNet with manual device_map: %s", manual_map)
        else:
            pipeline.unet = dispatch_model(pipeline.unet, device_map=device_map)
            ensure_logging_configured()
            logger.info("Sharded UNet across GPUs. Device map uses devices: %s", used_devices)
    except Exception as err:
        ensure_logging_configured()
        logger.warning("Auto sharding failed: %s. Trying a simple manual split...", err)
        try:
            manual_map = _build_manual_unet_map(pipeline.unet)
            pipeline.unet = dispatch_model(pipeline.unet, device_map=manual_map)
            logger.info("Sharded UNet with manual device_map: %s", manual_map)
        except Exception as final_err:
            logger.warning("UNet sharding failed, continuing without sharding: %s", final_err)


def _build_manual_unet_map(unet: torch.nn.Module) -> Dict[str, int]:
    """Construct a manual two-GPU mapping for UNet modules.

    ポリシー:
    - 高解像度側 (最終 up_blocks 付近) は GPU0 に寄せて転送回数を削減
    - それ以外は GPU1 に配置
    """
    manual_map: Dict[str, int] = {}
    prefer_gpu0 = {"conv_in", "down_blocks", "mid_block", "time_proj", "time_embedding", "add_time_proj", "add_embedding"}
    for name, _ in unet.named_children():
        if name == "up_blocks":
            continue
        manual_map[name] = 0 if name in prefer_gpu0 else 1

    up_mod = getattr(unet, "up_blocks", None)
    if up_mod is not None and hasattr(up_mod, "_modules"):
        up_keys = list(up_mod._modules.keys())
        n_up = len(up_keys)
        move_last = 2 if n_up >= 4 else 1
        for i, key in enumerate(up_keys):
            dev = 0 if (i >= n_up - move_last) else 1
            manual_map[f"up_blocks.{key}"] = dev
        last_dev = 0 if move_last >= 1 else 1
        manual_map["conv_norm_out"] = last_dev
        manual_map["conv_act"] = last_dev
        manual_map["conv_out"] = last_dev
    return manual_map


def configure_unet_memory_features(
    pipeline: MambaStableVideoDiffusionInpaintingPipeline,
    enable_gradient_checkpointing: bool,
    attn_mode: str,
    ff_chunk_size: int | None = None,
    ff_chunk_dim: int = 1,
) -> None:
    """Enable gradient checkpointing and attention optimizations on the UNet.

    順序:
    1) gradient checkpointing を可能なら有効化
    2) xFormers -> torch SDP の順で試行
    3) attention slicing を最後に有効化
    """
    ensure_logging_configured()
    if enable_gradient_checkpointing:
        try:
            pipeline.unet.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing on UNet")
        except Exception as err:
            logger.warning("Gradient checkpointing not available: %s", err)

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
            pipeline.unet.set_attn_processor("torch-sdp")
            logger.info("Using PyTorch scaled dot-product attention")
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
    """Turn on optional VAE helper flags if available (slicing/tiling)."""
    ensure_logging_configured()
    for fn_name in ("enable_slicing", "enable_tiling"):
        try:
            getattr(pipeline.vae, fn_name)()
            logger.info("VAE %s enabled", fn_name)
        except Exception:
            continue


def is_compiled_vae(pipeline: MambaStableVideoDiffusionInpaintingPipeline) -> bool:
    """Expose whether the VAE is wrapped in torch.compile (has _orig_mod)."""
    return hasattr(pipeline.vae, "_orig_mod") and is_compiled_module(pipeline.vae)
