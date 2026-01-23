# =============================================
# File: /workspace/stereocraft/blocks/clip_embed.py
# ---------------------------------------------
# 目的: CLIP画像埋め込みの互換取得
# =============================================

from __future__ import annotations

import torch


def encode_clip_image_like_pipeline(
    clip_model,
    clip_processor,
    image_chw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror pipeline _encode_image preprocessing for a single [C,H,W] image."""
    img = image_chw.clamp(0, 1).unsqueeze(0).to(device=device, dtype=dtype)  # [1,C,H,W]
    img = img * 2.0 - 1.0
    try:
        from pipelines.stereo_video_inpainting import _resize_with_antialiasing

        img = _resize_with_antialiasing(img, (224, 224))
    except Exception:
        import torch.nn.functional as F

        img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=True)
    img = (img + 1.0) / 2.0
    pixel = clip_processor(
        images=img,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values.to(device=device, dtype=dtype)
    embeds = clip_model(pixel).image_embeds  # [1, Cctx]
    return embeds.unsqueeze(1)  # [1,1,Cctx]
