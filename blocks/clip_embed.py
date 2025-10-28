from __future__ import annotations
import torch

def encode_clip_image_like_pipeline(
    clip_model,
    clip_processor,
    image_chw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Match the pipeline's _encode_image preprocessing exactly:
      - input image_chw: [C,H,W] in [0,1]
      - map to [-1,1], antialias-resize to 224x224, back to [0,1]
      - CLIPImageProcessor normalization (no rescale/resize)
      - project via CLIP vision -> [1,1,Cctx]
    """
    # Ensure range is [0,1]
    img = image_chw.clamp(0, 1).unsqueeze(0).to(device=device, dtype=dtype)  # [1,C,H,W]
    # [-1,1]
    img = img * 2.0 - 1.0
    # Antialiasing resize identical to pipeline helper
    try:
        from pipelines.stereo_video_inpainting import _resize_with_antialiasing
        img = _resize_with_antialiasing(img, (224, 224))
    except Exception:
        # Fallback to simple bilinear if helper not available
        import torch.nn.functional as F
        img = F.interpolate(img, size=(224, 224), mode="bicubic", align_corners=True)
    # back to [0,1]
    img = (img + 1.0) / 2.0
    # Final CLIP normalization without rescale/resize
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
