#!/usr/bin/env python
"""blur_diag (deep_20261004) -- VAE round trips through the pipeline's OWN methods.  Definitions: PREREG.txt.

Rows (per clip), each written as a single-view LOSSLESS FFV1 video of n_s frames, 576x1024:
  VAE_GT    registered GT -> bf16 VAE path
  VAE_GT32  registered GT -> fp32 VAE path
  VAE_GTx   registered GT bicubic-upsampled to 1024x1792 -> bf16 VAE path at 1024x1792 -> INTER_AREA to 576x1024
  RS_GTx    registered GT bicubic-upsampled to 1024x1792 -> uint8 -> INTER_AREA to 576x1024 (no VAE; control)
  VAE_BR    model input BR -> bf16 VAE path
VAE path (verbatim pipeline functions, called on a namespace holding the same objects the pipeline holds):
  image_processor.preprocess -> MambaStableVideoDiffusionInpaintingPipeline._encode_vae_frames(n_frames_per_time=5,
  CFG off; .latent_dist.mode()) -> (z * scaling_factor).to(vae dtype) -> ..._Pipe.decode_latents(decode_chunk_size=2)
  -> tensor2vid(output_type="pil") -> torch.tensor(np.array(img)).permute(2,0,1).float()/255 -> per-window keep rule of
  inpainting_inference.main (14-frame windows, overlap 3, keep generated[cur_overlap:] for i != 0) -> (x*255).to(uint8).
VAE loading verbatim from inpainting_inference.main: AutoencoderKLTemporalDecoder.from_pretrained(pre_trained_path,
  subfolder="vae", variant="fp16", torch_dtype=dtype), .to(dtype), .to("cuda"); enable_vae_memory_helpers is a no-op for
  this class (no enable_slicing / enable_tiling attribute -- checked).
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python vae_roundtrip_r2.py <clip> [rows] [--determinism]
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, REPO)
os.chdir(REPO)
_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)                       # FFV1 writer (_ffv1_write); patches nothing we use
import inpainting_inference as II                    # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder  # noqa: E402

PRE = "weights/stable-video-diffusion-img2vid-xt-1-1/"
CACHE = "/mnt/ssd_data/deep_20261004/blur_diag/cache_r2"
OUTR = "outputs/deep_20261004/blur_diag/vae_rt_r2"
TH, TW = 576, 1024
UH, UW = 1024, 1792
FC, OV, DCS = 14, 3, 2
CLIP = sys.argv[1]
ROWS = sys.argv[2].split(",") if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else \
    ["VAE_GT", "VAE_BR", "VAE_GT32", "VAE_GTx", "RS_GTx"]
DETERMINISM = "--determinism" in sys.argv
T0 = time.time()


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


def make_shim(dtype):
    vae = AutoencoderKLTemporalDecoder.from_pretrained(PRE, subfolder="vae", variant="fp16", torch_dtype=dtype)
    vae.requires_grad_(False)
    vae.to(dtype=dtype)
    vae = vae.to("cuda")
    assert not hasattr(vae, "enable_tiling") and not hasattr(vae, "enable_slicing")
    sf = 2 ** (len(vae.config.block_out_channels) - 1)
    shim = SimpleNamespace(vae=vae, vae_scale_factor=sf, image_processor=VaeImageProcessor(vae_scale_factor=sf))
    log(f"VAE loaded dtype={next(vae.parameters()).dtype} scaling_factor={vae.config.scaling_factor} vae_scale_factor={sf}")
    return shim


@torch.no_grad()
def vae_path(shim, frames01):
    """frames01: float32 CPU [n,3,h,w] in [0,1].  Returns uint8 [n,h,w,3] via the deployed window/keep/quantise path."""
    n, _, h, w = frames01.shape
    results, generated = [], None
    for i in range(0, n, FC - OV):
        if i + OV >= n:
            break
        if generated is not None and i + FC > n:
            cur_i = max(n + OV - FC, 0)
            cur_ov = i - cur_i + OV
        else:
            cur_i, cur_ov = i, OV
        x = frames01[cur_i:cur_i + FC]
        xp = shim.image_processor.preprocess(x, height=h, width=w)
        z = II._Pipe._encode_vae_frames(shim, xp, torch.device("cuda"), 1, False, n_frames_per_time=5)
        lat = (z.float() * shim.vae.config.scaling_factor).to(shim.vae.dtype)
        vf = II._Pipe.decode_latents(shim, lat, num_frames=lat.shape[1], decode_chunk_size=DCS)
        vf = II.tensor2vid(vf, shim.image_processor, output_type="pil")[0]
        g = torch.stack([torch.tensor(np.array(im)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0 for im in vf])
        generated = g
        if i != 0:
            generated = generated[cur_ov:]
        results.append(generated)
    out = torch.cat(results, dim=0)
    assert out.shape[0] == n, (out.shape, n)
    return (out * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu().numpy()


def to01(u8):
    return torch.from_numpy(u8).permute(0, 3, 1, 2).float() / 255.0


def upsample01(u8):
    """uint8 [n,576,1024,3] -> float [n,3,1024,1792] in [0,1] (bicubic, align_corners=False, clamped), chunked."""
    outs = []
    for s in range(0, len(u8), 16):
        outs.append(F.interpolate(to01(u8[s:s + 16]), size=(UH, UW), mode="bicubic", align_corners=False).clamp_(0, 1))
    return torch.cat(outs)


def area_down(u8_big):
    return np.stack([cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA) for f in u8_big])


def write(row, arr, fps, meta):
    os.makedirs(f"{OUTR}/{CLIP}", exist_ok=True)
    p = f"{OUTR}/{CLIP}/{CLIP}_{row}.mkv"
    assert not os.path.exists(p), f"refusing to overwrite {p}"
    dig = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
    _IL._ffv1_write(arr, fps, p)
    with open(p + ".md5", "w") as fh:
        fh.write(f"{dig}  {tuple(arr.shape)}  fps={fps:.6f}\n")
    meta[row] = dict(path=p, md5=dig, shape=list(arr.shape), seconds=time.time() - T0)
    log(f"wrote {row} {p} md5={dig} shape={arr.shape}")


def main():
    from decord import VideoReader, cpu
    fps = float(VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0)).get_avg_fps())
    cm = json.load(open(f"{CACHE}/{CLIP}/meta.json"))
    n_s = cm["n_s"]
    GT = np.load(f"{CACHE}/{CLIP}/GTreg.npy", mmap_mode="r")
    assert GT.shape == (n_s, TH, TW, 3)
    meta_p = f"{OUTR}/{CLIP}/meta_vae_rt{'_determinism' if DETERMINISM else ''}.json"
    os.makedirs(f"{OUTR}/{CLIP}", exist_ok=True)
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.setdefault("clip", CLIP)
    meta.setdefault("n_s", n_s)
    meta.setdefault("fps", fps)
    if DETERMINISM:
        shim = make_shim(torch.bfloat16)
        arr = vae_path(shim, to01(np.ascontiguousarray(GT)))
        dig = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
        ref = json.load(open(f"{OUTR}/{CLIP}/meta_vae_rt.json"))["VAE_GT"]["md5"]
        meta["VAE_GT_rerun"] = dict(md5=dig, ref=ref, identical=dig == ref)
        json.dump(meta, open(meta_p, "w"), indent=1)
        log(f"DETERMINISM VAE_GT rerun md5={dig} ref={ref} -> {'IDENTICAL' if dig == ref else 'DIFFERENT'}")
        return
    bf = [r for r in ROWS if r in ("VAE_GT", "VAE_BR", "VAE_GTx")]
    if bf:
        shim = make_shim(torch.bfloat16)
        for r in bf:
            if r in meta:
                log(f"{r} exists -> skip")
                continue
            if r == "VAE_GT":
                write(r, vae_path(shim, to01(np.ascontiguousarray(GT))), fps, meta)
            elif r == "VAE_BR":
                BR = np.load(f"{CACHE}/{CLIP}/BR.npy")
                write(r, vae_path(shim, to01(BR)), fps, meta)
                del BR
            elif r == "VAE_GTx":
                big = vae_path(shim, upsample01(np.ascontiguousarray(GT)))
                assert big.shape[1:3] == (UH, UW)
                write(r, area_down(big), fps, meta)
                del big
            json.dump(meta, open(meta_p, "w"), indent=1)
        del shim
        torch.cuda.empty_cache()
    if "VAE_GT32" in ROWS and "VAE_GT32" not in meta:
        shim = make_shim(torch.float32)
        write("VAE_GT32", vae_path(shim, to01(np.ascontiguousarray(GT))), fps, meta)
        json.dump(meta, open(meta_p, "w"), indent=1)
        del shim
        torch.cuda.empty_cache()
    if "RS_GTx" in ROWS and "RS_GTx" not in meta:
        up = upsample01(np.ascontiguousarray(GT))
        big = (up * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        write("RS_GTx", area_down(big), fps, meta)
        json.dump(meta, open(meta_p, "w"), indent=1)
    meta["total_seconds"] = time.time() - T0
    json.dump(meta, open(meta_p, "w"), indent=1)
    log("VAE_RT_DONE")


if __name__ == "__main__":
    main()
