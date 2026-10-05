#!/usr/bin/env python
"""blur_diag PREREG ADDENDUM 2 (R4-L, filter sensitivity).  CPU only.
  RS_GTx_L   registered GT (cache, all n_s frames) -> bicubic up to 1024x1792 (identical to vae_roundtrip_r2.upsample01)
             -> round to uint8 -> PIL Image.LANCZOS (antialiased) down to 576x1024
             -> outputs/deep_20261004/blur_diag/lanczos_r2/<clip>/<clip>_RS_GTx_L.mkv  (single view, FFV1)
  HIRES_B_L  (only if the HIRES_B render exists) its saved full-resolution right eye
             <clip>_inpainting_results_hires_right_1024x1792.mkv -> PIL LANCZOS down -> paired with the HIRES_B left half
             -> outputs/deep_20261004/blur_diag/hiresB_r2/lanczos/<clip>_origin_upx175_L/<clip>_inpainting_results_sbs.mkv
usage: python make_lanczos_rows_r2.py [--rs] [--hb] <clip> [...]"""
import hashlib
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
CACHE = "/mnt/ssd_data/deep_20261004/blur_diag/cache_r2"
O = "outputs/deep_20261004/blur_diag"
TH, TW, UH, UW = 576, 1024, 1024, 1792
FF = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"


def ffv1(arr, fps, path):
    import subprocess
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{arr.shape[2]}x{arr.shape[1]}",
           "-r", f"{float(fps):.6f}", "-i", "-", "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1",
           "-threads", "8", "-pix_fmt", "bgr0", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in arr:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    dig = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
    open(path + ".md5", "w").write(f"{dig}  {tuple(arr.shape)}  fps={float(fps):.6f}\n")
    print(f"wrote {path} md5 {dig} {arr.shape}", flush=True)


def lanczos_down(u8):
    return np.stack([np.asarray(Image.fromarray(f).resize((TW, TH), Image.LANCZOS)) for f in u8])


args = [a for a in sys.argv[1:] if not a.startswith("--")]
DO_RS = "--rs" in sys.argv or ("--hb" not in sys.argv)
DO_HB = "--hb" in sys.argv or ("--rs" not in sys.argv)
for clip in args:
    fps = float(VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0)).get_avg_fps())
    if DO_RS:
        dst = f"{O}/lanczos_r2/{clip}/{clip}_RS_GTx_L.mkv"
        if os.path.exists(dst):
            print(f"{clip}: RS_GTx_L exists -> skip")
        else:
            GT = np.load(f"{CACHE}/{clip}/GTreg.npy")
            out = np.empty_like(GT)
            for s in range(0, len(GT), 16):
                x = torch.from_numpy(GT[s:s + 16]).permute(0, 3, 1, 2).float() / 255.0
                up = F.interpolate(x, size=(UH, UW), mode="bicubic", align_corners=False).clamp_(0, 1)
                big = (up * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                out[s:s + 16] = lanczos_down(big)
            ffv1(out, fps, dst)
    if DO_HB:
        od = f"{O}/hiresB_r2/clips/{clip}_origin_upx175"
        hr = f"{od}/{clip}_inpainting_results_hires_right_{UH}x{UW}.mkv"
        sbs = f"{od}/{clip}_inpainting_results_sbs.mkv"
        dst = f"{O}/hiresB_r2/lanczos/{clip}_origin_upx175_L/{clip}_inpainting_results_sbs.mkv"
        if not (os.path.exists(hr) and os.path.exists(sbs)):
            print(f"{clip}: no HIRES_B render -> HIRES_B_L skipped")
        elif os.path.exists(dst):
            print(f"{clip}: HIRES_B_L exists -> skip")
        else:
            a, b = VideoReader(hr, ctx=cpu(0)), VideoReader(sbs, ctx=cpu(0))
            n = len(a)
            assert n == len(b), (n, len(b))
            out = np.empty((n, TH, 2 * TW, 3), np.uint8)
            for s in range(0, n, 16):
                idx = list(range(s, min(n, s + 16)))
                R = a.get_batch(idx).asnumpy()
                assert R.shape[1:] == (UH, UW, 3), R.shape
                out[s:s + len(idx), :, :TW] = b.get_batch(idx).asnumpy()[:, :, :TW]
                out[s:s + len(idx), :, TW:] = lanczos_down(R)
            ffv1(out, fps, dst)
