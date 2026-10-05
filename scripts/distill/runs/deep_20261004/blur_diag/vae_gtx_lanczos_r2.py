#!/usr/bin/env python
"""blur_diag PREREG ADDENDUM 5: VAE_GTx_L = the VAE_GTx path (vae_roundtrip_r2.py, imported, NOT modified) with the
1024x1792 output downsampled by PIL Image.LANCZOS instead of cv2.INTER_AREA.  The INTER_AREA version is re-derived from
the same hi-res output and must reproduce the md5 recorded for VAE_GTx (path-consistency check).
Writes outputs/deep_20261004/blur_diag/lanczos_r2/<clip>/<clip>_VAE_GTx_L.mkv (+ .md5) and <clip>_VAE_GTx_L_meta.json.
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python vae_gtx_lanczos_r2.py <clip> [...]"""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = sys.argv[1:]
sys.argv = [sys.argv[0], CLIPS[0], "VAE_GTx"]          # vae_roundtrip_r2 reads argv at import (CLIP/ROWS globals)
sys.path.insert(0, HERE)
import torch  # noqa: E402
import vae_roundtrip_r2 as V  # noqa: E402

O = "outputs/deep_20261004/blur_diag"


def lanczos_down(u8):
    return np.stack([np.asarray(Image.fromarray(f).resize((V.TW, V.TH), Image.LANCZOS)) for f in u8])


shim = V.make_shim(torch.bfloat16)
for clip in CLIPS:
    dst = f"{O}/lanczos_r2/{clip}/{clip}_VAE_GTx_L.mkv"
    if os.path.exists(dst):
        print(f"{clip}: exists -> skip", flush=True)
        continue
    rec = json.load(open(f"{O}/vae_rt_r2/{clip}/meta_vae_rt.json"))
    GT = np.load(f"{V.CACHE}/{clip}/GTreg.npy")
    big = V.vae_path(shim, V.upsample01(GT))
    area = V.area_down(big)
    md5_area = hashlib.md5(np.ascontiguousarray(area).tobytes()).hexdigest()
    ok = md5_area == rec["VAE_GTx"]["md5"]
    print(f"{clip}: INTER_AREA re-derivation md5 {md5_area} vs recorded {rec['VAE_GTx']['md5']} -> "
          f"{'CONSISTENT' if ok else 'MISMATCH'}", flush=True)
    lz = lanczos_down(big)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    V._IL._ffv1_write(lz, rec["fps"], dst)
    dig = hashlib.md5(np.ascontiguousarray(lz).tobytes()).hexdigest()
    open(dst + ".md5", "w").write(f"{dig}  {tuple(lz.shape)}  fps={rec['fps']:.6f}\n")
    json.dump(dict(clip=clip, md5=dig, area_md5=md5_area, area_md5_recorded=rec["VAE_GTx"]["md5"], consistent=ok),
              open(f"{O}/lanczos_r2/{clip}/{clip}_VAE_GTx_L_meta.json", "w"), indent=1)
    print(f"{clip}: wrote {dst} md5 {dig}", flush=True)
    del big, area, lz, GT
print("VAE_GTX_L_DONE", flush=True)
