#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- extract the REAL right eye at the deployed window, all frames (CPU only).

For the decoder-only round-trip diagnostic (GT -> deployed encode -> decoder -> GT; registration-free because the target is
the frame itself).  Window = the deployed crop of the splatting tile (quadrant cropped to /128, centred 576x1024), applied
to the TR quadrant of video_data/train/<clip>_train.mp4 (quadrant sizes asserted equal, as score_registered_v1.py does).
Validity (task hard rule): clip int < 0310, readlink -f not into train_leftGT_broken, real right eye != left eye
(mean PSNR(TL crop, TR crop) over sampled frames must be < 30 dB; a left-eye copy gives >= 35 dB).
writes <out>/<clip>_TR.npy uint8 [n_t,576,1024,3] and <out>/<clip>_TR.json (n, md5, psnr_lr)
usage: python extract_gt_v1.py <out_dir> <clip,clip,...>
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
TH, TW = 576, 1024
OUT, CL = sys.argv[1], sys.argv[2].split(",")
os.makedirs(OUT, exist_ok=True)
for c in CL:
    op = f"{OUT}/{c}_TR.npy"
    if os.path.exists(op):
        print("SKIP", op, flush=True)
        continue
    assert int(c) < 310, c
    rp = os.path.realpath(f"video_data/train/{c}_train.mp4")
    assert "train_leftGT_broken" not in rp, (c, rp)
    vt = VideoReader(rp, ctx=cpu(0))
    vs = VideoReader(f"video_data/splatting/{c}_splatting_results.mp4", ctx=cpu(0))
    f0, s0 = vt[0].asnumpy(), vs[0].asnumpy()
    H, W = f0.shape[0] // 2, f0.shape[1] // 2
    assert (H, W) == (s0.shape[0] // 2, s0.shape[1] // 2), (c, f0.shape, s0.shape)
    t0, l0 = (H // 128 * 128 - TH) // 2, (W // 128 * 128 - TW) // 2
    n = len(vt)
    TR = np.empty((n, TH, TW, 3), np.uint8)
    ps = []
    for s in range(0, n, 16):
        idx = list(range(s, min(n, s + 16)))
        b = vt.get_batch(idx).asnumpy()
        for j, fi in enumerate(idx):
            TR[fi] = b[j, t0:t0 + TH, W + l0:W + l0 + TW]
            if fi % 10 == 0:
                tl = b[j, t0:t0 + TH, l0:l0 + TW].astype(np.float64)
                mse = ((tl - TR[fi].astype(np.float64)) ** 2).mean()
                ps.append(10 * math.log10(255.0 ** 2 / max(mse, 1e-9)))
    psnr_lr = float(np.mean(ps))
    assert psnr_lr < 30.0, f"{c}: right eye looks like a copy of the left eye (PSNR {psnr_lr:.1f} dB)"
    np.save(op, TR)
    dig = hashlib.md5(TR.tobytes()).hexdigest()
    json.dump(dict(clip=c, src=rp, n=n, window=[t0, l0], quadrant=[H, W], md5=dig, psnr_left_vs_right_db=psnr_lr),
              open(f"{OUT}/{c}_TR.json", "w"), indent=1)
    print(f"{c}: n={n} window ({t0},{l0}) quadrant {H}x{W} PSNR(left,right) {psnr_lr:.2f} dB md5 {dig}", flush=True)
print("EXTRACT_DONE", flush=True)
