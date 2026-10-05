#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- DECODE-PAIR FLICKER ratio (CPU).  PREREG.txt guard g5.

The deployed decoder runs with decode_chunk_size=2: inside each 14-frame window the TemporalDecoder sees frame pairs
(local 0,1), (2,3), ..., (12,13).  A decoder change could make frames differ more ACROSS pair boundaries than within a pair
(period-2 flicker), which the window-seam metric does not see.  For consecutive kept frames (t, t+1) of the SAME window
(window grid of inpainting_inference.main, frames_chunk 14, overlap 3): local l = t - cur_i; within-pair if l even,
across-pair if l odd.  d_t = mean |R_{t+1} - R_t| (RGB in [0,1]).  ratio = mean_across / mean_within.  The same index sets
on the real right eye at the deployed window (no pair structure) give the reference ratio.
usage: python score_pairflicker_v1.py <out_json> <clip> <label=sbs_path> [<label=sbs_path> ...]
"""
import json
import os
import sys

import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
TH, TW = 576, 1024
FC, OV = 14, 3
OUTJ, CLIP, specs = sys.argv[1], sys.argv[2], [s.split("=", 1) for s in sys.argv[3:]]
assert not os.path.exists(OUTJ), f"refusing to overwrite {OUTJ}"


def windows(n):
    out, gen = [], False
    for i in range(0, n, FC - OV):
        if i + OV >= n:
            break
        if gen and i + FC > n:
            cur_i = max(n + OV - FC, 0)
            cur_ov = i - cur_i + OV
        else:
            cur_i, cur_ov = i, OV
        out.append((i, cur_i, cur_ov))
        gen = True
    return out


def sets(n):
    within, across = [], []
    for (i, cur_i, cur_ov) in windows(n):
        first = cur_i if i == 0 else cur_i + cur_ov
        last = min(cur_i + FC, n) - 1
        for t in range(first, last):
            (within if (t - cur_i) % 2 == 0 else across).append(t)
    return within, across


def diffs(R):
    x = R.astype(np.float32) / 255.0
    return np.array([float(np.abs(x[t + 1] - x[t]).mean()) for t in range(len(x) - 1)])


res = dict(clip=CLIP, labels={})
n_ref = None
for lab, p in specs:
    vr = VideoReader(p, ctx=cpu(0))
    n = len(vr)
    a = vr.get_batch(list(range(n))).asnumpy()
    R = a[:, :, a.shape[2] // 2:]
    w, c = sets(n)
    d = diffs(R)
    res["labels"][lab] = dict(path=p, n=n, within=float(d[w].mean()), across=float(d[c].mean()),
                              ratio=float(d[c].mean() / d[w].mean()), n_within=len(w), n_across=len(c))
    n_ref = n if n_ref is None else n_ref
    print(f"{CLIP} {lab:40s} within {d[w].mean():.5f} across {d[c].mean():.5f} ratio {d[c].mean() / d[w].mean():.4f}",
          flush=True)
# reference: real right eye at the deployed window (unregistered), same index sets
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
f0 = vt[0].asnumpy()
H, W = f0.shape[0] // 2, f0.shape[1] // 2
t0, l0 = (H // 128 * 128 - TH) // 2, (W // 128 * 128 - TW) // 2
n = min(n_ref, len(vt))
G = np.stack([vt[k].asnumpy()[t0:t0 + TH, W + l0:W + l0 + TW] for k in range(n)])
w, c = sets(n)
d = diffs(G)
res["GT"] = dict(n=n, within=float(d[w].mean()), across=float(d[c].mean()), ratio=float(d[c].mean() / d[w].mean()))
print(f"{CLIP} {'GT (real right eye)':40s} within {d[w].mean():.5f} across {d[c].mean():.5f} ratio "
      f"{d[c].mean() / d[w].mean():.4f}", flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
print("PAIRFLICKER_DONE", CLIP, flush=True)
