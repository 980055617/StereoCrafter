#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- DS-SEED diagnostic (PREREG.txt section 3; never a selection input).  GPU.

What the consistency decoder's NOISE decides (and hence is not information from the latent) on the models' own latents.
Frames: the kept frames of windows 0, 4, 8 of <clip>_<latlabel> (36 frames).  cd is decoded with seeds 1 and 2 here; the
primary cd (seed 20261005), stock, stock32, ftmse, ftema rows are read from this lane's dev re-decode renders.
Regions in RENDER geometry from the model input (registration-free): BR = warped right eye at the deployed window
(splatting video BR quadrant), hole = splatting BL quadrant mean > 127.5 (score_registered_v1.py HOLEs), dilated by 8 px;
EDGE = top decile of BR gradient magnitude outside the dilated hole, FLAT = bottom half outside the dilated hole, HOLE = the
dilated hole.  Per pair: mean |a-b| (RGB, [0,1]) and mean |Laplacian(gray a - gray b)| (the high-frequency part) per region.
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python seed_diag_v1.py <out_dir> <clip> <latlabel>
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dswap_lib as DL  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402

OUTD, CLIP, LAT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}_{LAT}.json"
assert not os.path.exists(OJ), OJ
NPD = "/mnt/ssd_data/vae_20261005/decoder_swap/seed_diag"
os.makedirs(NPD, exist_ok=True)
RED = "/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev"
LD = f"/mnt/ssd_data/deep_20261004/decoder_ft/latents/{CLIP}_{LAT}"
TH, TW, DIL = 576, 1024, 8
T0 = time.time()
vs = VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0))
n = len(vs)
fm = DL.frame_map(n)
frames = sorted(t for t, (k, p) in fm.items() if k in (0, 4, 8))
S = vs.get_batch(frames).asnumpy()
Hs, Ws = S.shape[1] // 2, S.shape[2] // 2
st0, sl0 = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
BR = S[:, Hs + st0:Hs + st0 + TH, Ws + sl0:Ws + sl0 + TW]
HOLE = S[:, Hs + st0:Hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) > 127.5
k = np.ones((2 * DIL + 1, 2 * DIL + 1), np.uint8)
HD = np.stack([cv2.dilate(h.astype(np.uint8), k) > 0 for h in HOLE])
g = BR.astype(np.float32).mean(-1) / 255.
gm = np.zeros_like(g)
gm[:, :, :-1] += np.abs(np.diff(g, axis=2))
gm[:, :-1, :] += np.abs(np.diff(g, axis=1))
valid = ~HD
EDGE = valid & (gm >= np.quantile(gm[valid], 0.90))
FLAT = valid & (gm <= np.quantile(gm[valid], 0.50))
REG = dict(edge=EDGE, flat=FLAT, hole=HD)

rows = {}
for d in ("cd", "stock", "stock32", "ftmse", "ftema"):
    vr = VideoReader(f"{RED}/{CLIP}_{LAT}__{d}/{CLIP}_inpainting_results_sbs.mkv", ctx=cpu(0))
    a = vr.get_batch(frames).asnumpy()
    rows[d] = np.ascontiguousarray(a[:, :, a.shape[2] // 2:])
for s in (1, 2):
    p = f"{NPD}/{CLIP}_{LAT}__cd@{s}.npy"
    if os.path.exists(p):
        rows[f"cd@{s}"] = np.load(p)
    else:
        dec = DL.Decoder("cd", seed=s)
        rows[f"cd@{s}"] = DL.decode_frames(dec, LD, n, frames)
        np.save(p, rows[f"cd@{s}"])
        dec.unload()
        del dec


def lap(x):
    return np.abs(4 * x[:, 1:-1, 1:-1] - x[:, :-2, 1:-1] - x[:, 2:, 1:-1] - x[:, 1:-1, :-2] - x[:, 1:-1, 2:])


def stats(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1) / 255.
    hf = lap(a.astype(np.float32).mean(-1) / 255. - b.astype(np.float32).mean(-1) / 255.)
    out = {}
    for r, m in REG.items():
        out[r] = dict(mad=float(d[m].mean()), hf=float(hf[m[:, 1:-1, 1:-1]].mean()))
    out["all"] = dict(mad=float(d.mean()), hf=float(hf.mean()))
    return out


PAIRS = [("cd", "cd@1"), ("cd", "cd@2"), ("cd@1", "cd@2"), ("cd", "stock"), ("ftmse", "stock"), ("ftema", "stock"),
         ("stock32", "stock"), ("cd", "ftmse")]
res = dict(clip=CLIP, latlabel=LAT, frames=frames, region_frac={r: float(m.mean()) for r, m in REG.items()},
           pairs={f"{a} vs {b}": stats(rows[a], rows[b]) for a, b in PAIRS}, seconds=None)
for kk, v in res["pairs"].items():
    print(f"{CLIP} {LAT} {kk:16s} " + "  ".join(f"{r} mad {v[r]['mad']:.5f} hf {v[r]['hf']:.5f}" for r in ("edge", "flat", "hole", "all")),
          flush=True)
res["seconds"] = time.time() - T0
json.dump(res, open(OJ, "w"), indent=1)
print("SEED_DIAG_DONE", CLIP, LAT, flush=True)
