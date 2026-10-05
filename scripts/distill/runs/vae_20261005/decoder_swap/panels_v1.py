#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- STEP 6 visual panels (PREREG.txt section 5).  CPU only.

Per clip, at frame F (default 76) of the <latlabel> renders: two 384x384 crops at 100 % (no resampling), chosen from the
REGISTERED real right eye (REG_FRAME smoothed shift of that frame, from this lane's registered-scorer json) and the hole mask
only -- never from a candidate image:
  E = window (stride 32) with the highest fraction of GT top-decile-gradient pixels (gradient = |dx|+|dy| of gray GT);
  F = window with the highest fraction of GT bottom-half-gradient pixels among windows with >= 5 % disocclusion pixels
      (hole = splatting BL quadrant mean > 127.5 at the deployed window); if none, the highest flat fraction overall.
Writes <out_dir>/<clip>_f<F>_<E|F>_main.png (GT | stock | <best>) and _all.png (GT | stock | stock32 | ftmse | ftema | cd),
plus <out_dir>/crops.json.  Labels are drawn in a strip ABOVE the crops (no pixel of a crop is covered).
usage: python panels_v1.py <out_dir> <redec_root> <regscore_dir> <best> <latlabel> <frame> <clip,clip,...>
"""
import json
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
OUTD, RED, REGD, BEST, LAT, FR, CLIPS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7].split(",")
os.makedirs(OUTD, exist_ok=True)
CJ = f"{OUTD}/crops.json"
assert not os.path.exists(CJ), CJ
TH, TW, CS, STRIDE = 576, 1024, 384, 32
ALLROWS = ["stock", "stock32", "ftmse", "ftema", "cd"]


def right(clip, d):
    vr = VideoReader(f"{RED}/{clip}_{LAT}__{d}/{clip}_inpainting_results_sbs.mkv", ctx=cpu(0))
    a = vr[FR].asnumpy()
    return np.ascontiguousarray(a[:, a.shape[1] // 2:])


def strip(labels, w):
    s = np.zeros((28, w * len(labels), 3), np.uint8)
    for i, t in enumerate(labels):
        cv2.putText(s, t, (i * w + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
    return s


def save(path, tiles, labels):
    sep = np.full((CS, 4, 3), 255, np.uint8)
    body = tiles[0]
    for t in tiles[1:]:
        body = np.concatenate([body, sep, t], 1)
    img = np.concatenate([strip(labels, CS + 4)[:, :body.shape[1]], body], 0)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


out = dict(frame=FR, latlabel=LAT, best=BEST, crops={})
for clip in CLIPS:
    R = json.load(open(f"{REGD}/{clip}.json"))
    t0, l0 = R["window"]
    H, W = R["quadrant"]
    ddy, ddx = int(R["reg"]["smooth_ddy"][FR]), int(R["reg"]["smooth_ddx"][FR])
    tile = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))[FR].asnumpy()
    GT = np.ascontiguousarray(tile[t0 + ddy:t0 + ddy + TH, W + l0 + ddx:W + l0 + ddx + TW])
    vs = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))
    s = vs[FR].asnumpy()
    Hs, Ws = s.shape[0] // 2, s.shape[1] // 2
    st0, sl0 = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
    hole = s[Hs + st0:Hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) > 127.5
    g = GT.astype(np.float32).mean(-1) / 255.
    gm = np.zeros_like(g)
    gm[:, :-1] += np.abs(np.diff(g, axis=1))
    gm[:-1, :] += np.abs(np.diff(g, axis=0))
    hi, lo = gm >= np.quantile(gm, 0.90), gm <= np.quantile(gm, 0.50)
    cand = [(y, x) for y in range(0, TH - CS + 1, STRIDE) for x in range(0, TW - CS + 1, STRIDE)]
    frac = lambda m, y, x: float(m[y:y + CS, x:x + CS].mean())
    E = max(cand, key=lambda p: frac(hi, *p))
    withhole = [p for p in cand if frac(hole, *p) >= 0.05]
    F = max(withhole or cand, key=lambda p: frac(lo, *p))
    out["crops"][clip] = dict(reg_shift=[ddy, ddx], E=dict(yx=list(E), edge_frac=frac(hi, *E), hole_frac=frac(hole, *E)),
                              F=dict(yx=list(F), flat_frac=frac(lo, *F), hole_frac=frac(hole, *F),
                                     rule="flat among >=5% hole" if withhole else "flat overall (no window with >=5% hole)"))
    imgs = {d: right(clip, d) for d in ALLROWS}
    for nm, (y, x) in (("E", E), ("F", F)):
        c = lambda a: np.ascontiguousarray(a[y:y + CS, x:x + CS])
        save(f"{OUTD}/{clip}_f{FR:03d}_{nm}_main.png", [c(GT), c(imgs["stock"]), c(imgs[BEST])],
             [f"GT reg ({ddy:+d},{ddx:+d})", "stock (deployed)", BEST])
        save(f"{OUTD}/{clip}_f{FR:03d}_{nm}_all.png", [c(GT)] + [c(imgs[d]) for d in ALLROWS], ["GT reg"] + ALLROWS)
        save(f"{OUTD}/{clip}_f{FR:03d}_{nm}_holemask.png",
             [np.repeat((hole[y:y + CS, x:x + CS, None] * 255).astype(np.uint8), 3, 2)], ["disocclusion mask"])
    print(f"{clip}: E {E} edge {frac(hi, *E):.3f} hole {frac(hole, *E):.3f} | F {F} flat {frac(lo, *F):.3f} hole {frac(hole, *F):.3f}",
          flush=True)
json.dump(out, open(CJ, "w"), indent=1)
print("PANELS_DONE", flush=True)
