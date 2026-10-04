#!/usr/bin/env python
"""[v2: clip, registered offset and frame count from argv; usage: OUT CLIP DDY DDX NFRAMES label=path ...]
Whole-frame artefact/detail decomposition for 0301 along the sampler axis (CPU only).
Operators: scripts/distill/runs/review_20261001/reviewlib.py (regions/decompose, verbatim ringing_metrics.py defs),
whole 576x1024 window, frames 0..150 step 8 (n=19), GT regions from the DISPARITY-REGISTERED real right eye at
(ddy,ddx)=(1,-15) relative to the scorer window (t0,l0)=(224,448) -- the review's own GEOMETRY.txt values for 0301.
Reproduction check: the origin / s25 rows must equal review_20261001/METRICS_WHOLEFRAME.txt (0301, registered):
stripeE 0.01686 / 0.02183, flatHF 0.03126 / 0.03646, edgeHF 0.07743 / 0.08322.
usage: stripe_decomp_v1.py OUT.txt label=path.mkv ..."""
import sys, os
import numpy as np
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/review_20261001")
import reviewlib as R          # chdirs to the repo
clip = sys.argv[2]
t0, l0, H, W = R.window(clip)
print("window", t0, l0)
gddy, gddx = int(sys.argv[3]), int(sys.argv[4])
frames = list(range(0, int(sys.argv[5]), 8))
gt = []
for f in frames:
    TL, TR, BL, BR, _, _ = R.tile_quadrants(clip, f)
    gt.append(R.gray(TR[t0 + gddy:t0 + gddy + R.TH, l0 + gddx:l0 + gddx + R.TW]))
gt = np.stack(gt)
reg = R.regions(gt)
g = R.decompose(reg, gt)
lines = [f"=== {clip} WHOLE 576x1024 WINDOW, n={len(frames)} frames (step 8), GT regions = DISPARITY-REGISTERED ({gddy},{gddx}) ===",
         R.HDR, R.fmt_row("GT", g, g)]
for spec in sys.argv[6:]:
    lab, p = spec.split("=", 1)
    y = np.stack([R.gray(R.grab(p, f)[:, R.TW:]) for f in frames])
    lines.append(R.fmt_row(lab, R.decompose(reg, y), g))
lines.append(R.LEGEND)
txt = "\n".join(lines) + "\n"
open(sys.argv[1], "a").write(txt); print(txt)
