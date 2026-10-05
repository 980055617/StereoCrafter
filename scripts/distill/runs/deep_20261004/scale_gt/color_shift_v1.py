#!/usr/bin/env python
"""scale_gt DIAGNOSTIC: inter-ocular colour shift.  For every render <clip>_<label> under outputs/deep_20261004/scale_gt/renders
(and optional extra "clip=label=path" specs), mean RGB of the generated RIGHT half minus the mean RGB of the reference render's right
half (same clip), every 8th frame, 0-255 units.  Also the training-data offset (registered real right eye - warped input,
non-hole pixels) from phase A, for comparison.  CPU only.
usage: python color_shift_v1.py <out_txt> <ref_label> <clip,clip,...> <label,label,...> [clip=label=path ...]
"""
import json, os, sys
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT, REF, CLIPS, LABS = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
extra = {}
for a in sys.argv[5:]:
    c, l, p = a.split("=", 2); extra[(c, l)] = p
R = "outputs/deep_20261004/scale_gt/renders"
def path(c, l):
    return extra.get((c, l), f"{R}/{c}_{l}/{c}_inpainting_results_sbs.mkv")
def mean_rgb(p):
    vr = VideoReader(p, ctx=cpu(0)); f = vr.get_batch(list(range(0, len(vr), 8))).asnumpy().astype(np.float64)
    h = f.shape[2] // 2
    return f[:, :, h:].reshape(-1, 3).mean(0), f[:, :, :h].reshape(-1, 3).mean(0)
lines = []
L = "scripts/distill/runs/deep_20261004/scale_gt"
s = json.load(open(f"{L}/spec_main_v1.json")); used = {(c, int(w)) for c, w in s["windows"]}
offs = []
for c in sorted({c for c, _ in used}):
    j = json.load(open(f"/mnt/ssd_data/deep_20261004/scale_gt/cache_v1/crops/{c}/clip.json"))
    offs += [ws["rgb_offset_255"] for ws in j["windows_stats"] if (c, ws["start"]) in used]
offs = np.array(offs)
lines.append(f"training data (MAIN windows, n={len(offs)}): registered real right eye - warped input, mean RGB offset {offs.mean(0).round(2).tolist()} "
             f"(median {np.median(offs, 0).round(2).tolist()})")
lines.append(f"render right-eye mean RGB minus {REF} right-eye mean RGB (every 8th frame), and left-eye check (must be 0)")
agg = {l: [] for l in LABS}
for c in CLIPS:
    rr, rl = mean_rgb(path(c, REF))
    cells = []
    for l in LABS:
        p = path(c, l)
        if not os.path.exists(p): cells.append(f"{l}: -"); continue
        r, lft = mean_rgb(p); d = r - rr; agg[l].append(d)
        cells.append(f"{l}: {d.round(2).tolist()} (left {float(np.abs(lft - rl).max()):.2f})")
    lines.append(f"{c}  " + " | ".join(cells))
for l, v in agg.items():
    if v: lines.append(f"MEAN over {len(v)} clips  {l}: {np.mean(v, 0).round(2).tolist()}")
open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
