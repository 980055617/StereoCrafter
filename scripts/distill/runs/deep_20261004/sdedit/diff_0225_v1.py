#!/usr/bin/env python
"""PREREG_ADDENDUM_3 report line (CPU): mean |deliv sd1fill - deliv sd1| of the right halves on the zero-hole clip 0225,
in 8-bit levels, over all frames, next to |deliv sd1 - deliverable deployed| for scale.  usage: diff_0225_v1.py OUT.txt"""
import os
import sys

import numpy as np
from decord import VideoReader, cpu

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}")
P = {"sd1fill": "outputs/deep_20261004/sdedit/clips/0225_deliv_g100_sd1fill/0225_inpainting_results_sbs.mkv",
     "sd1": "outputs/deep_20261004/sdedit/clips/0225_deliv_g100_sd1/0225_inpainting_results_sbs.mkv",
     "deployed": "outputs/beyond_distil_mamba_scaled/clips/0225_mstudent2_step800_deliv_ll/0225_inpainting_results_sbs.mkv"}
V = {k: VideoReader(p, ctx=cpu(0)) for k, p in P.items()}
n = min(len(v) for v in V.values())
acc = {"sd1fill-sd1": [0.0, 0.0], "sd1-deployed": [0.0, 0.0]}
for s in range(0, n, 16):
    idx = list(range(s, min(n, s + 16)))
    F = {k: v.get_batch(idx).asnumpy()[:, :, 1024:].astype(np.float32) for k, v in V.items()}
    for key, (a, b) in (("sd1fill-sd1", ("sd1fill", "sd1")), ("sd1-deployed", ("sd1", "deployed"))):
        d = np.abs(F[a] - F[b])
        acc[key][0] += float(d.sum())
        acc[key][1] += d.size
lines = [f"0225 (zero-hole clip), right halves, {n} frames, mean |difference| in 8-bit levels:"]
for k, (s_, c_) in acc.items():
    lines.append(f"  {k:14s} {s_ / c_:.4f}")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
