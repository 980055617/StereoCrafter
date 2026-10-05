#!/usr/bin/env python
"""blur_diag PREREG ADDENDUM 4 reading (NR-CAL).  CPU only.
For each calibration clip pick the Gaussian sigma whose b1/GT is nearest to ORIGIN's 12-clip mean b1/GT (from
TABLE_BLUR_DIAG_r2.json), take the NR change GT -> that sigma, average over the 4 calibration clips, and compare with
the seed yardstick Y (TABLE_BLUR_DIAG_r2.json) in the direction of worse quality.  Also prints per-clip signs and the
monotonicity of each metric over sigma.
usage: python nrcal_analyze_r2.py <nr_calib.json> <TABLE_BLUR_DIAG_r2.json> <out_txt>"""
import json
import sys

import numpy as np

cal = json.load(open(sys.argv[1]))
tab = json.load(open(sys.argv[2]))
out = []
target = tab["decomposition"]["ALL"]["ORIGIN"]["gt_b1"]
out.append(f"NR-CAL (PREREG ADDENDUM 4).  ORIGIN 12-clip mean b1/GT = {target:.3f}  (source TABLE_BLUR_DIAG_r2.json)")
out.append(f"calibration data: {sys.argv[1]}")
lower = {"musiq": False, "clipiqa": False, "niqe": True}
res = {}
for m in cal["metrics"]:
    ch, signs, mono = [], [], []
    for c in cal["clips"]:
        rows = cal["rows"][c]
        sig = min((s for s in cal["sigmas"] if s > 0), key=lambda s: abs(rows[f"sigma_{s}"]["b1_ratio"] - target))
        d = rows[f"sigma_{sig}"][m] - rows["sigma_0.0"][m]
        worse = d > 0 if lower[m] else d < 0
        ch.append(d)
        signs.append(f"{c}: sigma {sig} (b1 {rows[f'sigma_{sig}']['b1_ratio']:.2f}) d {d:+.4f} {'worse' if worse else 'NOT worse'}")
        seq = [rows[f"sigma_{s}"][m] for s in cal["sigmas"]]
        dif = np.diff(seq) * (1 if lower[m] else -1)
        mono.append(bool((dif > 0).all()))
    y = tab["yardstick"][m]["max"]
    mean_d = float(np.mean(ch))
    reg = (mean_d > y) if lower[m] else (-mean_d > y)
    res[m] = dict(mean_change=mean_d, Y=y, registers=reg, per_clip=signs, monotone_all_clips=all(mono),
                  n_clips_worse=sum(1 for s in signs if "NOT" not in s))
    out.append(f"  {m:8s} mean change GT->matched blur {mean_d:+.4f} (Y {y:.4f}) -> "
               f"{'REGISTERS blur at origin level' if reg else 'does NOT register'}; worse on "
               f"{res[m]['n_clips_worse']}/4 clips; monotone in sigma on all clips: {all(mono)}")
    for s in signs:
        out.append(f"      {s}")
open(sys.argv[3], "w").write("\n".join(out) + "\n")
json.dump(res, open(sys.argv[3].replace(".txt", ".json"), "w"), indent=1)
print("\n".join(out))
