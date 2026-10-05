#!/usr/bin/env python
"""deep_20261004 / sdedit lane: COPY of scripts/distill/runs/more_20261004/stripes/run_decomp_v1.py (md5 67f3d1940ca8c046d50cbf086ae60d39,
kept verbatim as run_decomp_v1_ORIG_COPY.py).  Changed ONLY: imports this lane's decomp_sdedit_v1 (12-clip offset table)
instead of decomp_v1.  Original docstring follows.
more_20261004 / stripes lane: M2 decomposition runner (CPU only; run with CUDA_VISIBLE_DEVICES="").

usage: run_decomp_v1.py OUT_JSON OUT_TXT [--input] clip:label=path [clip:label=path ...]
  Specs are grouped by clip.  For each clip: geometry() (frame choice + global GT registration, asserted against
  outputs/review_20261001/metrics.json where the review covered the clip), registered GT regions over
  frames range(0, nvalid, 8), then reviewlib.decompose() for the GT and for every render's right half.
  Every render row also carries stripeE split into GT-flat pixels within 2 px of a hole pixel (mask >= 0.5) vs
  farther.  --input adds the same decomposition of the model's warped right-eye INPUT (none / rowlin / telea).
  The origin_ll row (label must start with 'origin_ll') is compared with METRICS_WHOLEFRAME.txt (K5).
Refuses to overwrite OUT_JSON / OUT_TXT.
"""
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decomp_sdedit_v1 as D  # noqa: E402  (chdir to the repo happens in reviewlib)
R = D.R

OUTJ, OUTT = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
DO_INPUT = "--input" in args
args = [a for a in args if a != "--input"]
for p in (OUTJ, OUTT):
    if os.path.exists(p):
        sys.exit(f"refusing to overwrite {p}")

specs = OrderedDict()
for s in args:
    cl, rest = s.split(":", 1)
    lab, path = rest.split("=", 1)
    assert os.path.exists(path), path
    specs.setdefault(cl, []).append((lab, path))

REVIEW_TXT = f"{D.REPO}/outputs/review_20261001/METRICS_WHOLEFRAME.txt"


def review_origin_row(clip):
    txt = open(REVIEW_TXT).read()
    m = re.search(rf"=== {clip}  WHOLE 576x1024 WINDOW, n=(\d+) frames \(step 8\), GT regions = DISPARITY-REGISTERED ===\n"
                  r"(.*?)\n\n", txt + "\n\n", re.S)
    if not m:
        return None
    for line in m.group(2).splitlines():
        if line.startswith("origin (deployed)"):
            v = line[len("origin (deployed)"):].split()
            return dict(n=int(m.group(1)), haloFrac=float(v[0]), flatHF=float(v[2]), edgeHF=float(v[4]),
                        stripeE=float(v[6]))
    return None


out = {}
fh = open(OUTT, "w")


def tee(*s):
    print(*s, flush=True)
    print(*s, file=fh, flush=True)


HDR = (f"{'label':38s} {'haloFrac%':>9s} {'flatHF':>9s} {'flatHF/GT':>9s} {'edgeHF':>9s} {'edgeHF/GT':>9s} "
       f"{'stripeE':>9s} {'stripeE/GT':>10s} {'strNear':>9s} {'strFar':>9s}")


def row(lab, d, g):
    return (f"{lab[:38]:38s} {d['haloFrac']:9.3f} {d['flatHF']:9.5f} {d['flatHF']/g['flatHF']:9.3f} "
            f"{d['edgeHF']:9.5f} {d['edgeHF']/g['edgeHF']:9.3f} {d['stripeE']:9.5f} {d['stripeE']/g['stripeE']:10.3f} "
            f"{d.get('stripeE_nearHole', float('nan')):9.5f} {d.get('stripeE_farHole', float('nan')):9.5f}")


for clip, lst in specs.items():
    geo = D.geometry(clip, [p for _, p in lst])
    frames = D.frames_for(geo)
    tee(f"\n=== {clip}  WHOLE 576x1024 WINDOW, n={len(frames)} frames (step {D.WHOLE_STEP}), GT regions = "
        f"DISPARITY-REGISTERED, shift {geo['gtShift']} (frame {geo['frame']}; {geo.get('review_check', 'not in review')}) ===")
    gts = D.gt_stack(clip, geo, frames)
    reg = R.regions(gts)
    gd = R.decompose(reg, gts)
    rows = OrderedDict(GT=gd)
    # hole proximity map (same frames) from the input pass with mode none
    ind, near = D.decompose_input(clip, reg, frames, "none")
    if DO_INPUT:
        rows["INPUT warped (none)"] = ind
        for mode in ("rowlin", "telea"):
            rows[f"INPUT warped ({mode})"], _ = D.decompose_input(clip, reg, frames, mode)
    for lab, p in lst:
        y = D.render_stack(p, frames)
        d = R.decompose(reg, y)
        d["stripeE_nearHole"], d["stripeE_farHole"] = D.split_stripe(reg, y, near)
        d["path"] = p
        rows[lab] = d
        del y
    tee(HDR)
    for lab, d in rows.items():
        tee(row(lab, d, gd))
    lo = reg["lo"][:, :, :-1]
    nr = near[:, :, :-1] | near[:, :, 1:]
    tee(f"  GT-flat pixels within 2 px of a hole: {100.0 * (lo & nr).sum() / lo.sum():.3f}%")
    rv = review_origin_row(clip)
    chk = None
    if rv is not None:
        o = [lab for lab in rows if lab.startswith("origin_ll")]
        if o:
            d = rows[o[0]]
            same = all(abs(round(d[k], 5) - rv[k]) < 0.6e-5 for k in ("stripeE", "edgeHF", "flatHF")) and \
                abs(round(d["haloFrac"], 3) - rv["haloFrac"]) < 0.6e-3 and rv["n"] == len(frames)
            chk = dict(review=rv, ours={k: d[k] for k in ("haloFrac", "flatHF", "edgeHF", "stripeE")}, reproduced=bool(same))
            tee(f"  K5 decomposition reproduction vs METRICS_WHOLEFRAME.txt (origin row): "
                f"{'PASS' if same else 'FAIL'}  review {rv}  ours "
                f"{ {k: round(d[k], 5) for k in ('haloFrac', 'flatHF', 'edgeHF', 'stripeE')} }")
    out[clip] = dict(geometry=geo, frames=frames, rows=rows, k5=chk)
    json.dump(out, open(OUTJ, "w"), indent=1, default=float)
    del gts, reg

tee("\n" + R.LEGEND)
tee("  strNear / strFar = stripeE restricted to GT-flat pixels within 2 px of a hole pixel (mask>=0.5) / farther")
fh.close()
print("DONE")
