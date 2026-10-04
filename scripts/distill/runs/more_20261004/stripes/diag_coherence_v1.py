#!/usr/bin/env python
"""DIAGNOSTIC (not pre-registered as a criterion; reported only): is the flat-region horizontal-difference energy
(reviewlib stripeE) VERTICALLY COHERENT (= vertical stripes) or incoherent (grain / noise / compression)?

For GT-flat pixels (registered GT regions, same frames as decomp_v1) that are > 2 px from any hole pixel,
dx = y[:, :, x+1] - y[:, :, x] is grouped into vertical 8-row blocks (rows 8k..8k+7 of one column) that are
entirely flat-and-far.  For each block: T = mean_k |dx|, C = |mean_k dx|.
  coherence = sum C / sum T     (iid Gaussian noise -> ~0.35 ; a perfect vertical stripe -> 1.0)
  coherentE = mean C            (energy of the vertically coherent part, same units as stripeE)
usage: diag_coherence_v1.py OUT_JSON OUT_TXT clip:label=path [...]     (CUDA_VISIBLE_DEVICES="" ; CPU only)
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decomp_v1 as D  # noqa: E402
R = D.R

OUTJ, OUTT = sys.argv[1], sys.argv[2]
for p in (OUTJ, OUTT):
    if os.path.exists(p):
        sys.exit(f"refusing to overwrite {p}")
specs = OrderedDict()
for s in sys.argv[3:]:
    cl, rest = s.split(":", 1)
    lab, path = rest.split("=", 1)
    specs.setdefault(cl, []).append((lab, path))
out = {}
fh = open(OUTT, "w")


def tee(*a):
    print(*a, flush=True)
    print(*a, file=fh, flush=True)


def coh(y, sel):
    n, H, W = y.shape
    dx = np.diff(y, axis=2)                                  # [n,H,W-1]
    dxb = dx.reshape(n, H // 8, 8, W - 1)
    sb = sel.reshape(n, H // 8, 8, W - 1).all(axis=2)        # [n,H/8,W-1]
    T = np.abs(dxb).mean(axis=2)[sb]
    C = np.abs(dxb.mean(axis=2))[sb]
    return dict(coherence=float(C.sum() / T.sum()), coherentE=float(C.mean()), totalE=float(T.mean()),
                nblocks=int(sb.sum()))


for clip, lst in specs.items():
    geo = D.geometry(clip, [p for _, p in lst])
    frames = D.frames_for(geo)
    gts = D.gt_stack(clip, geo, frames)
    reg = R.regions(gts)
    _, near = D.decompose_input(clip, reg, frames, "none")
    lo = reg["lo"][:, :, :-1]
    nr = near[:, :, :-1] | near[:, :, 1:]
    sel = lo & ~nr
    rows = OrderedDict()
    rows["GT"] = coh(gts, sel)
    ys = {}
    for mode in ("none", "rowlin"):
        yy = []
        for f in frames:
            m, wp = D.splat_quads(clip, f)
            import torch
            w = torch.from_numpy(wp.astype(np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
            mm = torch.from_numpy(m)[None, None].contiguous()
            D.CF.process(w, mm, 0, 0, R.TH, R.TW, mode, "keep", maxw=3, margin=0)
            yy.append(w[0].permute(1, 2, 0).numpy().mean(-1))
        rows[f"INPUT ({mode})"] = coh(np.stack(yy).astype(np.float32), sel)
    tl = []
    for f in frames:   # the left eye at the render's own left half (passthrough), same pixel grid
        tl.append(R.gray(R.grab(lst[0][1], f)[:, :R.TW]))
    rows["left eye (render passthrough half)"] = coh(np.stack(tl), sel)
    for lab, p in lst:
        rows[lab] = coh(D.render_stack(p, frames), sel)
    tee(f"\n=== {clip}  n={len(frames)} frames, GT-flat & >2px from holes, 8-row vertical blocks ===")
    tee(f"{'label':40s} {'coherence':>10s} {'coherentE':>10s} {'totalE':>9s} {'coh/GT':>8s} {'tot/GT':>8s} {'nblocks':>9s}")
    g = rows["GT"]
    for lab, d in rows.items():
        tee(f"{lab[:40]:40s} {d['coherence']:10.4f} {d['coherentE']:10.6f} {d['totalE']:9.6f} "
            f"{d['coherentE']/g['coherentE']:8.3f} {d['totalE']/g['totalE']:8.3f} {d['nblocks']:9d}")
    out[clip] = dict(frames=frames, rows=rows)
    json.dump(out, open(OUTJ, "w"), indent=1)
tee("\ncoherence = sum|mean_8 dx| / sum mean_8|dx| over 8-row vertical blocks (iid noise ~0.35, vertical stripe -> 1)")
fh.close()
print("DONE")
