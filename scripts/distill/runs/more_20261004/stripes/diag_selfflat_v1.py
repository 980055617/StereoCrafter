#!/usr/bin/env python
"""DIAGNOSTIC (reported only, not a PREREG criterion): how much of reviewlib's stripeE (mean |dx| inside GT-flat
regions of the DISPARITY-REGISTERED real right eye) is texture that the image really has, and how much is
residual misregistration (an edge of the image landing inside a GT-flat region)?

For every image X (registered GT, warped INPUT none/rowlin, renders) over the same frames as decomp_v1:
  stripeE_GTflat   reviewlib stripeE: mean |dx_X| on GT-flat pixels (GT gradient <= its median)
  stripeE_selfflat mean |dx_X| on X's OWN flat pixels (X's gradient <= X's median)        -> no registration needed
  hp_selfflat      mean |d2x_X| (horizontal 2nd difference, removes ramps) on X's own flat pixels
  hpcoh_selfflat   vertical coherence of d2x over 8-row blocks entirely inside X's own flat pixels
                   (sum |mean_8 d2x| / sum mean_8 |d2x| ; iid noise ~0.35, a 1-px vertical stripe -> 1.0)
All pixels within 2 px of a hole (mask >= 0.5) are excluded everywhere (the crack fill only acts there).
usage: diag_selfflat_v1.py OUT_JSON OUT_TXT clip:label=path [...]     (CUDA_VISIBLE_DEVICES="" ; CPU only)
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

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


def gmag(y):
    g = np.zeros_like(y)
    g[:, :, :-1] += np.abs(np.diff(y, axis=2))
    g[:, :-1, :] += np.abs(np.diff(y, axis=1))
    return g


def stats(y, gt_lo, far):
    n, H, W = y.shape
    dx = np.abs(np.diff(y, axis=2))                        # [n,H,W-1]
    own_lo = gmag(y) <= np.quantile(gmag(y), 0.50)
    m_gt = gt_lo[:, :, :-1] & far[:, :, :-1] & far[:, :, 1:]
    m_own = own_lo[:, :, :-1] & far[:, :, :-1] & far[:, :, 1:]
    d2 = y[:, :, 2:] - 2 * y[:, :, 1:-1] + y[:, :, :-2]   # [n,H,W-2] centred on x=1..W-2
    m2 = own_lo[:, :, 1:-1] & far[:, :, :-2] & far[:, :, 1:-1] & far[:, :, 2:]
    d2b = d2.reshape(n, H // 8, 8, W - 2)
    mb = m2.reshape(n, H // 8, 8, W - 2).all(axis=2)
    T = np.abs(d2b).mean(axis=2)[mb]
    C = np.abs(d2b.mean(axis=2))[mb]
    return dict(stripeE_GTflat=float(dx[m_gt].mean()), stripeE_selfflat=float(dx[m_own].mean()),
                hp_selfflat=float(np.abs(d2[m2]).mean()), hpcoh_selfflat=float(C.sum() / max(T.sum(), 1e-12)),
                ownflat_frac_far=float(m_own.mean()))


for clip, lst in specs.items():
    geo = D.geometry(clip, [p for _, p in lst])
    frames = D.frames_for(geo)
    gts = D.gt_stack(clip, geo, frames)
    reg = R.regions(gts)
    _, near = D.decompose_input(clip, reg, frames, "none")
    far = ~near
    rows = OrderedDict()
    rows["GT (registered)"] = stats(gts, reg["lo"], far)
    for mode in ("none", "rowlin"):
        yy = []
        for f in frames:
            m, wp = D.splat_quads(clip, f)
            w = torch.from_numpy(wp.astype(np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
            mm = torch.from_numpy(m)[None, None].contiguous()
            D.CF.process(w, mm, 0, 0, R.TH, R.TW, mode, "keep", maxw=3, margin=0)
            yy.append(w[0].permute(1, 2, 0).numpy().mean(-1))
        rows[f"INPUT ({mode})"] = stats(np.stack(yy).astype(np.float32), reg["lo"], far)
    for lab, p in lst:
        rows[lab] = stats(D.render_stack(p, frames), reg["lo"], far)
    g = rows["GT (registered)"]
    tee(f"\n=== {clip}  n={len(frames)} frames, pixels > 2 px from any hole ===")
    tee(f"{'label':34s} {'strE_GTflat':>11s} {'/GT':>6s} {'strE_selfflat':>13s} {'/GT':>6s} {'hp_selfflat':>11s} {'/GT':>6s} {'hpcoh':>6s}")
    for lab, d in rows.items():
        tee(f"{lab[:34]:34s} {d['stripeE_GTflat']:11.6f} {d['stripeE_GTflat']/g['stripeE_GTflat']:6.2f} "
            f"{d['stripeE_selfflat']:13.6f} {d['stripeE_selfflat']/g['stripeE_selfflat']:6.2f} "
            f"{d['hp_selfflat']:11.6f} {d['hp_selfflat']/g['hp_selfflat']:6.2f} {d['hpcoh_selfflat']:6.3f}")
    out[clip] = dict(frames=frames, rows=rows)
    json.dump(out, open(OUTJ, "w"), indent=1)
tee("\nstrE_GTflat = reviewlib stripeE restricted to >2px from holes; strE_selfflat = same |dx| on the image's OWN")
tee("flattest half (no registration involved); hp = |horizontal 2nd difference|; hpcoh = its 8-row vertical coherence")
tee("(iid noise ~0.35; vertical 1-px stripes -> 1.0).  A large strE_GTflat/GT with a small strE_selfflat/GT means")
tee("the excess is edges landing in GT-flat pixels (residual misregistration), not texture inside flat areas.")
fh.close()
print("DONE")
