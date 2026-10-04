#!/usr/bin/env python
"""DIAGNOSTIC (reported only): does reviewlib's stripeE excess survive a FINER GT registration?

decomp_v1 registers the real right eye to the render grid with ONE integer shift per clip (build_review.py).
Here every 64x64 block of every frame gets its own extra shift (ddy in [-2,2], ddx in [-12,12]) around that
global shift, chosen to maximise PSNR against the model's warped INPUT (hole pixels excluded) -- the same
target build_review.py used, independent of every render config.  The block-registered GT mosaic then defines
the flat (gradient <= median) and edge (top decile) regions exactly as reviewlib.regions() does, and stripeE /
edgeHF / flatHF are recomputed for GT, the INPUT and the renders.  If the INPUT's stripeE/GT drops a lot, the
excess measured with the single global shift was misregistration (edges landing in "flat" pixels).
usage: diag_blockreg_v1.py OUT_JSON OUT_TXT clip:label=path [...]     (CUDA_VISIBLE_DEVICES="" ; CPU only)
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
B, SY, SX = 64, 2, 12
out = {}
fh = open(OUTT, "w")


def tee(*a):
    print(*a, flush=True)
    print(*a, file=fh, flush=True)


for clip, lst in specs.items():
    geo = D.geometry(clip, [p for _, p in lst])
    frames = D.frames_for(geo)
    t0, l0 = geo["t0"], geo["l0"]
    gy, gx = geo["gtShift"]
    mosaic, inputs, gains, shifts = [], [], [], []
    for f in frames:
        TLq, TR, BLq, BRq, H, W = R.tile_quadrants(clip, f)
        m, wp = D.splat_quads(clip, f)
        valid = m < 0.5
        tgt = wp.astype(np.float32)
        y0, x0 = t0 + gy - SY, l0 + gx - SX
        big = TR[max(0, y0):y0 + R.TH + 2 * SY, max(0, x0):x0 + R.TW + 2 * SX].astype(np.float32)
        assert y0 >= 0 and x0 >= 0 and big.shape[:2] == (R.TH + 2 * SY, R.TW + 2 * SX), (clip, f, y0, x0, big.shape)
        mos = np.zeros((R.TH, R.TW, 3), np.float32)
        for by in range(0, R.TH, B):
            for bx in range(0, R.TW, B):
                tb = tgt[by:by + B, bx:bx + B]
                vb = valid[by:by + B, bx:bx + B]
                if vb.sum() < 0.2 * vb.size:
                    vb = np.ones_like(vb)
                best = None
                for a in range(-SY, SY + 1):
                    for c in range(-SX, SX + 1):
                        gb = big[SY + by + a:SY + by + a + B, SX + bx + c:SX + bx + c + B]
                        e = (((gb - tb) ** 2).sum(-1) * vb).sum() / (3 * vb.sum())
                        if best is None or e < best[0]:
                            best = (e, a, c)
                e0 = ((((big[SY + by:SY + by + B, SX + bx:SX + bx + B] - tb) ** 2).sum(-1) * vb).sum() / (3 * vb.sum()))
                _, a, c = best
                mos[by:by + B, bx:bx + B] = big[SY + by + a:SY + by + a + B, SX + bx + c:SX + bx + c + B]
                gains.append(10 * np.log10(max(e0, 1e-6) / max(best[0], 1e-6)))
                shifts.append((a, c))
        mosaic.append(mos.mean(-1) / 255.0)
        inputs.append(wp.astype(np.float32).mean(-1) / 255.0)
    G = np.stack(mosaic).astype(np.float32)
    reg = R.regions(G)
    gd = R.decompose(reg, G)
    rows = OrderedDict(GT_blockreg=gd)
    rows["INPUT (none)"] = R.decompose(reg, np.stack(inputs))
    # the same quantities with the single global shift, for the side-by-side
    G1 = D.gt_stack(clip, geo, frames)
    reg1 = R.regions(G1)
    g1 = R.decompose(reg1, G1)
    rows1 = OrderedDict(GT_global=g1)
    rows1["INPUT (none)"] = R.decompose(reg1, np.stack(inputs))
    for lab, p in lst:
        y = D.render_stack(p, frames)
        rows[lab] = R.decompose(reg, y)
        rows1[lab] = R.decompose(reg1, y)
    sh = np.array(shifts)
    tee(f"\n=== {clip}  n={len(frames)} frames, {B}x{B} blocks, extra shift ddy in [-{SY},{SY}] ddx in [-{SX},{SX}] around "
        f"global {geo['gtShift']}; mean PSNR gain vs global {np.mean(gains):.2f} dB; |ddx|>0 on "
        f"{100*np.mean(sh[:,1]!=0):.0f}% of blocks, at the search edge on {100*np.mean(np.abs(sh[:,1])==SX):.0f}% ===")
    tee(f"{'label':28s} {'stripeE/GT glob':>15s} {'stripeE/GT block':>16s} {'edgeHF/GT glob':>15s} {'edgeHF/GT block':>15s} {'flatHF/GT glob':>15s} {'flatHF/GT block':>15s}")
    for lab in rows:
        if lab.startswith("GT"):
            continue
        a, b = rows1[lab], rows[lab]
        tee(f"{lab[:28]:28s} {a['stripeE']/g1['stripeE']:15.3f} {b['stripeE']/gd['stripeE']:16.3f} "
            f"{a['edgeHF']/g1['edgeHF']:15.3f} {b['edgeHF']/gd['edgeHF']:15.3f} {a['flatHF']/g1['flatHF']:15.3f} {b['flatHF']/gd['flatHF']:15.3f}")
    out[clip] = dict(frames=frames, blockreg=rows, globalreg=rows1, mean_psnr_gain_db=float(np.mean(gains)),
                     frac_blocks_shifted=float(np.mean(sh[:, 1] != 0)), frac_at_edge=float(np.mean(np.abs(sh[:, 1]) == SX)))
    json.dump(out, open(OUTJ, "w"), indent=1, default=float)
fh.close()
print("DONE")
