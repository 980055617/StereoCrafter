#!/usr/bin/env python
"""Registration-free ringing test: 1-D intensity profiles across a strong edge.

Ringing/halo has an unmistakable signature in a scanline across a step edge -- an overshoot
above the bright plateau and an undershoot below the dark plateau, within ~1-4 px of the
transition.  Genuine detail does not overshoot the plateaus.  The edge is located in the
ORIGIN render, so the five panels are pixel-registered by construction and no GT alignment
enters this test; the GT profile is drawn from the disparity-registered window for reference.

Also prints, per clip, a scalar PLATEAU-OVERSHOOT statistic averaged over many edges.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reviewlib as R  # noqa: E402

OUT = "outputs/review_20261001/profiles"
os.makedirs(OUT, exist_ok=True)
M = json.load(open("outputs/review_20261001/metrics.json"))
HALF = 16          # scanline half-width around the edge
PLAT = (6, 14)     # plateau sampled 6..14 px from the edge on each side
rep = open("outputs/review_20261001/RINGING_PROFILES.txt", "w")


def tee(*s):
    print(*s, flush=True)
    print(*s, file=rep, flush=True)


tee("""REGISTRATION-FREE RINGING TEST -- plateau overshoot across strong edges
=====================================================================
For every strong, locally monotone vertical step edge in the ORIGIN render (gradient in the
top 2%, isolated, with flat plateaus 6-14 px either side), each panel's scanline is read over
+-16 px.  overshoot = max excursion BEYOND the panel's own two plateau levels, in units of the
edge's own contrast, averaged over all such edges.  Ringing/halo shows up as a positive
overshoot that grows with sharpening; genuine detail does not push past the plateaus.
All panels are pixel-registered by construction (same window of the same render grid), so no
GT alignment enters this number.
""")

SUM = {}
for clip, info in M.items():
    f = info["frame"]
    t0, l0 = info["t0"], info["l0"]
    c = info["crops"]["texture"]
    y0, x0, S = c["y"], c["x"], c["size"]
    ddy, ddx = c["gtShift"]
    _, TR, _, _, _, _ = R.tile_quadrants(clip, f)
    gt = R.gray(TR[t0 + y0 + ddy:t0 + y0 + ddy + S, l0 + x0 + ddx:l0 + x0 + ddx + S])
    labs, imgs = ["GT (registered)"], [gt]
    for lab, tag, roots in R.PANELS:
        imgs.append(R.gray(R.grab(R.panel_path(clip, tag, roots), f)[:, R.TW:][y0:y0 + S, x0:x0 + S]))
        labs.append(lab)
    org = imgs[1]

    # find strong isolated vertical step edges in the ORIGIN panel
    gx = np.abs(np.diff(org, axis=1))
    thr = np.quantile(gx, 0.98)
    cands = []
    for yy in range(0, S, 2):
        row = org[yy]
        for xx in range(HALF + 1, S - HALF - 1):
            if gx[yy, xx] < thr:
                continue
            a = row[xx - PLAT[1]:xx - PLAT[0]]
            b = row[xx + PLAT[0] + 1:xx + PLAT[1] + 1]
            lo, hi = a.mean(), b.mean()
            contrast = abs(hi - lo)
            if contrast < 0.12:
                continue
            if a.std() > 0.02 * 1.5 + 0.01 or b.std() > 0.02 * 1.5 + 0.01:
                continue  # plateaus must be flat, else we are measuring texture not an edge
            cands.append((contrast, yy, xx, lo, hi))
    cands.sort(reverse=True)
    # de-duplicate: keep edges at least 24 px apart
    keep = []
    for cand in cands:
        if all(abs(cand[1] - k[1]) > 8 or abs(cand[2] - k[2]) > 24 for k in keep):
            keep.append(cand)
        if len(keep) >= 60:
            break
    if not keep:
        tee(f"{clip}: no isolated step edge found in the texture crop -- skipped")
        continue

    stats = {lab: [] for lab in labs}
    for contrast, yy, xx, lo, hi in keep:
        # ONE denominator for every panel -- the edge contrast measured on the ORIGIN render --
        # so a panel whose own plateaus happen to converge cannot blow the ratio up.
        for lab, im in zip(labs, imgs):
            row = im[yy]
            a = row[xx - PLAT[1]:xx - PLAT[0]].mean()
            b = row[xx + PLAT[0] + 1:xx + PLAT[1] + 1].mean()
            p_lo, p_hi = min(a, b), max(a, b)
            seg = row[xx - PLAT[0] + 1:xx + PLAT[0]]
            over = max(seg.max() - p_hi, p_lo - seg.min(), 0.0)
            stats[lab].append(over / contrast)
    SUM[clip] = {lab: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                           p90=float(np.percentile(v, 90))) for lab, v in stats.items()}
    tee(f"{clip}  n_edges={len(keep)}  plateau overshoot / origin edge contrast"
        f"   (mean | median | p90):")
    for lab in labs:
        d = SUM[clip][lab]
        tee(f"    {lab:24s} {d['mean']:7.4f} | {d['median']:7.4f} | {d['p90']:7.4f}")

    # figure: the single highest-contrast edge
    contrast, yy, xx, lo, hi = keep[0]
    xs = np.arange(-HALF, HALF + 1)
    plt.figure(figsize=(7.2, 4.2))
    styles = {"GT (registered)": dict(c="k", lw=2.4, ls="-"),
              "origin (deployed)": dict(c="#4878d0", lw=1.6),
              "shipped 5-slot Mamba": dict(c="#6acc64", lw=1.6),
              "THIS DELIVERABLE": dict(c="#d65f5f", lw=2.2),
              "origin+s25 (ceiling)": dict(c="#956cb4", lw=1.6, ls="--")}
    for lab, im in zip(labs, imgs):
        plt.plot(xs, im[yy, xx - HALF:xx + HALF + 1], label=lab, **styles[lab])
    for lv in (min(lo, hi), max(lo, hi)):
        plt.axhline(lv, color="#999", lw=0.7, ls=":")
    plt.axvline(0, color="#bbb", lw=0.7)
    plt.title(f"{clip} f{f}  scanline across the strongest step edge (y={yy}, x={xx})", fontsize=10)
    plt.xlabel("pixels from the edge"); plt.ylabel("intensity")
    plt.legend(fontsize=7.5, loc="best"); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(f"{OUT}/{clip}_f{f}_edge_profile.png", dpi=125)
    plt.close()

tee("\nNOTE: the GT row is NOT a ringing measure -- the plateaus are chosen for flatness in the\nORIGIN render, and real GT grain/texture inside them registers as overshoot.  It is an upper\nreference for how much excursion the real right eye itself carries there.\n\nINTERPRETATION: if THIS DELIVERABLE's overshoot is at or below origin+s25's, its extra")
tee("sharpness is not ringing; if it is materially above origin's AND above s25's, it is.")
json.dump(SUM, open("outputs/review_20261001/ringing_profiles.json", "w"), indent=1)
rep.close()
