#!/usr/bin/env python
"""blur_diag (deep_20261004) -- detail statistics for every row of one clip (CPU only).  Definitions: PREREG.txt M3.

Chains (both on diagnostic composites: dilated-hole pixels of the row replaced by the reference):
  GT chain   reference GT (registered real right eye); edge set E / flat set F from GT
  BR chain   reference BR (the model input, registration-free); E / F from BR; render-geometry rows only
Statistics (gray = RGB mean / 255, pooled over the 38 scored frames, valid = not dilated-hole):
  edgeHF  mean |4-nb Laplacian| on E (interior)   edgeGy  mean |vertical fwd diff| on E
  flatHF  mean |Laplacian| on F (interior)        stripeE mean |horizontal fwd diff| on F
  b1..b4  DoG band RMS (sigma 1,2,4,8; scipy gaussian_filter, mode reflect, truncate 4) over valid pixels >= 24 px
          from the border
  bandsFF the same bands on the RAW row (no composite, no mask; border 24 px) -- includes LEFT
  sharp   score_clip_ll.py statistic on the raw row (mean |horizontal diff| over RGB in [0,1])
usage: python score_detail_r2.py <out_dir> <clip>
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402

OUTD, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}.json"
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
T0 = time.time()
SIG = (1, 2, 4, 8)
BORDER = 24


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


M = B.meta(CLIP)
FR = M["frames"]
paths = B.row_paths(CLIP)
GT = B.load_row(CLIP, "GT", FR)
BR = B.load_row(CLIP, "BR", FR)
hole, dil = B.holes(CLIP, FR)
valid = ~dil
n, H, W = valid.shape
inner = np.zeros((H, W), bool)
inner[BORDER:H - BORDER, BORDER:W - BORDER] = True


def gray(u8):
    return u8.astype(np.float32).mean(-1) / 255.0


def lap_abs(g):
    return np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])


def sets(ref_u8):
    g = gray(ref_u8)
    gm = np.zeros_like(g)
    gm[:, :, :-1] += np.abs(np.diff(g, axis=2))
    gm[:, :-1, :] += np.abs(np.diff(g, axis=1))
    q90, q50 = np.quantile(gm[valid], 0.90), np.quantile(gm[valid], 0.50)
    E = (gm >= q90) & valid
    Fl = (gm <= q50) & valid
    return dict(E=E, F=Fl, E_i=E[:, 1:-1, 1:-1], F_i=Fl[:, 1:-1, 1:-1], E_y=E[:, :-1, :], F_x=Fl[:, :, :-1],
                q90=float(q90), q50=float(q50), nE=int(E.sum()), nF=int(Fl.sum()))


def bands(g, mask):
    """RMS of the four DoG bands over mask (per-frame filtering, pooled)."""
    acc = np.zeros(4)
    cnt = 0
    for f in range(len(g)):
        lv = [g[f]] + [gaussian_filter(g[f], s, mode="reflect", truncate=4.0) for s in SIG]
        m = mask[f]
        for k in range(4):
            b = lv[k] - lv[k + 1]
            acc[k] += float((b[m].astype(np.float64) ** 2).sum())
        cnt += int(m.sum())
    return [float(np.sqrt(a / cnt)) for a in acc]


def stats(x_u8, ref_u8, S):
    xc = B.composite(x_u8, ref_u8, dil)
    g = gray(xc)
    L = lap_abs(g)
    dy = np.abs(np.diff(g, axis=1))
    dx = np.abs(np.diff(g, axis=2))
    bm = valid & inner[None]
    b = bands(g, bm)
    return dict(edgeHF=float(L[S["E_i"]].mean()), edgeGy=float(dy[S["E_y"]].mean()), flatHF=float(L[S["F_i"]].mean()),
                stripeE=float(dx[S["F_x"]].mean()), b1=b[0], b2=b[1], b3=b[2], b4=b[3])


def raw_stats(x_u8):
    g = gray(x_u8)
    bf = bands(g, np.broadcast_to(inner[None], g.shape))
    xf = x_u8.astype(np.float32) / 255.0
    sharp = float(np.abs(xf[:, :, 1:] - xf[:, :, :-1]).mean())
    return dict(ff_b1=bf[0], ff_b2=bf[1], ff_b3=bf[2], ff_b4=bf[3], sharp=sharp)


S_GT = sets(GT)
S_BR = sets(BR)
log(f"GT sets q90 {S_GT['q90']:.4f} nE {S_GT['nE']}; BR sets q90 {S_BR['q90']:.4f} nE {S_BR['nE']}")
ROWS = ["GT", "LEFT"] + [r for r in ["VAE_GT", "VAE_GT32", "VAE_GTx", "RS_GTx", "RS_GTx_L", "BR", "VAE_BR", "ORIGIN", "DELIV", "S25",
                                     "T5NAT", "T5PAD", "HIRES_A", "HIRES_B", "HIRES_B_L"] if r in paths or r == "BR"]
res = {}
for row in ROWS:
    x = GT if row == "GT" else (BR if row == "BR" else B.load_row(CLIP, row, FR, paths))
    d = dict(raw=raw_stats(x))
    if row != "LEFT":
        d["gtchain"] = stats(x, GT, S_GT)
        if row not in B.GT_GEOMETRY:
            d["brchain"] = stats(x, BR, S_BR)
    res[row] = d
    log(f"{row:9s} " + (f"GT: edgeHF {d['gtchain']['edgeHF']:.5f} b1 {d['gtchain']['b1']:.5f} " if "gtchain" in d else "")
        + (f"BR: edgeHF {d['brchain']['edgeHF']:.5f} b1 {d['brchain']['b1']:.5f} " if "brchain" in d else "")
        + f"raw ff_b1 {d['raw']['ff_b1']:.5f} sharp {d['raw']['sharp']:.5f}")
json.dump(dict(clip=CLIP, frames=FR, sets_GT={k: v for k, v in S_GT.items() if not isinstance(v, np.ndarray)},
               sets_BR={k: v for k, v in S_BR.items() if not isinstance(v, np.ndarray)}, rows=res,
               seconds=time.time() - T0), open(OJ, "w"))
log(f"wrote {OJ}")
print("CLIP_DONE", CLIP, flush=True)
