#!/usr/bin/env python
"""blur_diag PREREG ADDENDUM 7 -- selection-symmetric edge measures (CPU only).  Same rows, frames, composites and valid
set as score_detail_r2.py (M3).
  edgeHF_self  mean |4-nb Laplacian| at the image's OWN top-decile gradient pixels (gmag = |dx|+|dy|, quantile over valid
               interior pixels of the stack)
  edgeHF_s1    mean |Laplacian| at the top decile of the gradient of the Gaussian(sigma=1)-smoothed REFERENCE
               (noise-suppressed edge set); GT chain (ref GT, composite with GT) and BR chain (ref BR, composite with BR)
Also the add-on rows (VAE_GTx_L, RS_GTx_L) when present under lanczos_r2/.
usage: python detail_selfedge_r2.py <out_dir> <clip> [...]"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402

OUTD = sys.argv[1]
os.makedirs(OUTD, exist_ok=True)


def gray(u8):
    return u8.astype(np.float32).mean(-1) / 255.0


def gmag(g):
    m = np.zeros_like(g)
    m[:, :, :-1] += np.abs(np.diff(g, axis=2))
    m[:, :-1, :] += np.abs(np.diff(g, axis=1))
    return m


def lap_abs(g):
    return np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])


for clip in sys.argv[2:]:
    T0 = time.time()
    OJ = f"{OUTD}/{clip}.json"
    if os.path.exists(OJ):
        print(f"{clip}: exists -> skip")
        continue
    M = B.meta(clip)
    FR = M["frames"]
    paths = B.row_paths(clip)
    GT = B.load_row(clip, "GT", FR)
    BR = B.load_row(clip, "BR", FR)
    hole, dil = B.holes(clip, FR)
    valid = ~dil
    vi = valid[:, 1:-1, 1:-1]

    def s1_set(ref):
        g = np.stack([gaussian_filter(f, 1.0, mode="reflect", truncate=4.0) for f in gray(ref)])
        m = gmag(g)[:, 1:-1, 1:-1]
        return (m >= np.quantile(m[vi], 0.90)) & vi

    E1 = {"GT": s1_set(GT), "BR": s1_set(BR)}
    rows = ["GT", "VAE_GT", "VAE_GT32", "VAE_GTx", "RS_GTx", "RS_GTx_L", "BR", "VAE_BR", "ORIGIN", "DELIV", "S25", "T5NAT",
            "T5PAD", "HIRES_A", "HIRES_B", "HIRES_B_L"]
    extra = {r: f"{B.OUT}/lanczos_r2/{clip}/{clip}_{r}.mkv" for r in ("VAE_GTx_L", "RS_GTx_L")}
    res = {}
    for r in rows + ["VAE_GTx_L"]:
        if r in ("GT", "BR"):
            x = GT if r == "GT" else BR
        elif r in paths:
            x = B.load_row(clip, r, FR, paths)
        elif r in extra and os.path.exists(extra[r]):
            x = B.read_frames(extra[r], FR)
        else:
            continue
        d = {}
        for chain, ref in (("gtchain", GT), ("brchain", BR)):
            if chain == "brchain" and r in B.GT_GEOMETRY | {"VAE_GTx_L"}:
                continue
            g = gray(B.composite(x, ref, dil))
            L = lap_abs(g)
            m = gmag(g)[:, 1:-1, 1:-1]
            Es = (m >= np.quantile(m[vi], 0.90)) & vi
            d[chain] = dict(edgeHF_self=float(L[Es].mean()), edgeHF_s1=float(L[E1["GT" if chain == "gtchain" else "BR"]].mean()))
        res[r] = d
    json.dump(dict(clip=clip, frames=FR, rows=res, seconds=time.time() - T0), open(OJ, "w"))
    print(f"{clip}: {len(res)} rows, {time.time() - T0:.0f}s -> {OJ}", flush=True)
