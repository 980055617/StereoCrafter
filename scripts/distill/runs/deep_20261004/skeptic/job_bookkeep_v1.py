#!/usr/bin/env python
"""EXPLORATORY (PREREG_ADDENDUM_5): how much of origin's LPIPS deficit against BR-geometry rows (O_C, BR_raw, COMP_*)
is registration bookkeeping?  REG_FRAME aligns the GT to BR, so rows that sit exactly on BR's geometry are favoured;
origin's output deviates from BR's geometry by ~0.5-2 px (M7) and LPIPS is shift-sensitive (K1).
S10 frames (as M3), target = REG_FRAME GT.  BRt = BR with holes Telea-filled (= COMP_telea).
  X   = BRt warped (bicubic) by DIS(origin -> BRt): BR's content placed at ORIGIN's geometry.
  X0  = BRt warped by DIS(Bb -> BRt), Bb = BRt Gaussian-blurred to origin's b3 amplitude ratio (M3 rho_origin_vs_input):
        same appearance gap, zero true displacement -> interpolation + flow-noise control.
  bookkeeping = LPIPS(X) - LPIPS(X0)   (cost of origin's geometry alone, on BR content).
  Also: Og = origin warped by DIS(BRt -> origin) = origin's content at BR's geometry (an optimistic geometry fix).
usage: python job_bookkeep_v1.py <out_json> <clip> [<clip> ...]"""
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeplib as S  # noqa: E402

torch.set_num_threads(int(os.environ.get("SK_THREADS", "3")))
cv2.setNumThreads(int(os.environ.get("SK_THREADS", "3")))
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
TH, TW = S.TH, S.TW
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
lp = S.Lp("alex")
F3 = 0.5 * (1 / 8 + 1 / 4)
res = {}


def gflow(a, b):
    return dis.calc(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None)


for CLIP in sys.argv[2:]:
    js, _ = S.regjson(CLIP)
    m3 = json.load(open(f"outputs/deep_20261004/skeptic/m3_v2/{CLIP}.json"))
    frames = m3["frames"]
    ratio = min(max(m3["m1"]["rho_origin_vs_input"][2], 0.05), 0.999)
    sig = math.sqrt(-math.log(ratio) / (2 * math.pi ** 2 * F3 ** 2))
    D = S.load_clip(CLIP, frames, want_splat=True)
    holes = D["BLext"][:, S.MY:S.MY + TH, S.MX:S.MX + TW].astype(np.float32).mean(-1) > 127.5
    BR = np.ascontiguousarray(D["BRext"][:, S.MY:S.MY + TH, S.MX:S.MX + TW])
    GT = np.stack([S.box_crop(D["TR"][j], *S.reg_shift(js, fi, "REG_FRAME")) for j, fi in enumerate(frames)])
    _, O = S.render_right(S.row_path(CLIP, "origin"), frames)
    BRt = np.stack([S.inpaint_holes(BR[j], holes[j]) for j in range(len(frames))])
    X, X0, Og, st = [], [], [], []
    for j in range(len(frames)):
        y = S.luma(BRt[j])
        tex = (np.abs(np.diff(y, axis=1, append=y[:, -1:])) + np.abs(np.diff(y, axis=0, append=y[-1:]))) > 0.02
        m = tex & ~holes[j]
        m[:16] = m[-16:] = False; m[:, :32] = m[:, -32:] = False
        f_ob = gflow(O[j], BRt[j])
        Bb = S.q8(S.gauss(BRt[j].astype(np.float32) / 255., sig))
        f_0 = gflow(Bb, BRt[j])
        f_bo = gflow(BRt[j], O[j])
        X.append(S.q8(S.warp(BRt[j], f_ob)))
        X0.append(S.q8(S.warp(BRt[j], f_0)))
        Og.append(S.q8(S.warp(O[j], f_bo)))
        st.append(dict(mean_abs_flow_origin=float(np.linalg.norm(f_ob, axis=-1)[m].mean()),
                       mean_abs_flow_ctrl=float(np.linalg.norm(f_0, axis=-1)[m].mean())))
    X, X0, Og = np.stack(X), np.stack(X0), np.stack(Og)
    e = {k: float(np.mean(lp(v, GT))) for k, v in (("origin", O), ("BRt", BRt), ("X_BR_at_origin_geom", X),
                                                    ("X0_ctrl", X0), ("Og_origin_at_BR_geom", Og))}
    e["bookkeeping"] = e["X_BR_at_origin_geom"] - e["X0_ctrl"]
    e["sigma_ctrl"] = sig
    e["flow_stats"] = dict(origin=float(np.mean([s["mean_abs_flow_origin"] for s in st])),
                           ctrl=float(np.mean([s["mean_abs_flow_ctrl"] for s in st])))
    res[CLIP] = e
    print(CLIP, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
