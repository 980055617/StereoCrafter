#!/usr/bin/env python
"""blur_diag PREREG ADDENDUM 5: score extra GT-geometry rows (VAE_GTx_L, RS_GTx_L) of one clip.  The metric code is
copied verbatim from score_lpips_r2.py (REG, VALID_GT), score_detail_r2.py (GT chain + raw) and score_nr_r2.py (6 NR
metrics, GPU).  Rows are read from outputs/deep_20261004/blur_diag/lanczos_r2/<clip>/<clip>_<ROW>.mkv (single view).
usage: CUDA_VISIBLE_DEVICES=0 [NR env] python score_addon_r2.py <out_dir> <clip> <ROW[,ROW]>"""
import json
import os
import sys
import time

import lpips
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402
import pyiqa  # noqa: E402

OUTD, CLIP, ROWS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}.json"
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
T0 = time.time()
dev = "cuda"
SIG = (1, 2, 4, 8)
BORDER = 24
METRICS = ["musiq", "clipiqa", "niqe", "topiq_nr", "clipiqa+", "arniqa"]


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


M = B.meta(CLIP)
FR = M["frames"]
GT = B.load_row(CLIP, "GT", FR)
hole, dil = B.holes(CLIP, FR)
valid = ~dil
n, H, W = valid.shape
inner = np.zeros((H, W), bool)
inner[BORDER:H - BORDER, BORDER:W - BORDER] = True
net = lpips.LPIPS(net="alex").to(dev).eval()
net_sp = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


@torch.no_grad()
def lp_scalar(x, ref):
    R, T = t01(x), t01(ref)
    out = []
    for i in range(0, len(R), 4):
        o = net((R[i:i + 4].cuda() * 2 - 1), (T[i:i + 4].cuda() * 2 - 1))
        out += [float(v) for v in o.view(-1)]
    return out


@torch.no_grad()
def lp_valid(x, ref):
    xc = B.composite(x, ref, dil)
    R, T = t01(xc), t01(ref)
    V = torch.from_numpy(valid).float()
    out = []
    for i in range(0, len(R), 4):
        m = net_sp((R[i:i + 4].cuda() * 2 - 1), (T[i:i + 4].cuda() * 2 - 1))[:, 0]
        v = V[i:i + 4].cuda()
        out += [float(a) for a in ((m * v).sum(dim=(1, 2)) / v.sum(dim=(1, 2)))]
    return out


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
mets = {m: pyiqa.create_metric(m, device=torch.device(dev)) for m in METRICS}
res = {}
for row in ROWS:
    p = f"{B.OUT}/lanczos_r2/{CLIP}/{CLIP}_{row}.mkv"
    x = B.read_frames(p, FR)
    assert x.shape == GT.shape, (row, x.shape)
    lp = dict(REG=lp_scalar(x, GT), VALID_GT=lp_valid(x, GT))
    det = dict(raw=raw_stats(x), gtchain=stats(x, GT, S_GT))
    per = {m: [] for m in METRICS}
    with torch.no_grad():
        for f in range(len(x)):
            t = torch.from_numpy(x[f]).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(dev)
            for m in METRICS:
                per[m].append(float(mets[m](t)))
    res[row] = dict(path=p, lpips=dict(perframe=lp, clip={k: float(np.mean(v)) for k, v in lp.items()}), detail=det,
                    nr=dict(perframe=per, clip={m: float(np.mean(v)) for m, v in per.items()}))
    log(f"{row:9s} REG {np.mean(lp['REG']):.4f} VALID_GT {np.mean(lp['VALID_GT']):.4f} | edgeHF {det['gtchain']['edgeHF']:.5f}"
        f" b1 {det['gtchain']['b1']:.5f} | " + "  ".join(f"{m} {np.mean(v):.4f}" for m, v in per.items()))
json.dump(dict(clip=CLIP, frames=FR, rows=res, seconds=time.time() - T0), open(OJ, "w"))
log(f"wrote {OJ}")
print("CLIP_DONE", CLIP, flush=True)
