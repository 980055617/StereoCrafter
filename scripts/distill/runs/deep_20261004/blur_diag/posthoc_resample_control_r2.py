#!/usr/bin/env python
"""blur_diag PREREG ADDENDUM 9 -- POST-HOC resampling control for HIRES_B / HIRES_B_L (CPU only, descriptive).
Rows per clip (scored frames, right eye 576x1024):
  ORIGIN       existing native render
  ORIGIN_RS    ORIGIN -> bicubic up to 1024x1792 (align_corners=False, clamp, round uint8) -> cv2.INTER_AREA down
  ORIGIN_RS_L  same up -> PIL Image.LANCZOS down
  HIRES_B, HIRES_B_L   the working-resolution renders (as scored in the main table)
All scored in this one CPU process (so CPU/GPU float differences cancel between rows): LPIPS-alex REG (registered GT),
UNREG (GT at zero shift), VALID_BR (spatial map, composite with BR, valid pixels) -- same calls as score_lpips_r2.py --
and detail ratios edgeHF@BR, b1/GT, b2/GT, flatHF/GT (score_detail_r2.py code).
usage: python posthoc_resample_control_r2.py <out_json> <clip> [...]"""
import json
import os
import sys

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402

OJ = sys.argv[1]
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
torch.set_num_threads(16)
TH, TW, UH, UW = 576, 1024, 1024, 1792
net = lpips.LPIPS(net="alex").eval()
net_sp = lpips.LPIPS(net="alex", spatial=True).eval()


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


@torch.no_grad()
def lp_scalar(x, ref):
    R, T = t01(x), t01(ref)
    return [float(v) for i in range(0, len(R), 4) for v in net(R[i:i + 4] * 2 - 1, T[i:i + 4] * 2 - 1).view(-1)]


@torch.no_grad()
def lp_valid(x, ref, dil, valid):
    R, T = t01(B.composite(x, ref, dil)), t01(ref)
    V = torch.from_numpy(valid).float()
    out = []
    for i in range(0, len(R), 4):
        m = net_sp(R[i:i + 4] * 2 - 1, T[i:i + 4] * 2 - 1)[:, 0]
        v = V[i:i + 4]
        out += [float(a) for a in (m * v).sum(dim=(1, 2)) / v.sum(dim=(1, 2))]
    return out


def resample(u8, lanczos):
    x = t01(u8)
    up = F.interpolate(x, size=(UH, UW), mode="bicubic", align_corners=False).clamp_(0, 1)
    big = (up * 255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    if lanczos:
        return np.stack([np.asarray(Image.fromarray(f).resize((TW, TH), Image.LANCZOS)) for f in big])
    return np.stack([cv2.resize(f, (TW, TH), interpolation=cv2.INTER_AREA) for f in big])


def gray(u8):
    return u8.astype(np.float32).mean(-1) / 255.0


def lap_abs(g):
    return np.abs(4 * g[:, 1:-1, 1:-1] - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:])


def sets(ref, valid):
    g = gray(ref)
    gm = np.zeros_like(g)
    gm[:, :, :-1] += np.abs(np.diff(g, axis=2))
    gm[:, :-1, :] += np.abs(np.diff(g, axis=1))
    q90, q50 = np.quantile(gm[valid], 0.90), np.quantile(gm[valid], 0.50)
    return ((gm >= q90) & valid)[:, 1:-1, 1:-1], ((gm <= q50) & valid)[:, 1:-1, 1:-1]


def band(g, mask, k):
    s = (1, 2, 4, 8)
    acc, cnt = 0.0, 0
    for f in range(len(g)):
        lv = [g[f]] + [gaussian_filter(g[f], x, mode="reflect", truncate=4.0) for x in s]
        b = lv[k] - lv[k + 1]
        acc += float((b[mask[f]].astype(np.float64) ** 2).sum())
        cnt += int(mask[f].sum())
    return float(np.sqrt(acc / cnt))


res = {}
for clip in sys.argv[2:]:
    M = B.meta(clip)
    FR = M["frames"]
    paths = B.row_paths(clip)
    GT = B.load_row(clip, "GT", FR)
    GTU = np.load(f"{B.CACHE}/{clip}/GTunreg_sc.npy")
    BR = B.load_row(clip, "BR", FR)
    _, dil = B.holes(clip, FR)
    valid = ~dil
    inner = np.zeros(valid.shape[1:], bool)
    inner[24:-24, 24:-24] = True
    bm = valid & inner[None]
    O = B.load_row(clip, "ORIGIN", FR, paths)
    rows = {"ORIGIN": O, "ORIGIN_RS": resample(O, False), "ORIGIN_RS_L": resample(O, True),
            "HIRES_B": B.load_row(clip, "HIRES_B", FR, paths), "HIRES_B_L": B.load_row(clip, "HIRES_B_L", FR, paths)}
    E_br, _ = sets(BR, valid)
    _, F_gt = sets(GT, valid)
    ref = {}
    gbr = gray(BR)
    ggt = gray(GT)
    ref["edgeHF@BR"] = float(lap_abs(gbr)[E_br].mean())
    ref["b1/GT"] = band(ggt, bm, 0)
    ref["b2/GT"] = band(ggt, bm, 1)
    ref["flatHF/GT"] = float(lap_abs(ggt)[F_gt].mean())
    res[clip] = {}
    for r, x in rows.items():
        d = dict(REG=float(np.mean(lp_scalar(x, GT))), UNREG=float(np.mean(lp_scalar(x, GTU))),
                 VALID_BR=float(np.mean(lp_valid(x, BR, dil, valid))))
        gb = gray(B.composite(x, BR, dil))
        gg = gray(B.composite(x, GT, dil))
        d["edgeHF@BR"] = float(lap_abs(gb)[E_br].mean()) / ref["edgeHF@BR"]
        d["b1/GT"] = band(gg, bm, 0) / ref["b1/GT"]
        d["b2/GT"] = band(gg, bm, 1) / ref["b2/GT"]
        d["flatHF/GT"] = float(lap_abs(gg)[F_gt].mean()) / ref["flatHF/GT"]
        res[clip][r] = d
        print(f"{clip} {r:11s} " + "  ".join(f"{k} {v:.4f}" for k, v in d.items()), flush=True)
json.dump(dict(note="POST-HOC (PREREG ADDENDUM 9), CPU LPIPS", rows=res), open(OJ, "w"), indent=1)
print("wrote", OJ)
