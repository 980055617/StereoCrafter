#!/usr/bin/env python
"""M6 LPIPS hackability (GT-independent post-processing of origin) + held-out metrics + M7 stereo retention.
One clip per call.  Definitions: PREREG.txt (M6, M7).  Needs outputs/deep_20261004/skeptic/trainsample_v1.json.
usage: python job_m6_v1.py <out_dir> <clip>"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeplib as S  # noqa: E402

sys.path.insert(0, os.path.join(S.REPO, "scripts/distill/runs/review_20261001"))
import reviewlib as RL  # noqa: E402

NT = int(os.environ.get("SK_THREADS", "6"))
torch.set_num_threads(NT)
cv2.setNumThreads(NT)
OUT, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
oj = os.path.join(OUT, f"{CLIP}.json")
assert not os.path.exists(oj), f"refusing to overwrite {oj}"
T0 = time.time()
TH, TW = S.TH, S.TW


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


TS = json.load(open(os.environ.get("SK_TS", "outputs/deep_20261004/skeptic/trainsample_v1.json")))
fam = S.family(CLIP)
PF = TS["per_family"][fam]
GRID = {g[0]: g[1:] for g in TS["grid"]}


def apply_P(img_u8, kind, s, a, seed=0):
    x = img_u8.astype(np.float32) / 255.
    if kind == "id":
        y = x
    elif kind == "unsharp":
        y = x + a * (x - S.gauss(x, s))
    elif kind == "blur":
        y = S.gauss(x, s)
    elif kind == "grain":
        y = x + np.random.default_rng(seed).normal(0, s, x.shape).astype(np.float32)
    return S.q8(y)


js, jpath = S.regjson(CLIP)
frames = list(js["frames"])
S6 = frames[0::7][:6]
D = S.load_clip(CLIP, frames, want_splat=True)
rows = {}
Lhalf = None
for r in ["origin", "AYS8", "deliverable", "s25"]:
    Lh, Rh = S.render_right(S.row_path(CLIP, r), frames)
    rows[r] = Rh
    Lhalf = Lh if Lhalf is None else Lhalf
log("loaded")
O = rows["origin"]
M11 = np.array(PF["P11_affine"])
variants = {}
for nm, (kind, s, a) in GRID.items():
    variants[nm] = np.stack([apply_P(O[j], kind, s, a, seed=fi) for j, fi in enumerate(frames)])
variants["P11"] = np.stack([S.apply_affine(o, M11) for o in O])
for tag, key in (("P12", "P12_choice_global"), ("P12f", "P12_choice_flow")):
    kind, s, a = GRID[PF[key]]
    variants[tag] = np.stack([apply_P(O[j], kind, s, a, seed=fi) for j, fi in enumerate(frames)])
kind, s, a = GRID[PF["P12_choice_global"]]
variants["P13"] = np.stack([apply_P(variants["P11"][j], kind, s, a, seed=fi) for j, fi in enumerate(frames)])
for r in ["AYS8", "deliverable", "s25"]:
    variants[r] = rows[r]
log("variants built:", list(variants))

gt = {v: np.stack([S.box_crop(D["TR"][j], *S.reg_shift(js, fi, v)) for j, fi in enumerate(frames)])
      for v in ("UNREG", "REG_FRAME")}
s6i = [frames.index(f) for f in S6]
TRg6 = gt["REG_FRAME"][s6i]
lpA = S.Lp("alex")
import lpips  # noqa: E402
import piq  # noqa: E402
lpV = lpips.LPIPS(net="vgg", verbose=False).eval()
dists = piq.DISTS(reduction="none")
greg = RL.gray(TRg6)
reg = RL.regions(greg)
res = dict(clip=CLIP, family=fam, frames=frames, S6=S6, regjson=jpath, P12=PF["P12_choice_global"],
           P12f=PF["P12_choice_flow"], rows={})
gt_dec = RL.decompose(reg, greg)
res["gt_decompose"] = gt_dec
res["gtSharp_REG"] = S.sharp_score(gt["REG_FRAME"])
for nm, V in variants.items():
    e = {}
    for v in ("UNREG", "REG_FRAME"):
        fl = lpA(V, gt[v])
        e[f"alex_{v}"] = float(np.mean(fl)); e[f"alex_{v}_frames"] = fl
    e["psnr_REG"] = float(np.mean([S.psnr_u8(V[j], gt["REG_FRAME"][j]) for j in range(len(frames))]))
    e["sharp"] = S.sharp_score(V)
    V6 = V[s6i]
    with torch.no_grad():
        A6, G6 = S.t01(V6), S.t01(TRg6)
        e["vgg_REG"] = float(np.mean([float(lpV(A6[k:k + 1] * 2 - 1, G6[k:k + 1] * 2 - 1)) for k in range(len(s6i))]))
        e["dists_REG"] = float(dists(A6, G6).mean())
        e["brisque"] = float(piq.brisque(A6, data_range=1.0, reduction="none").mean())
    e["decompose"] = RL.decompose(reg, RL.gray(V6))
    res["rows"][nm] = e
    log(f"{nm:12s} alexU {e['alex_UNREG']:.4f} alexR {e['alex_REG_FRAME']:.4f} psnrR {e['psnr_REG']:.2f} "
        f"vgg {e['vgg_REG']:.4f} dists {e['dists_REG']:.4f} brisque {e['brisque']:.1f} flatHF {e['decompose']['flatHF']:.4f} "
        f"stripeE {e['decompose']['stripeE']:.4f} sharp {e['sharp']:.4f}")
with torch.no_grad():
    res["gt_brisque"] = float(piq.brisque(S.t01(TRg6), data_range=1.0, reduction="none").mean())

# ---- M7 stereo retention (DIS flow output -> left half), S6
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
holes = D["BLext"][:, S.MY:S.MY + TH, S.MX:S.MX + TW].astype(np.float32).mean(-1) > 127.5
BR = D["BRext"][:, S.MY:S.MY + TH, S.MX:S.MX + TW]
m7 = {}
dBR, msk = [], []
for k, j in enumerate(s6i):
    br = S.inpaint_holes(np.ascontiguousarray(BR[j]), holes[j])
    gL = cv2.cvtColor(Lhalf[j], cv2.COLOR_RGB2GRAY)
    f = dis.calc(cv2.cvtColor(br, cv2.COLOR_RGB2GRAY), gL, None)
    dBR.append(f[..., 0])
    y = S.luma(Lhalf[j])
    tex = (np.abs(np.diff(y, axis=1, append=y[:, -1:])) + np.abs(np.diff(y, axis=0, append=y[-1:]))) > 0.02
    m = tex & ~holes[j]
    m[:24] = m[-24:] = False; m[:, :48] = m[:, -48:] = False
    msk.append(m)
for nm in ["origin", "AYS8", "deliverable", "s25", "P4", "P8"]:
    V = variants[nm]
    xs, ys = [], []
    for k, j in enumerate(s6i):
        f = dis.calc(cv2.cvtColor(V[j], cv2.COLOR_RGB2GRAY), cv2.cvtColor(Lhalf[j], cv2.COLOR_RGB2GRAY), None)
        xs.append(dBR[k][msk[k]]); ys.append(f[..., 0][msk[k]])
    x = np.concatenate(xs); y = np.concatenate(ys)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / max((xc * xc).sum(), 1e-9))
    m7[nm] = dict(slope=slope, mean_abs_diff=float(np.abs(y - x).mean()), sd_dBR=float(x.std()), n=int(x.size))
    log(f"M7 {nm:12s} slope {slope:.3f} mean|d_O-d_BR| {m7[nm]['mean_abs_diff']:.3f}px (sd d_BR {x.std():.2f}px)")
res["m7"] = m7
res["seconds"] = time.time() - T0
json.dump(res, open(oj, "w"))
log("wrote", oj)
print("CLIP_DONE", CLIP, flush=True)
