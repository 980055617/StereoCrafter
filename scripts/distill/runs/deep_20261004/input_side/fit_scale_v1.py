"""input_side lane, probe 1 FIT (GPU): per-clip disparity scale s and offset o so that the re-splatted input
(disp' = s * disp_deployed + o, disp_deployed = (2d-1)*20 px, the recovered splatted depth d) aligns with the REAL
right eye.  Estimator = the registered scorer's (score_registered_v1.py): PSNR between the real right eye (train-tile
TR) and the warped input at the deployed 576x1024 window, holes excluded, maximised over integer GT shifts (ddy,ddx);
here additionally over s.  o = -ddx (a GT window shifted by ddx is the same as adding -ddx px of disparity).
Common support: per frame, pixels that are a hole (coverage < 0.5) for ANY candidate s of the stage are excluded for
all candidates, so larger s (bigger holes) cannot win by discarding hard pixels.
Objective (clip level): PSNR of the SSE summed over the fit frames, at one (ddy,ddx) shared by all frames.
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python fit_scale_v1.py <clip> <out_dir>
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from decord import VideoReader, cpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import splatlib as S  # noqa: E402

clip, OUT = sys.argv[1], sys.argv[2]
PREP = f"/mnt/ssd_data/deep_20261004/input_side/prep_v2/{clip}"
os.makedirs(OUT, exist_ok=True)
oj = os.path.join(OUT, f"fit_{clip}.json")
assert not os.path.exists(oj), f"refusing to overwrite {oj}"
dev = "cuda"
T0 = time.time()
meta = json.load(open(f"{PREP}/meta.json"))
top, lft, Hq, Wq, T = meta["top"], meta["lft"], meta["Hq"], meta["Wq"], meta["T"]
DEP = np.load(f"{PREP}/depth_rows.npy", mmap_mode="r")
LR = np.load(f"{PREP}/left_rows.npy", mmap_mode="r")
FIT_STEP = int(os.environ.get("FIT_STEP", "8"))
DY, XL, XR = 4, 240, 80                      # GT shift search: ddy in [-DY,DY], ddx in [-XL, XR]
DDY = list(range(-DY, DY + 1))
DDX = list(range(-XL, XR + 1))
XC = 24

# ---------------------------------------------------------------- GT (train tile TR), sequential decode
vt = VideoReader(f"{S.REPO}/video_data/train/{clip}_train.mp4", ctx=cpu(0))
n_t = len(vt)
n_g = min(n_t, T)
frames = list(range(0, n_g, FIT_STEP))
fs = set(frames)
TRreg = {}
for i in range(n_g):
    f = vt.next().asnumpy()
    assert f.shape[0] // 2 == Hq and f.shape[1] // 2 == Wq
    if i in fs:
        TRreg[i] = f[top - DY:top + S.TH + DY, Wq + lft - XL:Wq + lft + S.TW + XR].copy()
print(f"[{clip}] GT decoded n_t={n_t} n_g={n_g} fit frames {len(frames)} ({time.time()-T0:.0f}s)", flush=True)


def warp(fi, s, o=0.0):
    left = torch.from_numpy(np.asarray(LR[fi]).astype(np.float32) / 255.0).to(dev).permute(2, 0, 1).contiguous()
    d = torch.from_numpy(np.asarray(DEP[fi])).to(dev)
    disp = (d * 2.0 - 1.0) * S.MAX_DISP * s + o
    out, cov, _ = S.splat_rows(left, disp)
    w = torch.floor(torch.clamp(out * 255.0, 0, 255))[:, :, lft:lft + S.TW]          # == to_u8 (truncation)
    hole = (1.0 - cov.clamp(0, 1))[:, lft:lft + S.TW] > 0.5
    return w.permute(1, 2, 0).contiguous(), hole


def sse_grid(Treg, B, v):
    out = torch.empty(len(DDY), len(DDX), dtype=torch.float64, device=dev)
    for iy, ddy in enumerate(DDY):
        rows = Treg[DY + ddy:DY + ddy + S.TH]
        for j0 in range(0, len(DDX), XC):
            js = DDX[j0:j0 + XC]
            St = torch.stack([rows[:, XL + ddx:XL + ddx + S.TW] for ddx in js])
            St.sub_(B).pow_(2).mul_(v)
            out[iy, j0:j0 + len(js)] = St.sum(dim=(1, 2, 3)).double()
    return out


def stage(scales, tag):
    # pass 1: common support per frame
    sup = {}
    for fi in frames:
        u = None
        for s in scales:
            _, h = warp(fi, s)
            u = h if u is None else (u | h)
        sup[fi] = ~u
    # pass 2: SSE grids
    tot = {s: torch.zeros(len(DDY), len(DDX), dtype=torch.float64, device=dev) for s in scales}
    per = {s: [] for s in scales}
    nval = 0.0
    for fi in frames:
        Tg = torch.from_numpy(TRreg[fi]).to(dev).float()
        v = sup[fi].float().unsqueeze(-1)
        nv = float(v.sum().item()) * 3.0
        nval += nv
        for s in scales:
            B, _ = warp(fi, s)
            g = sse_grid(Tg, B, v)
            tot[s] += g
            a, b = np.unravel_index(int(torch.argmin(g).item()), g.shape)
            per[s].append(dict(frame=fi, ddy=DDY[a], ddx=DDX[b],
                               psnr=20 * math.log10(255) - 10 * math.log10(max(float(g[a, b]) / nv, 1e-12)),
                               psnr0=20 * math.log10(255) - 10 * math.log10(max(float(g[DY, XL]) / nv, 1e-12))))
        print(f"[{clip}] {tag} frame {fi} done ({time.time()-T0:.0f}s)", flush=True)
    res = {}
    for s in scales:
        g = tot[s]
        a, b = np.unravel_index(int(torch.argmin(g).item()), g.shape)
        ps = lambda x: 20 * math.log10(255) - 10 * math.log10(max(float(x) / nval, 1e-12))
        res[f"{s:.3f}"] = dict(s=s, ddy=DDY[a], ddx=DDX[b], o=-DDX[b], psnr=ps(g[a, b]), psnr_zero_shift=ps(g[DY, XL]),
                               boundary=bool(abs(DDY[a]) == DY or DDX[b] in (DDX[0], DDX[-1])),
                               per_frame=per[s])
        np.save(os.path.join(OUT, f"grid_{clip}_{tag}_s{s:.3f}.npy"), (g / nval).cpu().numpy().astype(np.float32))
    best = max(res.values(), key=lambda r: r["psnr"])
    sup_frac = float(np.mean([float(sup[fi].float().mean().item()) for fi in frames]))
    return res, best, sup_frac


S1 = [round(0.5 + 0.25 * k, 3) for k in range(19)]                 # 0.50 .. 5.00
r1, b1, sf1 = stage(S1, "coarse")
print(f"[{clip}] coarse best s={b1['s']} ddy={b1['ddy']} ddx={b1['ddx']} psnr={b1['psnr']:.3f}", flush=True)
S2 = sorted(set([round(b1["s"] + 0.05 * k, 3) for k in range(-5, 6) if b1["s"] + 0.05 * k > 0.1] + [1.0]))
r2, b2, sf2 = stage(S2, "fine")
print(f"[{clip}] fine best s={b2['s']} ddy={b2['ddy']} ddx={b2['ddx']} psnr={b2['psnr']:.3f}", flush=True)
out = dict(clip=clip, frames=frames, fit_step=FIT_STEP, search=dict(ddy=[DDY[0], DDY[-1]], ddx=[DDX[0], DDX[-1]]),
           coarse=dict(scales=S1, support_frac=sf1, results=r1, best={k: b1[k] for k in ("s", "ddy", "ddx", "o", "psnr")}),
           fine=dict(scales=S2, support_frac=sf2, results=r2, best={k: b2[k] for k in ("s", "ddy", "ddx", "o", "psnr")}),
           deployed_mapping_fine=r2.get("1.000"),
           fit=dict(s=b2["s"], o=b2["o"], ddy=b2["ddy"], psnr=b2["psnr"]),
           seconds=time.time() - T0)
for k in ("results",):
    pass
json.dump(out, open(oj, "w"), indent=1)
print(f"[{clip}] FIT s={b2['s']} o={b2['o']} (ddy {b2['ddy']}) PSNR {b2['psnr']:.3f} dB; deployed mapping s=1: "
      f"best shift ({r2['1.000']['ddy']},{r2['1.000']['ddx']}) PSNR {r2['1.000']['psnr']:.3f} dB, zero shift "
      f"{r2['1.000']['psnr_zero_shift']:.3f} dB  ({time.time()-T0:.0f}s)", flush=True)
print("FIT_DONE", clip, flush=True)
