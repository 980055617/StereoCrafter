#!/usr/bin/env python
"""vae_20261005 / verify_swap -- V1 (PREREG.txt): independent re-score of decoder_swap's dev rows on the UNet's own latents.

My own data path (frames, deployed window, registration with EXACT integer SSE over ALL n_g frames, edge-padded med5, GT crop),
library metric calls (lpips Alex batches of 4, pyiqa NIQE per frame, reviewlib.decompose).  Every scored render's full SBS uint8
array md5 is checked against its .md5 sidecar.  Compared with decoder_swap's per-clip JSONs (score_reg_dev_v1, score_aux_dev_v1).
If my shifts differ from decoder_swap's at any scored frame, every row is also scored with decoder_swap's shifts.
usage: CUDA_VISIBLE_DEVICES=<g> flock /tmp/claude-gpu<g>.lock python v1_rescore_v1.py <out_json> <clip> <label[,label...]>
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/review_20261001"))
import reviewlib as RL  # noqa: E402
import lpips  # noqa: E402
import pyiqa  # noqa: E402

OUT, CLIP, LABELS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
RED = "/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev"
LANE_O = f"{REPO}/outputs/vae_20261005/decoder_swap"
TH, TW = 576, 1024
DDY = np.arange(-10, 11)
DDX = np.arange(-120, 41)
STEP = 4
dev = torch.device("cuda")
T0 = time.time()


def log(*a):
    print(f"[v1 {CLIP} {time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ model input (splatting) and real right eye (train TR)
vs = VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0))
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
n_s, n_t = len(vs), len(vt)
n_g = min(n_s, n_t)
s0 = vs[0].asnumpy()
Hs, Ws = s0.shape[0] // 2, s0.shape[1] // 2
st0, sl0 = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2          # deployed window (utils/inpainting.py)
t_0 = vt[0].asnumpy()
H, W = t_0.shape[0] // 2, t_0.shape[1] // 2
assert (H, W) == (Hs, Ws), ((H, W), (Hs, Ws))
y_lo, y_hi = st0 + DDY[0], st0 + TH + DDY[-1]
x_lo, x_hi = sl0 + DDX[0], sl0 + TW + DDX[-1]
assert y_lo >= 0 and x_lo >= 0 and y_hi <= H and x_hi <= W, "search region leaves the quadrant"
log(f"n_splat {n_s} n_train {n_t} n_g {n_g} quadrant {H}x{W} window ({st0},{sl0})")

TRR = np.empty((n_g, y_hi - y_lo, x_hi - x_lo, 3), np.uint8)   # real right eye, search region
BR = np.empty((n_g, TH, TW, 3), np.uint8)
HOLE = np.empty((n_g, TH, TW), bool)
for a in range(0, n_g, 16):
    idx = list(range(a, min(a + 16, n_g)))
    tb = vt.get_batch(idx).asnumpy()
    sb = vs.get_batch(idx).asnumpy()
    for k, fi in enumerate(idx):
        TRR[fi] = tb[k, y_lo:y_hi, W + x_lo:W + x_hi]
        BR[fi] = sb[k, Hs + st0:Hs + st0 + TH, Ws + sl0:Ws + sl0 + TW]
        HOLE[fi] = sb[k, Hs + st0:Hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) > 127.5
del vs
log("decoded model input + real right eye")

# ------------------------------------------------------------------ registration, exact SSE
oy, ox = -DDY[0], -DDX[0]
P255 = 20 * math.log10(255.0)
raw = np.zeros((n_g, 2), int)
margin = np.zeros(n_g)
best_psnr = np.zeros(n_g)
for fi in range(n_g):
    T = torch.from_numpy(TRR[fi]).to(dev).float()
    B = torch.from_numpy(BR[fi]).to(dev).float()
    v = torch.from_numpy(~HOLE[fi]).to(dev).float()[..., None]
    nval = float(v.sum().item()) * 3.0
    sse = np.empty((len(DDY), len(DDX)), np.float64)
    for iy, dy in enumerate(DDY):
        rows = T[oy + dy:oy + dy + TH]
        for j0 in range(0, len(DDX), 32):
            xs = DDX[j0:j0 + 32]
            S = torch.stack([rows[:, ox + dx:ox + dx + TW] for dx in xs])
            S = ((S - B) ** 2) * v                                  # exact in float32 (integer values <= 65025)
            sse[iy, j0:j0 + len(xs)] = S.sum(dim=(1, 2, 3), dtype=torch.float64).cpu().numpy()
    ps = P255 - 10 * np.log10(np.maximum(sse / nval, 1e-12))
    flat = ps.ravel()
    a = int(np.argmax(flat))                                       # first maximum, ddy-major
    raw[fi] = (DDY[a // len(DDX)], DDX[a % len(DDX)])
    srt = np.sort(flat)
    margin[fi] = srt[-1] - srt[-2]
    best_psnr[fi] = srt[-1]
pad = np.pad(raw, ((2, 2), (0, 0)), mode="edge")
smooth = np.array([[int(np.median(pad[i:i + 5, c])) for c in range(2)] for i in range(n_g)])
log(f"registration done: raw ddx [{raw[:, 1].min()},{raw[:, 1].max()}] ddy [{raw[:, 0].min()},{raw[:, 0].max()}]")

lane_reg = json.load(open(f"{LANE_O}/score_reg_dev_v1/{CLIP}.json"))
lane_aux = json.load(open(f"{LANE_O}/score_aux_dev_v1/{CLIP}.json"))
lr = lane_reg["reg"]
lane_smooth = np.stack([np.array(lr["smooth_ddy"]), np.array(lr["smooth_ddx"])], 1)
lane_raw = np.stack([np.array(lr["raw_ddy"]), np.array(lr["raw_ddx"])], 1)
assert list(lane_reg["window"]) == [st0, sl0], (lane_reg["window"], st0, sl0)
samp_t = list(range(0, n_t, STEP))
raw_diff = [int(f) for f in range(n_g) if tuple(raw[f]) != tuple(lane_raw[f])]
sm_diff_all = [int(f) for f in range(n_g) if tuple(smooth[f]) != tuple(lane_smooth[f])]
log(f"raw shifts differ from decoder_swap at {len(raw_diff)} of {n_g} frames {raw_diff[:20]}; smoothed differ at "
    f"{len(sm_diff_all)} frames {sm_diff_all[:20]}")

# ------------------------------------------------------------------ scoring
net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
niqe = pyiqa.create_metric("niqe", device=dev)


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


def gt_stack(frames, sh):
    return np.stack([TRR[fi, oy + sh[fi][0]:oy + sh[fi][0] + TH, ox + sh[fi][1]:ox + sh[fi][1] + TW] for fi in frames])


@torch.no_grad()
def lp_alex(Rr, G):
    A, Gt = t01(Rr), t01(G)
    vals = []
    for i in range(0, len(A), 4):
        vals += [float(x) for x in net(A[i:i + 4].to(dev) * 2 - 1, Gt[i:i + 4].to(dev) * 2 - 1).view(-1)]
    return vals


@torch.no_grad()
def niqe_mean(x):
    return float(np.mean([float(niqe(t01(x[f:f + 1]).to(dev))) for f in range(len(x))]))


def rpsnr(Rr, G):
    A, Gt = t01(Rr), t01(G)
    mse = [float(x) for x in (A - Gt).pow(2).mean(dim=(1, 2, 3))]
    return 10 * math.log10(1 / max(float(np.mean(mse)), 1e-12))


res = dict(clip=CLIP, window=[st0, sl0], quadrant=[H, W], n_g=n_g, n_train=n_t, n_splat=n_s,
           reg=dict(raw=raw.tolist(), smooth=smooth.tolist(), margin_db=margin.tolist(), best_psnr=best_psnr.tolist(),
                    raw_differs_from_lane=raw_diff, smooth_differs_from_lane=sm_diff_all),
           labels={}, gt=None)
zero = {fi: (0, 0) for fi in range(n_g)}
mine_sh = {fi: tuple(int(x) for x in smooth[fi]) for fi in range(n_g)}
lane_sh = {fi: tuple(int(x) for x in lane_smooth[fi]) for fi in range(n_g)}
for lab in LABELS:
    p = f"{RED}/{CLIP}_{lab}/{CLIP}_inpainting_results_sbs.mkv"
    side = open(p + ".md5").read().split()[0]
    vr = VideoReader(p, ctx=cpu(0))
    n_r = len(vr)
    allf = vr.get_batch(list(range(n_r))).asnumpy()
    md5 = hashlib.md5(np.ascontiguousarray(allf).tobytes()).hexdigest()
    idx = list(range(0, n_r, STEP))
    n = min(len(idx), len(samp_t))
    frames = samp_t[:n]
    assert all(f < n_g for f in frames)
    Rr = np.ascontiguousarray(allf[idx[:n], :, TW:])
    del allf
    sm_diff = [f for f in frames if mine_sh[f] != lane_sh[f]]
    G_reg = gt_stack(frames, mine_sh)
    G_un = gt_stack(frames, zero)
    e = dict(path=p, md5=md5, md5_sidecar=side, md5_ok=(md5 == side), n=n, frames=frames,
             scored_frames_with_shift_diff=sm_diff)
    l_reg, l_un = lp_alex(Rr, G_reg), lp_alex(Rr, G_un)
    e["REG_FRAME"] = float(np.sum(l_reg) / n)
    e["UNREG"] = float(np.sum(l_un) / n)
    e["rPSNR_REG_FRAME"] = rpsnr(Rr, G_reg)
    e["NIQE"] = niqe_mean(Rr)
    greg = RL.gray(G_reg)
    e["decompose"] = RL.decompose(RL.regions(greg), RL.gray(Rr))
    if sm_diff:
        G_l = gt_stack(frames, lane_sh)
        e["REG_FRAME_laneShifts"] = float(np.sum(lp_alex(Rr, G_l)) / n)
        e["rPSNR_laneShifts"] = rpsnr(Rr, G_l)
        e["decompose_laneShifts"] = RL.decompose(RL.regions(RL.gray(G_l)), RL.gray(Rr))
    if res["gt"] is None:
        res["gt"] = dict(NIQE=niqe_mean(G_reg), decompose=RL.decompose(RL.regions(greg), greg), frames=frames)
    # decoder_swap's per-clip values for this label
    lc = lane_reg["configs"][lab]
    la = lane_aux["labels"][lab]
    e["lane"] = dict(REG_FRAME=lc["lpips_clip"]["REG_FRAME"], UNREG=lc["lpips_clip"]["UNREG"], rPSNR=lc["rPSNR"]["REG_FRAME"],
                     NIQE=la["niqe"], decompose=la["decompose"], frames=lane_reg["frames"])
    e["diff"] = dict(REG_FRAME=e["REG_FRAME"] - e["lane"]["REG_FRAME"], UNREG=e["UNREG"] - e["lane"]["UNREG"],
                     rPSNR=e["rPSNR_REG_FRAME"] - e["lane"]["rPSNR"], NIQE=e["NIQE"] - e["lane"]["NIQE"],
                     decompose_rel={k: (e["decompose"][k] - la["decompose"][k]) / la["decompose"][k]
                                    for k in ("flatHF", "edgeHF", "stripeE") if la["decompose"][k] != 0})
    res["labels"][lab] = e
    d = e["diff"]
    log(f"{lab:20s} md5_ok {e['md5_ok']} REG_FRAME {e['REG_FRAME']:.6f} (lane {e['lane']['REG_FRAME']:.6f} d {d['REG_FRAME']:+.1e})"
        f" UNREG {e['UNREG']:.6f} (d {d['UNREG']:+.1e}) rPSNR {e['rPSNR_REG_FRAME']:.4f} (d {d['rPSNR']:+.1e}) NIQE {e['NIQE']:.4f}"
        f" (d {d['NIQE']:+.1e}) flatHF {e['decompose']['flatHF']:.5f} stripeE {e['decompose']['stripeE']:.5f}"
        f" edgeHF {e['decompose']['edgeHF']:.5f} rel {max(abs(x) for x in d['decompose_rel'].values()):.1e}"
        + (f" | scored frames with shift diff {sm_diff} -> REG_FRAME with lane shifts {e['REG_FRAME_laneShifts']:.6f}" if sm_diff else ""))
res["seconds"] = time.time() - T0
json.dump(res, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
print("V1_DONE", CLIP, flush=True)
