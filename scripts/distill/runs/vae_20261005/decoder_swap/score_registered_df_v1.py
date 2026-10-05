#!/usr/bin/env python
"""scale_gt COPY of scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py (md5 recorded in TOOL_MD5_v1.txt).
ONLY change: the inspection panel no longer hard-codes the labels "origin_ll" / "mstudent2_step800_deliv_ll" (absent on dev clips,
which made the original raise KeyError before writing the json); it shows LABELS[0] and LABELS[-1].  Scoring is untouched.
Original header follows.
REGISTERED-GT LPIPS scorer -- more_20261004 / eval_robustness lane.  Definitions: PREREG.txt (same dir).

One clip per process.  The UNREG column is scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py verbatim
(same frames, same left-eye window search, same LPIPS-alex calls in batches of 4, same accumulation), so it must
reproduce the published ROW lpips (gate G0).  The registered columns move ONLY the GT crop; the shift is estimated
from model-independent data (real right eye TR vs the model's own input BR from the splatting video, holes
excluded) and is identical for every config, so it cannot favour any of them.

usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock \
         python score_registered_v1.py <out_dir> <published_rows.json> <clip>
writes <out_dir>/<clip>.json, <clip>_grid_splatBR.npy (PSNR grid, all GT frames), <clip>_inspect.png
Nothing tracked is modified; renders are only read.
"""
import hashlib
import json
import math
import os
import sys
import time

import cv2
import lpips
import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
STEP = int(os.environ.get("SCORE_STEP", "4"))
TH, TW = 576, 1024
DDY = list(range(-10, 11))            # PREREG search grid (exhaustive)
DDX = list(range(-120, 41))
BY, BX, BH, BW = 3, 4, 192, 256       # exploratory block grid
BLK_DY, BLK_DX = 3, 16
FRAMES_CHUNK, OVERLAP = 14, 3         # config/0160_overfit_inference_matched.json
XC = 23                               # shifts per GPU chunk in the grid search
dev = "cuda"

OUT, PUB, CLIP = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
out_json = os.path.join(OUT, f"{CLIP}.json")
assert not os.path.exists(out_json), f"refusing to overwrite {out_json}"
pub = json.load(open(PUB))
LABELS = pub["labels"]
cells = pub["cells"][CLIP]
T0 = time.time()


def log(*a):
    print(f"[{CLIP} {time.time() - T0:7.1f}s]", *a, flush=True)


def decode_all(vr, idxs, fn, chunk=16):
    for s in range(0, len(idxs), chunk):
        part = idxs[s:s + chunk]
        b = vr.get_batch(part).asnumpy()
        for k, fi in enumerate(part):
            fn(fi, b[k])


# ------------------------------------------------------------------------------------------------ geometry
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
vs = VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0))
n_t, n_s = len(vt), len(vs)
f0 = vt[0].asnumpy()
H, W = f0.shape[0] // 2, f0.shape[1] // 2
s0 = vs[0].asnumpy()
Hs, Ws = s0.shape[0] // 2, s0.shape[1] // 2
# deployed window = crop to /128 then centre (utils/inpainting.py), for the splatting tile (the model's input)
st0, sl0 = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
n_g = min(n_t, n_s)                                   # frames that have both GT and model input
samp_t = list(range(0, n_t, STEP))                    # score_clip_ll.py gt sampling
log(f"train {n_t} frames quadrant {H}x{W}; splat {n_s} frames quadrant {Hs}x{Ws} window ({st0},{sl0}); n_g={n_g}")

# ------------------------------------------------------------------------------------------------ left-eye search (score_clip_ll.py verbatim)
# region of TL needed by the search: centre +- (60+6) px
cy0, cx0 = (H - TH) // 2, (W - TW) // 2
SR = 66
assert cy0 - SR >= 0 and cx0 - SR >= 0 and cy0 + TH + SR <= H and cx0 + TW + SR <= W
search_k = list(range(0, len(samp_t), 6))             # gtL[:n:6] -> every 6th SAMPLED frame
TLs = {}
decode_all(vt, [samp_t[k] for k in search_k],
           lambda fi, f: TLs.__setitem__(fi, f[cy0 - SR:cy0 + TH + SR, cx0 - SR:cx0 + TW + SR].copy()))


def left_search(Lsub, kk):
    """Lsub: uint8 [m, TH, TW, 3] = L[::6] of the render; kk: matching sampled-frame indices."""
    Lg = torch.from_numpy(Lsub).to(dev).float() / 255.
    G = torch.from_numpy(np.stack([TLs[samp_t[k]] for k in kk])).to(dev).float() / 255.
    best = ((0, 0), -1)
    for st, rng in ((4, 60), (1, 6)):
        cy, cx = best[0]
        for dy in range(cy - rng, cy + rng + 1, st):
            for dx in range(cx - rng, cx + rng + 1, st):
                t0 = (H - TH) // 2 + dy
                l0 = (W - TW) // 2 + dx
                if t0 < 0 or l0 < 0 or t0 + TH > H or l0 + TW > W:
                    continue
                y, x = t0 - (cy0 - SR), l0 - (cx0 - SR)
                if y < 0 or x < 0 or y + TH > G.shape[1] or x + TW > G.shape[2]:
                    raise RuntimeError("left search left the decoded region")
                m = (Lg - G[:, y:y + TH, x:x + TW]).pow(2).mean().item()
                ps = 10 * math.log10(1 / max(m, 1e-12))
                if ps > best[1]:
                    best = ((dy, dx), ps)
    return best


# ------------------------------------------------------------------------------------------------ decode GT + model input
# the scorer window must equal the deployed window of the splatting tile (checked per config below as well)
pdy, pdx = cells[LABELS[0]]["dy"], cells[LABELS[0]]["dx"]
t0, l0 = (H - TH) // 2 + pdy, (W - TW) // 2 + pdx
assert (t0, l0) == (st0, sl0), f"published scorer window {(t0, l0)} != splatting deployed window {(st0, sl0)}"
assert (H, W) == (Hs, Ws), "train / splatting quadrant sizes differ"
ry0, rx0 = -DDY[0], -DDX[0]                          # offset of the zero shift inside the TR region
R_T, R_L = t0 + DDY[0], l0 + DDX[0]
R_B, R_R = t0 + TH + DDY[-1], l0 + TW + DDX[-1]
assert R_T >= 0 and R_L >= 0 and R_B <= H and R_R <= W, "search region leaves the quadrant"

TRreg = np.empty((n_g, R_B - R_T, R_R - R_L, 3), np.uint8)    # real right eye, search region, ALL frames
TLwin = {}                                                    # real left eye at the window, scored frames
BRt, HOLEt = {}, {}                                           # train-tile BR / BL (anchor check), scored frames
samp_set = set(samp_t)


def _train(fi, f):
    if fi < n_g:
        TRreg[fi] = f[R_T:R_B, W + R_L:W + R_R]
    if fi in samp_set:
        TLwin[fi] = f[t0:t0 + TH, l0:l0 + TW].copy()
        BRt[fi] = f[H + t0:H + t0 + TH, W + l0:W + l0 + TW].copy()
        HOLEt[fi] = f[H + t0:H + t0 + TH, l0:l0 + TW].astype(np.float32).mean(-1) > 127.5


decode_all(vt, list(range(n_t)), _train)
BRs = np.empty((n_g, TH, TW, 3), np.uint8)                    # model input BR at the window, ALL frames
HOLEs = np.empty((n_g, TH, TW), bool)


def _splat(fi, f):
    if fi < n_g:
        BRs[fi] = f[Hs + st0:Hs + st0 + TH, Ws + sl0:Ws + sl0 + TW]
        HOLEs[fi] = f[Hs + st0:Hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) > 127.5


decode_all(vs, list(range(n_g)), _splat)
log("decoded GT + model input")

# ------------------------------------------------------------------------------------------------ registration
P255 = 20 * math.log10(255.0)


def psnr_from_sse(sse, nval):
    return P255 - 10 * math.log10(max(sse / nval, 1e-12))


def sse_grid(Treg, B, v, ys, xs, oy, ox, h=TH, w=TW, by=0, bx=0):
    """SSE over valid pixels of Treg[oy+ddy+by : +h, ox+ddx+bx : +w] - B, for every (ddy, ddx) in ys x xs."""
    out = torch.empty(len(ys), len(xs), dtype=torch.float64, device=dev)
    for iy, ddy in enumerate(ys):
        rows = Treg[oy + ddy + by:oy + ddy + by + h]
        for j0 in range(0, len(xs), XC):
            js = xs[j0:j0 + XC]
            S = torch.stack([rows[:, ox + ddx + bx:ox + ddx + bx + w] for ddx in js])
            S.sub_(B).pow_(2).mul_(v)
            out[iy, j0:j0 + len(js)] = S.sum(dim=(1, 2, 3)).double()
    return out


def frame_grid(Treg_u8, B_u8, hole):
    T = torch.from_numpy(Treg_u8).to(dev).float()
    B = torch.from_numpy(B_u8).to(dev).float()
    v = torch.from_numpy(~hole).to(dev).float().unsqueeze(-1)
    nval = float(v.sum().item()) * 3.0
    sse = sse_grid(T, B, v, DDY, DDX, ry0, rx0)
    return (P255 - 10 * torch.log10(torch.clamp(sse / nval, min=1e-12))).cpu().numpy(), nval


GRID = np.empty((n_g, len(DDY), len(DDX)), np.float32)
for fi in range(n_g):
    GRID[fi], _ = frame_grid(TRreg[fi], BRs[fi], HOLEs[fi])
np.save(os.path.join(OUT, f"{CLIP}_grid_splatBR.npy"), GRID)
iy0, ix0 = DDY.index(0), DDX.index(0)
raw = [np.unravel_index(int(np.argmax(GRID[fi])), GRID[fi].shape) for fi in range(n_g)]
raw_ddy = np.array([DDY[a] for a, b in raw])
raw_ddx = np.array([DDX[b] for a, b in raw])
raw_psnr = np.array([float(GRID[fi][a, b]) for fi, (a, b) in enumerate(raw)])
zero_psnr = GRID[:, iy0, ix0].astype(np.float64)


def med5(a):
    ap = np.pad(a, (2, 2), mode="edge")
    return np.array([int(np.median(ap[i:i + 5])) for i in range(len(a))])


sm_ddy, sm_ddx = med5(raw_ddy), med5(raw_ddx)
sm_psnr = np.array([float(GRID[fi][DDY.index(sm_ddy[fi]), DDX.index(sm_ddx[fi])]) for fi in range(n_g)])
meanG = GRID.astype(np.float64).mean(0)
ci, cj = np.unravel_index(int(np.argmax(meanG)), meanG.shape)
c_ddy, c_ddx = DDY[ci], DDX[cj]
bnd = lambda dy, dx: abs(dy) == 10 or dx in (DDX[0], DDX[-1])
boundary_raw = [int(fi) for fi in range(n_g) if bnd(raw_ddy[fi], raw_ddx[fi])]
log(f"REG splat-BR: clip optimum ({c_ddy:+d},{c_ddx:+d}) meanPSNR {meanG[ci, cj]:.3f} (zero {meanG[iy0, ix0]:.3f}); "
    f"raw ddx [{raw_ddx.min()},{raw_ddx.max()}] ddy [{raw_ddy.min()},{raw_ddy.max()}]; smoothed ddx "
    f"[{sm_ddx.min()},{sm_ddx.max()}]; boundary raw frames {len(boundary_raw)} clip-boundary {bnd(c_ddy, c_ddx)}")

# anchor check: same estimator against the TRAIN-tile BR + train BL mask (regB / review target), scored frames
anch = {}
for fi in samp_t:
    if fi >= n_g:
        continue
    g, _ = frame_grid(TRreg[fi], BRt[fi], HOLEt[fi])
    a, b = np.unravel_index(int(np.argmax(g)), g.shape)
    d = BRt[fi].astype(np.float64) - BRs[fi].astype(np.float64)
    vv = (~HOLEs[fi])[..., None]
    mse_tb = float((d * d * vv).sum() / (vv.sum() * 3))
    anch[fi] = dict(ddy=DDY[a], ddx=DDX[b], psnr=float(g[a, b]), psnr0=float(g[iy0, ix0]),
                    trainBR_vs_splatBR_psnr=P255 - 10 * math.log10(max(mse_tb, 1e-12)),
                    hole_disagree=float((HOLEt[fi] != HOLEs[fi]).mean()))
log("anchor (train-BR) raw shifts at scored frames computed")

# ------------------------------------------------------------------------------------------------ sampler windows
win_k, win_pos = {}, {}
gen_len = 0
k = 0
for i in range(0, n_s, FRAMES_CHUNK - OVERLAP):
    if i + OVERLAP >= n_s:
        break
    if gen_len > 0 and i + FRAMES_CHUNK > n_s:
        cur_i = max(n_s + OVERLAP - FRAMES_CHUNK, 0)
        cur_ov = i - cur_i + OVERLAP
    else:
        cur_i, cur_ov = i, OVERLAP
    nwin = min(FRAMES_CHUNK, n_s - cur_i)
    keep = range(nwin) if i == 0 else range(cur_ov, nwin)
    for p in keep:
        win_k[gen_len], win_pos[gen_len] = k, p
        assert cur_i + p == gen_len, (cur_i, p, gen_len)
        gen_len += 1
    k += 1
assert gen_len == n_s, (gen_len, n_s)

# ------------------------------------------------------------------------------------------------ LPIPS
net = lpips.LPIPS(net="alex").to(dev).eval()


def gt_crop(fi, ddy, ddx, by=0, bx=0, h=TH, w=TW):
    return TRreg[fi, ry0 + ddy + by:ry0 + ddy + by + h, rx0 + ddx + bx:rx0 + ddx + bx + w]


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


# block-local shifts at the scored frames (model-independent, computed once)
blk = {}
for fi in samp_t:
    if fi >= n_g:
        continue
    fy, fx = int(sm_ddy[fi]), int(sm_ddx[fi])
    T = torch.from_numpy(TRreg[fi]).to(dev).float()
    sh = []
    for by in range(BY):
        for bx in range(BX):
            ys_, xs_ = slice(by * BH, (by + 1) * BH), slice(bx * BW, (bx + 1) * BW)
            valid = ~HOLEs[fi][ys_, xs_]
            if valid.mean() < 0.5:
                sh.append((fy, fx, 0))
                continue
            yl = [d for d in range(fy - BLK_DY, fy + BLK_DY + 1) if DDY[0] <= d <= DDY[-1]]
            xl = [d for d in range(fx - BLK_DX, fx + BLK_DX + 1) if DDX[0] <= d <= DDX[-1]]
            B = torch.from_numpy(np.ascontiguousarray(BRs[fi][ys_, xs_])).to(dev).float()
            v = torch.from_numpy(valid).to(dev).float().unsqueeze(-1)
            sse = sse_grid(T, B, v, yl, xl, ry0, rx0, h=BH, w=BW, by=by * BH, bx=bx * BW).cpu().numpy()
            a, b = np.unravel_index(int(np.argmin(sse)), sse.shape)
            sh.append((yl[a], xl[b], 1))
    blk[fi] = sh

VARIANTS = ["UNREG", "REG_CLIP", "REG_FRAME", "REG_FRAME_RAW"]
res_cfg = {}
for lab in LABELS:
    path = cells[lab]["path"]
    vr = VideoReader(path, ctx=cpu(0))
    idx = list(range(0, len(vr), STEP))
    v = vr.get_batch(idx).asnumpy()
    half = v.shape[2] // 2
    n = min(len(idx), len(samp_t))
    Lr, Rr = v[:n, :, :half], v[:n, :, half:]
    assert Lr.shape[1:3] == (TH, TW), Lr.shape
    md5_left = hashlib.md5(np.ascontiguousarray(Lr).tobytes()).hexdigest()
    ks = list(range(0, n, 6))
    (dy, dx), al = left_search(Lr[::6], ks)
    frames = [samp_t[j] for j in range(n)]
    assert all(fi < n_g for fi in frames)
    shifts = {"UNREG": [(0, 0)] * n, "REG_CLIP": [(c_ddy, c_ddx)] * n,
              "REG_FRAME": [(int(sm_ddy[fi]), int(sm_ddx[fi])) for fi in frames],
              "REG_FRAME_RAW": [(int(raw_ddy[fi]), int(raw_ddx[fi])) for fi in frames]}
    lp = {k_: [] for k_ in VARIANTS + ["BLK_FRAME", "BLK_LOCAL"]}
    tot = {k_: 0.0 for k_ in VARIANTS}
    mse = {k_: [] for k_ in VARIANTS}
    with torch.no_grad():
        for var in VARIANTS:
            gt = np.stack([gt_crop(fi, *shifts[var][j]) for j, fi in enumerate(frames)])
            R = t01(Rr)
            t = t01(gt)
            for i in range(0, n, 4):                      # score_clip_ll.py batching, verbatim
                o = net((R[i:i + 4].cuda() * 2 - 1), (t[i:i + 4].cuda() * 2 - 1))
                tot[var] += float(o.sum())
                lp[var] += [float(x) for x in o.view(-1)]
            mse[var] = [float(x) for x in (R - t).pow(2).mean(dim=(1, 2, 3))]
        # exploratory block metric
        for var in ("BLK_FRAME", "BLK_LOCAL"):
            for j0 in range(0, n, 4):
                pr, pg = [], []
                for j in range(j0, min(j0 + 4, n)):
                    fi = frames[j]
                    for b_, (by, bx) in enumerate([(a, b) for a in range(BY) for b in range(BX)]):
                        if var == "BLK_FRAME":
                            sy, sx = shifts["REG_FRAME"][j]
                        else:
                            sy, sx = blk[fi][b_][0], blk[fi][b_][1]
                        pr.append(Rr[j, by * BH:(by + 1) * BH, bx * BW:(bx + 1) * BW])
                        pg.append(gt_crop(fi, sy, sx, by * BH, bx * BW, BH, BW))
                o = net(t01(np.stack(pr)).cuda() * 2 - 1, t01(np.stack(pg)).cuda() * 2 - 1).view(-1, BY * BX)
                lp[var] += [float(x) for x in o.mean(dim=1)]
    leftf = [10 * math.log10(1 / max(float(((Lr[j].astype(np.float64) - TLwin[frames[j]]) / 255.) .__pow__(2).mean()), 1e-12))
             for j in range(n)]
    res_cfg[lab] = dict(path=path, md5_left=md5_left, n=n, dy=dy, dx=dx, leftPSNR=al, leftPSNR_frame=leftf,
                        lpips_clip={k_: tot[k_] / n for k_ in VARIANTS} |
                                   {k_: float(np.mean(lp[k_])) for k_ in ("BLK_FRAME", "BLK_LOCAL")},
                        lpips=lp, mse=mse,
                        rPSNR={k_: 10 * math.log10(1 / max(float(np.mean(mse[k_])), 1e-12)) for k_ in VARIANTS})
    pubv = cells[lab]["lpips"]
    log(f"{lab:28s} off ({dy},{dx}) left {al:.2f}  UNREG {tot['UNREG'] / n:.6f} (pub {pubv:.6f}, d {tot['UNREG'] / n - pubv:+.1e})"
        f"  CLIP {tot['REG_CLIP'] / n:.4f}  FRAME {tot['REG_FRAME'] / n:.4f}  RAW {tot['REG_FRAME_RAW'] / n:.4f}"
        f"  BLKf {np.mean(lp['BLK_FRAME']):.4f}  BLKl {np.mean(lp['BLK_LOCAL']):.4f}")

# ------------------------------------------------------------------------------------------------ inspection panel (half res)
mid = samp_t[min(res_cfg[LABELS[0]]["n"], len(samp_t)) // 2]
j_mid = samp_t.index(mid)


def half(a):
    return cv2.resize(a, (TW // 2, TH // 2), interpolation=cv2.INTER_AREA)


def rframe(lab):
    vr = VideoReader(res_cfg[lab]["path"], ctx=cpu(0))
    return vr[mid].asnumpy()[:, TW:]


gt_un, gt_re = gt_crop(mid, 0, 0), gt_crop(mid, int(sm_ddy[mid]), int(sm_ddx[mid]))
d_un = np.abs(gt_un.astype(np.int16) - BRs[mid].astype(np.int16)).clip(0, 255).astype(np.uint8)
d_re = np.abs(gt_re.astype(np.int16) - BRs[mid].astype(np.int16)).clip(0, 255).astype(np.uint8)
hole_vis = np.zeros_like(BRs[mid]); hole_vis[HOLEs[mid]] = (255, 0, 0)
top = [BRs[mid], gt_un, gt_re, rframe(LABELS[0]), rframe(LABELS[-1])]
bot = [hole_vis, d_un, d_re, np.zeros_like(gt_un), np.zeros_like(gt_un)]
panel = np.concatenate([np.concatenate([half(a) for a in top], 1), np.concatenate([half(a) for a in bot], 1)], 0)
labels = ["model input BR", "GT unregistered", f"GT REG_FRAME ({int(sm_ddy[mid]):+d},{int(sm_ddx[mid]):+d})",
          LABELS[0][:24], LABELS[-1][:24], "holes (red)", "|GT-BR| unreg", "|GT-BR| reg", "", ""]
for q, s in enumerate(labels):
    if s:
        cv2.putText(panel, s, ((q % 5) * (TW // 2) + 6, (q // 5) * (TH // 2) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2)
cv2.putText(panel, f"{CLIP} frame {mid}", (6, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.imwrite(os.path.join(OUT, f"{CLIP}_inspect_f{mid:03d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

# ------------------------------------------------------------------------------------------------ write
scored = [samp_t[j] for j in range(res_cfg[LABELS[0]]["n"])]
out = dict(
    clip=CLIP, step=STEP, quadrant=[H, W], window=[t0, l0], n_train=n_t, n_splat=n_s, n_g=n_g,
    search=dict(ddy=[DDY[0], DDY[-1]], ddx=[DDX[0], DDX[-1]]),
    frames=scored, window_k=[win_k[fi] for fi in scored], window_pos=[win_pos[fi] for fi in scored],
    hole_frac=[float(HOLEs[fi].mean()) for fi in scored],
    reg=dict(raw_ddy=raw_ddy.tolist(), raw_ddx=raw_ddx.tolist(), raw_psnr=raw_psnr.tolist(),
             zero_psnr=zero_psnr.tolist(), smooth_ddy=sm_ddy.tolist(), smooth_ddx=sm_ddx.tolist(),
             smooth_psnr=sm_psnr.tolist(), clip_ddy=c_ddy, clip_ddx=c_ddx,
             clip_mean_psnr=float(meanG[ci, cj]), zero_mean_psnr=float(meanG[iy0, ix0]),
             smooth_mean_psnr=float(sm_psnr.mean()), raw_mean_psnr=float(raw_psnr.mean()),
             boundary_raw_frames=boundary_raw, boundary_clip=bool(bnd(c_ddy, c_ddx))),
    anchor_trainBR={str(k_): v_ for k_, v_ in anch.items()},
    blk_local={str(k_): v_ for k_, v_ in blk.items()},
    configs=res_cfg, seconds=time.time() - T0,
)
json.dump(out, open(out_json, "w"))
log(f"wrote {out_json}")
print("CLIP_DONE", CLIP, flush=True)
