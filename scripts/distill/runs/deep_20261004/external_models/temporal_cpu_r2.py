#!/usr/bin/env python
"""Temporal check of the M2SVid renders, CPU only (no GPU lock needed).  Descriptive, not a PREREG_r2 gate.

Same metric definitions as finalcheck_20261004/validate/score_temporal_ll.py (tLP = mean LPIPS-alex(R_t, R_t+1) over all
frames; seam ratio = mean |R_t+1 - R_t| at seam transitions / elsewhere), but each method is measured at ITS OWN window
seams: origin / deliverable (14-frame windows, overlap 3) at t = 11k+2 (k>=1), M2SVid (16-frame non-overlapping
windows) at t = 16k-1 (k>=1) plus the anchored last window's first new frame.  GT = real right eye at the UNREG window.
RAFT warp error is NOT computed (needs the GPU).  usage: python temporal_cpu_r2.py <out_dir> <tag> <clip>...
"""
import json
import os
import sys
import time

import lpips
import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
TH, TW = 576, 1024
OUT, TAG = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(16)
net = lpips.LPIPS(net="alex").eval()
pub = json.load(open("scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json"))["cells"]


def m2_seams(T, win=16):
    starts = list(range(0, T - win + 1, win))
    s = [st for st in starts[1:]]
    end = starts[-1] + win
    if end < T:
        s.append(end)                       # first NEW frame of the anchored last window
    return [x - 1 for x in s]               # transition t -> t+1 with t+1 = window start


def orig_seams(T):
    return [11 * k + 2 for k in range(1, T) if 11 * k + 2 < T - 1]


for clip in sys.argv[3:]:
    oj = os.path.join(OUT, f"{clip}.json")
    if os.path.exists(oj):
        print("exists, skip", oj)
        continue
    t0 = time.time()
    paths = {"origin": f"outputs/beyond4_lossless/clips/{clip}_origin_ll/{clip}_inpainting_results_sbs.mkv",
             "deliv": f"outputs/beyond_distil_mamba_scaled/clips/{clip}_mstudent2_step800_deliv_ll/"
                      f"{clip}_inpainting_results_sbs.mkv",
             "m2svid": f"outputs/deep_20261004/external_models/clips/{clip}_{TAG}_ll/{clip}_inpainting_results_sbs.mkv"}
    V = {}
    for k, p in paths.items():
        vr = VideoReader(p, ctx=cpu(0))
        V[k] = vr.get_batch(list(range(len(vr)))).asnumpy()[:, :, TW:]
    vt = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))
    f0 = vt[0].asnumpy()
    H, W = f0.shape[0] // 2, f0.shape[1] // 2
    o = pub[clip]["origin_ll"]
    r0, c0 = (H - TH) // 2 + o["dy"], (W - TW) // 2 + o["dx"]
    V["GT"] = np.stack([vt[i].asnumpy()[r0:r0 + TH, W + c0:W + c0 + TW] for i in range(len(vt))])
    n = min(len(v) for v in V.values())
    res = {}
    for k, v in V.items():
        x = torch.from_numpy(np.ascontiguousarray(v[:n])).permute(0, 3, 1, 2).float() / 255.
        d = (x[1:] - x[:-1]).abs().mean(dim=(1, 2, 3)).numpy()
        tl = []
        with torch.no_grad():
            for i in range(0, n - 1, 8):
                a, b = x[i:min(i + 8, n - 1)], x[i + 1:min(i + 9, n)]
                tl += [float(z) for z in net(a * 2 - 1, b * 2 - 1).view(-1)]
        res[k] = dict(tLP=float(np.mean(tl)), d=d.tolist(), tlp=tl)
    sm, so = m2_seams(n), orig_seams(n)

    def ratio(d, seams):
        d = np.asarray(d)
        msk = np.zeros(len(d), bool)
        msk[[s for s in seams if s < len(d)]] = True
        return float(d[msk].mean() / d[~msk].mean())

    for k in res:
        res[k]["seam_ratio_m2grid"] = ratio(res[k]["d"], sm)
        res[k]["seam_ratio_origgrid"] = ratio(res[k]["d"], so)
    out = dict(clip=clip, n=n, m2_seams=sm, orig_seams=so, gt_window=[r0, c0], rows=res, seconds=time.time() - t0)
    json.dump(out, open(oj, "w"))
    print(f"TEMPORAL {clip} n={n} " + "  ".join(
        f"{k}: tLP {res[k]['tLP']:.4f} seam(own) "
        f"{res[k]['seam_ratio_m2grid'] if k == 'm2svid' else res[k]['seam_ratio_origgrid']:.3f}"
        f" [m2grid {res[k]['seam_ratio_m2grid']:.3f} origgrid {res[k]['seam_ratio_origgrid']:.3f}]" for k in res)
        + f"  ({time.time() - t0:.0f}s)", flush=True)
