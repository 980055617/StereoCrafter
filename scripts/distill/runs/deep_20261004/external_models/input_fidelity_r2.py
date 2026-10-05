#!/usr/bin/env python
"""Diagnostic (post hoc, NOT pre-registered, CPU): how faithful is each render to its OWN warped input (splatting BR)
outside the holes?  Registration-free (BR and the renders share the render geometry by construction).
Metric: LPIPS-alex spatial map (lpips spatial=True) averaged over valid pixels (StereoCrafter soft mask < 0.5, dilated 7 px
excluded), and mean |render - BR| on the same pixels; frames 0,8,..  Also the same against the registered GT
(REG_FRAME shift) for comparison.  usage: python input_fidelity_r2.py <out_json> <clip>...
"""
import json
import os
import sys

import cv2
import lpips
import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
torch.set_num_threads(8)
TH, TW = 576, 1024
OUTJ = sys.argv[1]
assert not os.path.exists(OUTJ)
net = lpips.LPIPS(net="alex", spatial=True).eval()
res = {}
for c in sys.argv[2:]:
    j = json.load(open(f"outputs/more_20261004/eval_robustness/score_v1/{c}.json"))
    t0, l0 = j["window"]
    frames = list(range(0, 148, 8))
    vs = VideoReader(f"video_data/splatting/{c}_splatting_results.mp4", ctx=cpu(0))
    s0 = vs[0].asnumpy()
    Hs, Ws = s0.shape[0] // 2, s0.shape[1] // 2
    st, sl = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
    vt = VideoReader(f"video_data/train/{c}_train.mp4", ctx=cpu(0))
    paths = {"origin": f"outputs/beyond4_lossless/clips/{c}_origin_ll/{c}_inpainting_results_sbs.mkv",
             "deliv": f"outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_deliv_ll/{c}_inpainting_results_sbs.mkv",
             "m2svid": f"outputs/deep_20261004/external_models/clips/{c}_m2svid_fa_w16_ll/{c}_inpainting_results_sbs.mkv"}
    V = {k: VideoReader(p, ctx=cpu(0)).get_batch(frames).asnumpy()[:, :, TW:] for k, p in paths.items()}
    acc = {k: dict(lp_in=[], l1_in=[], lp_gt=[], l1_gt=[]) for k in paths}
    for q, fi in enumerate(frames):
        f = vs[fi].asnumpy()
        br = f[Hs + st:Hs + st + TH, Ws + sl:Ws + sl + TW]
        hole = f[Hs + st:Hs + st + TH, sl:sl + TW].astype(np.float32).mean(-1) > 127.5
        valid = ~(cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        g = vt[fi].asnumpy()
        H, W = g.shape[0] // 2, g.shape[1] // 2
        dy, dx = j["reg"]["smooth_ddy"][fi], j["reg"]["smooth_ddx"][fi]
        gt = g[t0 + dy:t0 + dy + TH, W + l0 + dx:W + l0 + dx + TW]
        tb = torch.from_numpy(br).permute(2, 0, 1)[None].float() / 127.5 - 1
        tg = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)[None].float() / 127.5 - 1
        vm = torch.from_numpy(valid)[None, None].float()
        for k in paths:
            tr = torch.from_numpy(np.ascontiguousarray(V[k][q])).permute(2, 0, 1)[None].float() / 127.5 - 1
            with torch.no_grad():
                mi = net(tr, tb)
                mg = net(tr, tg)
            acc[k]["lp_in"].append(float((mi * vm).sum() / vm.sum()))
            acc[k]["l1_in"].append(float(((tr - tb).abs().mean(1, keepdim=True) * vm).sum() / vm.sum() * 127.5))
            acc[k]["lp_gt"].append(float(mg.mean()))
            acc[k]["l1_gt"].append(float((tr - tg).abs().mean() * 127.5))
    res[c] = {k: {m: float(np.mean(v)) for m, v in a.items()} for k, a in acc.items()}
    print(c, "  ".join(f"{k}: LPIPS-vs-input(valid) {r['lp_in']:.4f} L1 {r['l1_in']:.2f} | LPIPS-vs-GT(reg) {r['lp_gt']:.4f}"
                       for k, r in res[c].items()), flush=True)
json.dump(res, open(OUTJ, "w"), indent=1)
