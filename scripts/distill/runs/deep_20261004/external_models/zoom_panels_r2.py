#!/usr/bin/env python
"""Full-resolution zoom panels for visual review (CPU).  Per clip, frame 76: the 2 most detailed 192x320 regions of the
REGISTERED real right eye (REG_FRAME shift from score_v1/<clip>.json), shown as
[warped input | GT right (registered) | origin | deliverable | M2SVid], x2 nearest-neighbour.
usage: python zoom_panels_r2.py <clip>...   -> outputs/deep_20261004/external_models/clips/<clip>_m2svid_fa_w16_ll/
"""
import json
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
TH, TW, FI = 576, 1024, 76
for c in sys.argv[1:]:
    od = f"outputs/deep_20261004/external_models/clips/{c}_m2svid_fa_w16_ll"
    j = json.load(open(f"outputs/more_20261004/eval_robustness/score_v1/{c}.json"))
    ddy, ddx = j["reg"]["smooth_ddy"][FI], j["reg"]["smooth_ddx"][FI]
    t0, l0 = j["window"]
    f = VideoReader(f"video_data/train/{c}_train.mp4", ctx=cpu(0))[FI].asnumpy()
    H, W = f.shape[0] // 2, f.shape[1] // 2
    gt = f[t0 + ddy:t0 + ddy + TH, W + l0 + ddx:W + l0 + ddx + TW]
    rd = lambda p: VideoReader(p, ctx=cpu(0))[FI].asnumpy()[:, TW:]
    o = rd(f"outputs/beyond4_lossless/clips/{c}_origin_ll/{c}_inpainting_results_sbs.mkv")
    d = rd(f"outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_deliv_ll/{c}_inpainting_results_sbs.mkv")
    m = rd(f"{od}/{c}_inpainting_results_sbs.mkv")
    s = VideoReader(f"video_data/splatting/{c}_splatting_results.mp4", ctx=cpu(0))[FI].asnumpy()
    Hs, Ws = s.shape[0] // 2, s.shape[1] // 2
    st, sl = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
    br = s[Hs + st:Hs + st + TH, Ws + sl:Ws + sl + TW]
    g = np.abs(np.diff(gt.astype(np.float32).mean(-1), axis=1))
    cand = sorted(((g[y:y + 192, x:x + 320].mean(), y, x) for y in range(0, TH - 192 + 1, 96)
                   for x in range(0, TW - 320 + 1, 160)), reverse=True)
    picks = []
    for sc, y, x in cand:
        if all(abs(y - py) >= 192 or abs(x - px) >= 320 for _, py, px in picks):
            picks.append((sc, y, x))
        if len(picks) == 2:
            break
    rows = []
    for _, y, x in picks:
        row = np.concatenate([a[y:y + 192, x:x + 320] for a in (br, gt, o, d, m)], 1)
        row = cv2.resize(row, (row.shape[1] * 2, row.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        for q, lab in enumerate(["warped input", f"GT right (REG {ddy:+d},{ddx:+d})", "origin", "deliverable",
                                 "M2SVid w16"]):
            cv2.putText(row, lab, (q * 640 + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(row, f"{c} f{FI} y{y} x{x}", (8, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        rows.append(row)
    p = f"{od}/{c}_zoom_detail_f{FI:03d}.png"
    cv2.imwrite(p, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print("wrote", p)
