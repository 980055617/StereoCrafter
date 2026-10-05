#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- DEV diagnostic panels (after the dev selection; not a selection input).

Dev clips 0245 (largest dREG_FRAME increase), 0082 (only consistent decrease), 0268 (iPhone), frame 76.  Crop 288x512 at full
resolution at the 3x4-grid block of highest REG_FRAME-registered-GT gradient energy (model-independent).  Rows = origin,
deliverable; columns = stock decoder | s1000 | s2000 | registered real right eye.  Plus a 2x zoom of the crop centre.
usage: python make_panels_dev_v1.py
"""
import json
import os

import cv2
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
R = "/mnt/ssd_data/deep_20261004/decoder_ft/redec_dev"
O = "outputs/deep_20261004/decoder_ft/score_reg_dev_main_v1"
P = "outputs/deep_20261004/decoder_ft/panels_dev_main_v1"
os.makedirs(P, exist_ok=True)
TH, TW, CH, CW, F = 576, 1024, 288, 512, 76
COLS = ["stock", "s1000", "s2000"]


def right(path, f):
    a = VideoReader(path, ctx=cpu(0))[f].asnumpy()
    return a[:, a.shape[2] // 2:]


def sep(h):
    return np.full((h, 4, 3), 255, np.uint8)


for clip in ("0245", "0082", "0268"):
    J = json.load(open(f"{O}/{clip}.json"))
    t0, l0 = J["window"]
    H, W = J["quadrant"]
    dy, dx = J["reg"]["smooth_ddy"][F], J["reg"]["smooth_ddx"][F]
    tile = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))[F].asnumpy()
    gt = tile[t0 + dy:t0 + dy + TH, W + l0 + dx:W + l0 + dx + TW]
    g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = np.abs(np.diff(g, axis=1))[:-1] + np.abs(np.diff(g, axis=0))[:, :-1]
    best, bb = -1, (0, 0)
    for by in range(3):
        for bx in range(4):
            s = float(e[by * 192:(by + 1) * 192, bx * 256:(bx + 1) * 256].mean())
            if s > best:
                best, bb = s, (by, bx)
    y0 = int(np.clip(bb[0] * 192 + 96 - CH // 2, 0, TH - CH))
    x0 = int(np.clip(bb[1] * 256 + 128 - CW // 2, 0, TW - CW))
    rows, zrows = [], []
    for lat, name in (("origin_cap", "origin"), ("deliv_cap", "deliverable")):
        tiles = [right(f"{R}/{clip}_{lat}__{c}/{clip}_inpainting_results_sbs.mkv", F)[y0:y0 + CH, x0:x0 + CW] for c in COLS]
        tiles.append(gt[y0:y0 + CH, x0:x0 + CW])
        labs = [f"{name} {c}" for c in COLS] + ["real right eye (REG_FRAME)"]
        tl = []
        for t_, s_ in zip(tiles, labs):
            t_ = np.ascontiguousarray(t_.copy())
            cv2.putText(t_, s_, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            tl.append(t_)
        rows.append(np.concatenate(sum([[t_, sep(CH)] for t_ in tl], [])[:-1], 1))
        zt = [cv2.resize(np.ascontiguousarray(t_[CH // 4:CH // 4 + CH // 2, CW // 4:CW // 4 + CW // 2]), (CW, CH),
                         interpolation=cv2.INTER_NEAREST) for t_ in tiles]
        zrows.append(np.concatenate(sum([[t_, sep(CH)] for t_ in zt], [])[:-1], 1))
    for nm, rr in (("crop", rows), ("zoom2x", zrows)):
        pan = np.concatenate([rr[0], np.full((4, rr[0].shape[1], 3), 255, np.uint8), rr[1]], 0)
        cv2.imwrite(f"{P}/{clip}_f{F:03d}_stock_s1000_s2000_{nm}.png", cv2.cvtColor(pan, cv2.COLOR_RGB2BGR))
    print(f"{clip}: crop y{y0} x{x0} shift ({dy},{dx})", flush=True)
print("DEV_PANELS_DONE ->", P)
