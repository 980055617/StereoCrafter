#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- 2-panel visual check (PREREG g8): stock decoder | fine-tuned decoder, same latents.

Clips 0301, 0170, 0052 (pre-registered), frame 76.  Crop 288x512 px at full resolution, placed on the 3x4 block grid
(192x256 blocks) at the block of highest REG_FRAME-registered-GT gradient energy (model-independent choice), expanded to
288x512 around its centre.  One PNG per clip: rows = origin, deliverable; columns = stock | fine-tuned | registered GT.
Also a 2x zoom of the central 144x256 of the same crop.
usage: python make_panels_v1.py <run> <sel>
"""
import json
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
RUN, SEL = sys.argv[1], sys.argv[2]
T = "outputs/deep_20261004/decoder_ft/test_renders"
O = f"outputs/deep_20261004/decoder_ft/score_reg_test_{RUN}"
P = f"outputs/deep_20261004/decoder_ft/panels_test_{RUN}"
os.makedirs(P, exist_ok=True)
TH, TW, CH, CW, F = 576, 1024, 288, 512, 76


def right(path, f):
    a = VideoReader(path, ctx=cpu(0))[f].asnumpy()
    return a[:, a.shape[2] // 2:]


for clip in ("0301", "0170", "0052"):
    R = json.load(open(f"{O}/{clip}.json"))
    t0, l0 = R["window"]
    H, W = R["quadrant"]
    dy, dx = R["reg"]["smooth_ddy"][F], R["reg"]["smooth_ddx"][F]
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
    cy, cx = bb[0] * 192 + 96, bb[1] * 256 + 128
    y0, x0 = int(np.clip(cy - CH // 2, 0, TH - CH)), int(np.clip(cx - CW // 2, 0, TW - CW))
    rows = []
    for lat, name in (("origin_cap", "origin"), ("deliv_cap", "deliverable")):
        a = right(f"{T}/{clip}_{lat}__stock/{clip}_inpainting_results_sbs.mkv", F)[y0:y0 + CH, x0:x0 + CW]
        b = right(f"{T}/{clip}_{lat}__{SEL}/{clip}_inpainting_results_sbs.mkv", F)[y0:y0 + CH, x0:x0 + CW]
        gg = gt[y0:y0 + CH, x0:x0 + CW]
        row = np.concatenate([a, np.full((CH, 4, 3), 255, np.uint8), b, np.full((CH, 4, 3), 255, np.uint8), gg], 1)
        for k, s in enumerate((f"{name} stock decoder", f"{name} + decoder {SEL}", "real right eye (REG_FRAME)")):
            cv2.putText(row, s, (k * (CW + 4) + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        rows.append(row)
    pan = np.concatenate([rows[0], np.full((4, rows[0].shape[1], 3), 255, np.uint8), rows[1]], 0)
    cv2.imwrite(f"{P}/{clip}_f{F:03d}_stock_vs_{SEL}_crop.png", cv2.cvtColor(pan, cv2.COLOR_RGB2BGR))
    zy, zx = CH // 4, CW // 4
    z = pan[:, :]  # 2x zoom of the central quarter of each panel
    zrows = []
    for r in rows:
        tiles = [r[zy:zy + CH // 2, k * (CW + 4) + zx:k * (CW + 4) + zx + CW // 2] for k in range(3)]
        tiles = [cv2.resize(t_, (CW, CH), interpolation=cv2.INTER_NEAREST) for t_ in tiles]
        zrows.append(np.concatenate([tiles[0], np.full((CH, 4, 3), 255, np.uint8), tiles[1],
                                     np.full((CH, 4, 3), 255, np.uint8), tiles[2]], 1))
    zp = np.concatenate([zrows[0], np.full((4, zrows[0].shape[1], 3), 255, np.uint8), zrows[1]], 0)
    cv2.imwrite(f"{P}/{clip}_f{F:03d}_stock_vs_{SEL}_zoom2x.png", cv2.cvtColor(zp, cv2.COLOR_RGB2BGR))
    print(f"{clip}: crop y{y0} x{x0} (block {bb}) shift ({dy},{dx}) -> {P}", flush=True)
print("PANELS_DONE")
