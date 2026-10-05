#!/usr/bin/env python
"""final_judge F7 visual check (CPU): crops at one scored frame, every gain-candidate row vs origin, deliverable and the
REG_FRAME-registered real right eye.  Two 320x320 crops per clip: the window with the most disocclusion holes (cracks /
smears) and the window with the most GT gradient energy (detail / texture).  1:1 pixels, no resampling.
usage: python make_visual_v1.py <score_dir> <out_dir> <frame> <clip> [<clip> ...]"""
import json, os, sys
import cv2
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
SD, OD, FRM = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(OD, exist_ok=True)
R = json.load(open("scripts/distill/runs/deep_20261004/final_judge/ROWS_v1.json"))
TH, TW, CS = 576, 1024, 320
SHOW = ["origin_ll", "mstudent2_step800_deliv_ll", "s25_ll", "AYS8_origin_g101", "origin_g100_sd7", "origin_g100_sd1fill",
        "deliv_g100_sd1fill", "m2svid_fa_w16_ll", "INPUT_fill", "HIRES_B_L", "selMain250_s8"]
for clip in sys.argv[4:]:
    S = json.load(open(f"{SD}/{clip}.json"))
    t0, l0 = S["window"]; H, W = S["quadrant"]
    assert FRM in S["frames"], (FRM, S["frames"][:5])
    dy, dx = S["reg"]["smooth_ddy"][FRM], S["reg"]["smooth_ddx"][FRM]
    tile = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))[FRM].asnumpy()
    gt = tile[t0 + dy:t0 + dy + TH, W + l0 + dx:W + l0 + dx + TW]
    sp = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))[FRM].asnumpy()
    hs, ws = sp.shape[0] // 2, sp.shape[1] // 2
    st, sl = (hs // 128 * 128 - TH) // 2, (ws // 128 * 128 - TW) // 2
    br = sp[hs + st:hs + st + TH, ws + sl:ws + sl + TW].copy()
    hole = sp[hs + st:hs + st + TH, sl:sl + TW].astype(np.float32).mean(-1) > 127.5
    inp = br.copy(); inp[hole] = (255, 0, 0)
    tiles = [("INPUT (holes red)", inp), (f"REAL RIGHT EYE reg({dy:+d},{dx:+d})", gt)]
    cells = R["cells"][clip]
    for lab in SHOW:
        if lab in cells:
            v = VideoReader(cells[lab]["path"], ctx=cpu(0))[FRM].asnumpy()
            tiles.append((lab, v[:, TW:]))
    g = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
    best = {"holes": (-1, 0, 0), "texture": (-1, 0, 0)}
    for y in range(0, TH - CS + 1, 32):
        for x in range(0, TW - CS + 1, 32):
            hf = float(hole[y:y + CS, x:x + CS].mean()); te = float(gm[y:y + CS - 1, x:x + CS - 1].mean())
            if hf > best["holes"][0]: best["holes"] = (hf, y, x)
            if te > best["texture"][0]: best["texture"] = (te, y, x)
    for name, (val, y, x) in best.items():
        crops = []
        for lab, im in tiles:
            c = np.ascontiguousarray(im[y:y + CS, x:x + CS]).copy()
            cv2.rectangle(c, (0, 0), (CS - 1, 18), (0, 0, 0), -1)
            cv2.putText(c, lab[:34], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
            crops.append(c)
        while len(crops) % 4:
            crops.append(np.zeros((CS, CS, 3), np.uint8))
        rows = [np.concatenate(crops[i:i + 4], 1) for i in range(0, len(crops), 4)]
        pan = np.concatenate(rows, 0)
        fn = f"{OD}/{clip}_f{FRM:03d}_{name}_y{y}x{x}.png"
        assert not os.path.exists(fn), fn
        cv2.imwrite(fn, cv2.cvtColor(pan, cv2.COLOR_RGB2BGR))
        print(clip, name, f"score {val:.4f}", fn, flush=True)
