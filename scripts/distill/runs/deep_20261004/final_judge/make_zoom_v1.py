#!/usr/bin/env python
"""final_judge F7 zoom (CPU): the 160x160 window with the most disocclusion holes at one frame, every listed row,
2x nearest-neighbour (no smoothing), plus the input with holes marked and the REG_FRAME-registered real right eye.
usage: python make_zoom_v1.py <score_dir> <out_dir> <frame> <clip> <row,row,...>"""
import json, os, sys
import cv2
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
SD, OD, FRM, clip, rows = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
os.makedirs(OD, exist_ok=True)
R = json.load(open("scripts/distill/runs/deep_20261004/final_judge/ROWS_v1.json"))
TH, TW, CS, Z = 576, 1024, 160, 2
S = json.load(open(f"{SD}/{clip}.json"))
t0, l0 = S["window"]; H, W = S["quadrant"]
dy, dx = S["reg"]["smooth_ddy"][FRM], S["reg"]["smooth_ddx"][FRM]
tile = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))[FRM].asnumpy()
gt = tile[t0 + dy:t0 + dy + TH, W + l0 + dx:W + l0 + dx + TW]
sp = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))[FRM].asnumpy()
hs, ws = sp.shape[0] // 2, sp.shape[1] // 2
st, sl = (hs // 128 * 128 - TH) // 2, (ws // 128 * 128 - TW) // 2
br = sp[hs + st:hs + st + TH, ws + sl:ws + sl + TW].copy()
hole = sp[hs + st:hs + st + TH, sl:sl + TW].astype(np.float32).mean(-1) > 127.5
best = (-1, 0, 0)
for y in range(0, TH - CS + 1, 16):
    for x in range(0, TW - CS + 1, 16):
        f = float(hole[y:y + CS, x:x + CS].mean())
        if f > best[0]:
            best = (f, y, x)
_, y, x = best
inp = br.copy(); inp[hole] = (255, 0, 0)
tiles = [("INPUT (holes red)", inp), ("INPUT raw", br), ("REAL RIGHT EYE (reg)", gt)]
for lab in rows:
    v = VideoReader(R["cells"][clip][lab]["path"], ctx=cpu(0))[FRM].asnumpy()
    tiles.append((lab, v[:, TW:]))
crops = []
for lab, im in tiles:
    c = cv2.resize(np.ascontiguousarray(im[y:y + CS, x:x + CS]), (CS * Z, CS * Z), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(c, (0, 0), (CS * Z - 1, 16), (0, 0, 0), -1)
    cv2.putText(c, lab[:36], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    crops.append(c)
while len(crops) % 4:
    crops.append(np.zeros((CS * Z, CS * Z, 3), np.uint8))
pan = np.concatenate([np.concatenate(crops[i:i + 4], 1) for i in range(0, len(crops), 4)], 0)
fn = f"{OD}/{clip}_f{FRM:03d}_holezoom_y{y}x{x}.png"
assert not os.path.exists(fn), fn
cv2.imwrite(fn, cv2.cvtColor(pan, cv2.COLOR_RGB2BGR))
print(clip, f"holefrac {best[0]:.3f}", fn)
