#!/usr/bin/env python
"""vae_20261005 / verify_swap -- V5b (inspection aid, NOT pre-registered): the small feature near the bottom-left of 0268 crop E
(frame 76, deliverable latents), where the consistency-decoder tile showed a bright cyan spot in panels_v5/0268_E_unet.png.
4x nearest-neighbour zoom of a 64x64 window around it (GT reg | stock | ftmse / ftema | cd | origin stock), and, for every
frame 60..92, the mean |row - stock| (RGB levels) inside a 24x24 box at the spot, to see whether it is stable or flickers.
CPU.  usage: python v5b_spot_v1.py <out_dir>
"""
import json
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
OUTD = sys.argv[1]
OJ = f"{OUTD}/spot_0268.json"
assert not os.path.exists(OJ), OJ
RED = "/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev"
CLIP, FR, Y0, X0 = "0268", 76, 192, 288           # crop E of 0268
R = json.load(open(f"{REPO}/outputs/vae_20261005/decoder_swap/score_reg_dev_v1/{CLIP}.json"))
t0, l0 = R["window"]
H, W = R["quadrant"]
rows = {d: VideoReader(f"{RED}/{CLIP}_deliv_cap__{d}/{CLIP}_inpainting_results_sbs.mkv", ctx=cpu(0)) for d in
        ("stock", "ftmse", "ftema", "cd")}
rows["origin_stock"] = VideoReader(f"{RED}/{CLIP}_origin_cap__stock/{CLIP}_inpainting_results_sbs.mkv", ctx=cpu(0))
frames = list(range(60, 93))
imgs = {d: np.ascontiguousarray(v.get_batch(frames).asnumpy()[:, :, 1024:]) for d, v in rows.items()}
j76 = frames.index(FR)
# locate the spot: max of |cd - stock| (gray) inside the lower-left quarter of crop E at frame 76
d76 = np.abs(imgs["cd"][j76].astype(np.int16) - imgs["stock"][j76].astype(np.int16)).mean(-1)
sub = d76[Y0 + 192:Y0 + 384, X0:X0 + 192]
cy, cx = np.unravel_index(int(np.argmax(cv2.blur(sub.astype(np.float32), (9, 9)))), sub.shape)
cy, cx = Y0 + 192 + cy, X0 + cx
ddy, ddx = int(R["reg"]["smooth_ddy"][FR]), int(R["reg"]["smooth_ddx"][FR])
tile = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))[FR].asnumpy()
GT = np.ascontiguousarray(tile[t0 + ddy:t0 + ddy + 576, W + l0 + ddx:W + l0 + ddx + 1024])
y1, x1 = max(cy - 32, 0), max(cx - 32, 0)


def z4(a):
    c = a[y1:y1 + 64, x1:x1 + 64]
    return np.repeat(np.repeat(c, 4, 0), 4, 1)


def lab(img, t):
    s = np.zeros((24, img.shape[1], 3), np.uint8)
    cv2.putText(s, t, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return np.concatenate([s, img], 0)


tiles = [lab(z4(GT), "GT reg 4x"), lab(z4(imgs["stock"][j76]), "deliv stock 4x"), lab(z4(imgs["ftmse"][j76]), "deliv ft-mse 4x"),
         lab(z4(imgs["ftema"][j76]), "deliv ft-ema 4x"), lab(z4(imgs["cd"][j76]), "deliv consistency 4x"),
         lab(z4(imgs["origin_stock"][j76]), "origin stock 4x")]
sep = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
r1 = np.concatenate([tiles[0], sep, tiles[1], sep, tiles[2]], 1)
r2 = np.concatenate([tiles[3], sep, tiles[4], sep, tiles[5]], 1)
img = np.concatenate([r1, np.full((4, r1.shape[1], 3), 255, np.uint8), r2], 0)
cv2.imwrite(f"{OUTD}/spot_0268_f076_4x.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
box = (slice(cy - 12, cy + 12), slice(cx - 12, cx + 12))
per = {d: [float(np.abs(imgs[d][j][box].astype(np.int16) - imgs["stock"][j][box].astype(np.int16)).mean()) for j in range(len(frames))]
       for d in ("ftmse", "ftema", "cd", "origin_stock")}
bright = {d: [float(imgs[d][j][box].astype(np.float32).mean()) for j in range(len(frames))] for d in ("stock", "ftmse", "cd")}
json.dump(dict(clip=CLIP, frame=FR, spot_yx=[int(cy), int(cx)], zoom_window_yx=[int(y1), int(x1)], frames=frames,
               mean_abs_vs_stock_24box=per, mean_level_24box=bright), open(OJ, "w"), indent=1)
print("spot", cy, cx)
for d, v in per.items():
    print(f"{d:13s} |row-stock| in 24x24 box, frames 60..92: " + " ".join(f"{x:.0f}" for x in v))
for d, v in bright.items():
    print(f"{d:13s} mean level in box: " + " ".join(f"{x:.0f}" for x in v))
