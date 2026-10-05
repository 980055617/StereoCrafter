#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- INSPECTION AID for step 6 (not a pre-registered panel): the central 192x192 of each
pre-registered 384x384 crop (crops.json), every row, enlarged 2x with NEAREST neighbour (no smoothing), written to
<out_dir>/<clip>_f<F>_<E|F>_zoom2x.png.  CPU.
usage: python zoom_panels_v1.py <panel_dir> <out_dir> <redec_root> <latlabel>
"""
import json, os, sys
import cv2
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
PD, OUTD, RED, LAT = sys.argv[1:5]
os.makedirs(OUTD, exist_ok=True)
C = json.load(open(f"{PD}/crops.json"))
FR = C["frame"]
ROWS = ["stock", "stock32", "ftmse", "ftema", "cd"]
for clip, cc in C["crops"].items():
    R = json.load(open(f"outputs/vae_20261005/decoder_swap/score_reg_dev_v1/{clip}.json"))
    t0, l0 = R["window"]; H, W = R["quadrant"]; ddy, ddx = cc["reg_shift"]
    tile = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))[FR].asnumpy()
    GT = tile[t0 + ddy:t0 + ddy + 576, W + l0 + ddx:W + l0 + ddx + 1024]
    imgs = {"GT reg": GT}
    for d in ROWS:
        a = VideoReader(f"{RED}/{clip}_{LAT}__{d}/{clip}_inpainting_results_sbs.mkv", ctx=cpu(0))[FR].asnumpy()
        imgs[d] = a[:, a.shape[1] // 2:]
    for nm in ("E", "F"):
        y, x = cc[nm]["yx"]; y, x = y + 96, x + 96
        tiles = [cv2.resize(np.ascontiguousarray(im[y:y + 192, x:x + 192]), (384, 384), interpolation=cv2.INTER_NEAREST) for im in imgs.values()]
        sep = np.full((384, 4, 3), 255, np.uint8)
        body = tiles[0]
        for t in tiles[1:]:
            body = np.concatenate([body, sep, t], 1)
        strip = np.zeros((28, body.shape[1], 3), np.uint8)
        for i, t in enumerate(imgs):
            cv2.putText(strip, f"{t} 2x", (i * 388 + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{OUTD}/{clip}_f{FR:03d}_{nm}_zoom2x.png", cv2.cvtColor(np.concatenate([strip, body], 0), cv2.COLOR_RGB2BGR))
        print(clip, nm, "zoom at", (y, x))
