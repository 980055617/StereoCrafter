#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- INSPECTION AID for step 6 (not a pre-registered panel): for each pre-registered 384x384
crop (crops.json), |row - stock| x8 (gray, clipped) for stock32 / ftmse / ftema / cd, next to stock itself, at 100 %.
Shows WHERE each decoder changes the deployed output (edges vs flat areas).  CPU.
usage: python diff_panels_v1.py <panel_dir> <out_dir> <redec_root> <latlabel>
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
    imgs = {}
    for d in ROWS:
        a = VideoReader(f"{RED}/{clip}_{LAT}__{d}/{clip}_inpainting_results_sbs.mkv", ctx=cpu(0))[FR].asnumpy()
        imgs[d] = a[:, a.shape[1] // 2:].astype(np.float32)
    for nm in ("E", "F"):
        y, x = cc[nm]["yx"]
        c = lambda a: a[y:y + 384, x:x + 384]
        tiles = [c(imgs["stock"]).astype(np.uint8)]
        labels = ["stock"]
        for d in ROWS[1:]:
            df = np.abs(c(imgs[d]) - c(imgs["stock"])).mean(-1) * 8
            tiles.append(np.repeat(np.clip(df, 0, 255).astype(np.uint8)[..., None], 3, 2))
            labels.append(f"|{d}-stock| x8 (mean {np.abs(c(imgs[d]) - c(imgs['stock'])).mean():.2f})")
        sep = np.full((384, 4, 3), 255, np.uint8)
        body = tiles[0]
        for t in tiles[1:]:
            body = np.concatenate([body, sep, t], 1)
        strip = np.zeros((28, body.shape[1], 3), np.uint8)
        for i, t in enumerate(labels):
            cv2.putText(strip, t, (i * 388 + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(f"{OUTD}/{clip}_f{FR:03d}_{nm}_diff8.png", cv2.cvtColor(np.concatenate([strip, body], 0), cv2.COLOR_RGB2BGR))
        print(clip, nm, labels[1:])
