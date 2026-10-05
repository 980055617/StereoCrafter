#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- INSPECTION AID (not pre-registered): one region (y, x, size) of frame F of the <latlabel> renders,
rows GT-reg | stock | ftmse | ftema | cd, enlarged k x with NEAREST neighbour.  The region is given on the command line.
usage: python zoom_region_v1.py <out_png> <clip> <latlabel> <frame> <y> <x> <size> <k>
"""
import json, os, sys
import cv2
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT, CLIP, LAT = sys.argv[1], sys.argv[2], sys.argv[3]
FR, Y, X, S, K = map(int, sys.argv[4:9])
RED = "/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev"
R = json.load(open(f"outputs/vae_20261005/decoder_swap/score_reg_dev_v1/{CLIP}.json"))
t0, l0 = R["window"]; H, W = R["quadrant"]
ddy, ddx = int(R["reg"]["smooth_ddy"][FR]), int(R["reg"]["smooth_ddx"][FR])
tile = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))[FR].asnumpy()
imgs = {f"GT reg ({ddy:+d},{ddx:+d})": tile[t0 + ddy:t0 + ddy + 576, W + l0 + ddx:W + l0 + ddx + 1024]}
for d in ["stock", "ftmse", "ftema", "cd"]:
    a = VideoReader(f"{RED}/{CLIP}_{LAT}__{d}/{CLIP}_inpainting_results_sbs.mkv", ctx=cpu(0))[FR].asnumpy()
    imgs[d] = a[:, a.shape[1] // 2:]
tiles = [cv2.resize(np.ascontiguousarray(im[Y:Y + S, X:X + S]), (S * K, S * K), interpolation=cv2.INTER_NEAREST) for im in imgs.values()]
sep = np.full((S * K, 4, 3), 255, np.uint8)
body = tiles[0]
for t in tiles[1:]:
    body = np.concatenate([body, sep, t], 1)
strip = np.zeros((26, body.shape[1], 3), np.uint8)
for i, t in enumerate(imgs):
    cv2.putText(strip, f"{t} {K}x", (i * (S * K + 4) + 4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
cv2.imwrite(OUT, cv2.cvtColor(np.concatenate([strip, body], 0), cv2.COLOR_RGB2BGR))
print("wrote", OUT)
