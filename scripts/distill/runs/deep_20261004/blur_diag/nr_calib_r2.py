#!/usr/bin/env python
"""blur_diag PREREG ADDENDUM 4 (NR-CAL): how do the primary NR metrics respond to synthetic blur of the GT?  CPU only.
Clips 0042 0141 0204 0301, every 4th scored frame (10/clip); variants GT and GT * Gaussian(sigma) for sigma in
0.5 0.75 1.0 1.5 (cv2.GaussianBlur, ksize 0 -> derived from sigma, BORDER_REFLECT); metrics musiq clipiqa niqe (pyiqa,
CPU); b1/GT = DoG band-1 RMS ratio (score_detail_r2 definition, full frame, 24-px border).
env: as score_nr_r2.py (PYTHONPATH pylib, TORCH_HOME, HF_HOME, offline), CUDA_VISIBLE_DEVICES=''
usage: python nr_calib_r2.py <out_json>"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402
import pyiqa  # noqa: E402

OJ = sys.argv[1]
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
torch.set_num_threads(8)
CLIPS = ["0042", "0141", "0204", "0301"]
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.5]
METRICS = ["musiq", "clipiqa", "niqe"]
T0 = time.time()
mets = {m: pyiqa.create_metric(m, device=torch.device("cpu")) for m in METRICS}


def b1(u8):
    acc, cnt = 0.0, 0
    for f in u8:
        g = f.astype(np.float32).mean(-1) / 255.0
        b = g - gaussian_filter(g, 1, mode="reflect", truncate=4.0)
        b = b[24:-24, 24:-24]
        acc += float((b.astype(np.float64) ** 2).sum())
        cnt += b.size
    return float(np.sqrt(acc / cnt))


res = {}
for c in CLIPS:
    fr = B.meta(c)["frames"][::4]
    gt = B.load_row(c, "GT", fr)
    b0 = b1(gt)
    res[c] = {"frames": fr}
    for s in SIGMAS:
        x = gt if s == 0 else np.stack([cv2.GaussianBlur(f, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT)
                                        for f in gt])
        d = {"b1_ratio": b1(x) / b0}
        with torch.no_grad():
            for m in METRICS:
                d[m] = float(np.mean([float(mets[m](torch.from_numpy(f).permute(2, 0, 1).float().div(255.).unsqueeze(0)))
                                      for f in x]))
        res[c][f"sigma_{s}"] = d
        print(f"[{time.time() - T0:6.0f}s] {c} sigma {s:4.2f} b1/GT {d['b1_ratio']:.3f} " +
              " ".join(f"{m} {d[m]:.4f}" for m in METRICS), flush=True)
json.dump(dict(clips=CLIPS, sigmas=SIGMAS, metrics=METRICS, rows=res, seconds=time.time() - T0), open(OJ, "w"), indent=1)
print("wrote", OJ)
