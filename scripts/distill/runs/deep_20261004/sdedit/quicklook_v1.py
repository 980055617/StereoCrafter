#!/usr/bin/env python
"""Smoke sanity look (CPU, no metric): right halves of several renders at one frame, whole window at half size plus
one 256x256 native crop (2x) at the densest-hole block, stacked side by side with their labels.
usage: quicklook_v1.py OUT.png clip frame label=path [label=path ...]
"""
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT, clip, fi = sys.argv[1], sys.argv[2], int(sys.argv[3])
specs = [s.split("=", 1) for s in sys.argv[4:]]
TH, TW = 576, 1024
sp = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))[fi].asnumpy()
hs, ws = sp.shape[0] // 2, sp.shape[1] // 2
st0, sl0 = (hs // 128 * 128 - TH) // 2, (ws // 128 * 128 - TW) // 2
holes = sp[hs + st0:hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) / 255.0 >= 0.5
hd = cv2.blur(holes.astype(np.float32), (64, 64))
hd[:128] = hd[-128:] = -1
hd[:, :128] = hd[:, -128:] = -1
y, x = np.unravel_index(int(np.argmax(hd)), hd.shape)
y, x = y - 128, x - 128
cols = []
for lab, p in specs:
    r = VideoReader(p, ctx=cpu(0))[fi].asnumpy()[:, TW:]
    a = cv2.resize(r, (TW // 2, TH // 2), interpolation=cv2.INTER_AREA)
    b = cv2.resize(r[y:y + 256, x:x + 256], (512, 512), interpolation=cv2.INTER_NEAREST)
    col = np.concatenate([a, b], 0).copy()
    cv2.putText(col, lab, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cols.append(col)
img = np.concatenate(cols, 1)
cv2.imwrite(OUT, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(OUT, "crop", y, x)
