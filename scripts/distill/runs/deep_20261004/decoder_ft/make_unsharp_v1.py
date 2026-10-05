#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- GT-FREE UNSHARP-MASK BASELINE rows (CPU).  PREREG.txt (baseline the decoder must beat).

Filter exactly as the skeptic lane's P-rows (scripts/distill/runs/deep_20261004/skeptic/job_m6_v2.py apply_P "unsharp"):
  x = u8/255 (float32); y = x + a * (x - cv2.GaussianBlur(x, (0,0), sigma, borderType=BORDER_REFLECT)); q8 = round(clip(y)*255)
applied per frame to the RIGHT half of a stock-decoded SBS render; the left half is copied unchanged.  Written as FFV1 SBS
via beyond4/infer_lossless.py _ffv1_write (+ .md5, writer_md5.txt).
usage: python make_unsharp_v1.py <src_sbs.mkv> <out_root> <out_dir_name> <sigma> <a>
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
_spec = importlib.util.spec_from_file_location(
    "infer_lossless", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
_IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IL)          # FIRST: importing cv2 before torch/diffusers segfaults in this env
import cv2  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
cv2.setNumThreads(4)

SRC, OUTR, NAME, SIG, A = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
od = f"{OUTR}/{NAME}"
assert not os.path.exists(od), f"refusing to overwrite {od}"
clip = NAME[:4]
vr = VideoReader(SRC, ctx=cpu(0))
n = len(vr)
fps = float(VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0)).get_avg_fps())
arr = vr.get_batch(list(range(n))).asnumpy()
half = arr.shape[2] // 2
out = arr.copy()
for f in range(n):
    x = arr[f, :, half:].astype(np.float32) / 255.0
    y = x + A * (x - cv2.GaussianBlur(x, (0, 0), SIG, borderType=cv2.BORDER_REFLECT))
    out[f, :, half:] = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
os.makedirs(od)
p = f"{od}/{clip}_inpainting_results_sbs.mkv"
dig = hashlib.md5(np.ascontiguousarray(out).tobytes()).hexdigest()
_IL._ffv1_write(np.ascontiguousarray(out), fps, p)
open(p + ".md5", "w").write(f"{dig}  {tuple(out.shape)}  fps={fps:.6f}\n")
open(f"{od}/writer_md5.txt", "w").write(f"{dig} {tuple(out.shape)} uint8 {clip}_inpainting_results_sbs.mp4\n")
json.dump(dict(src=SRC, sigma=SIG, a=A, n=n, md5=dig), open(f"{od}/unsharp.json", "w"), indent=1)
print(f"wrote {p} md5={dig}", flush=True)
