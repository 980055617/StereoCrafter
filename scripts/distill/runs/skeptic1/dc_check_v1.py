#!/usr/bin/env python
"""Independent check of beyond4's DC-offset mechanism claim:
(a) one cv2-mp4v write + decord read shifts the mean by ~-2.1/255 on the SAME pixels;
(b) the splatting input sits ~+2.1/255 above the train GT, so the two cancel.
"""
import os, sys, math, tempfile
import numpy as np, torch
sys.path.insert(0, "/home/kawa/master_project/StereoCrafter"); os.chdir("/home/kawa/master_project/StereoCrafter")
from decord import VideoReader, cpu
from utils.inpainting import write_video_opencv, read_and_prepare_video

clip = sys.argv[1] if len(sys.argv) > 1 else "0301"
src = f"outputs/beyond4_lossless/clips/{clip}_origin_ll/{clip}_inpainting_results_sbs.mkv"
vr = VideoReader(src, ctx=cpu(0))
A = vr.get_batch(list(range(len(vr)))).asnumpy()
m0 = A.astype(np.float64).mean() / 255.0
d = tempfile.mkdtemp(prefix="dccheck_")
p = os.path.join(d, "rt_sbs.mp4")
write_video_opencv(A, 29.97, p)
B = VideoReader(p, ctx=cpu(0)).get_batch(list(range(len(A)))).asnumpy()
m1 = B.astype(np.float64).mean() / 255.0
mse = ((A.astype(np.float64) - B.astype(np.float64)) / 255.0 ** 1) ** 2
mse = (mse / 255.0 ** 1).mean() if False else (((A.astype(np.float64)/255.0) - (B.astype(np.float64)/255.0)) ** 2).mean()
print(f"[{clip}] (a) cv2-mp4v write + decord read on IDENTICAL pixels: mean {m0:.5f} -> {m1:.5f} "
      f"= {(m1-m0)*255:+.3f}/255   std gain {B.astype(np.float64).std()/A.astype(np.float64).std():.4f}   "
      f"PSNR {10*math.log10(1/max(mse,1e-18)):.2f} dB")

fps, fl, fw, fm = read_and_prepare_video(f"video_data/splatting/{clip}_splatting_results.mp4")
gt = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))
G = gt.get_batch(list(range(len(gt)))).asnumpy()
H, W = G.shape[1] // 2, G.shape[2] // 2
gtl = G[:, :H, :W].astype(np.float64) / 255.0
h, w = fl.shape[2], fl.shape[3]
# crop the GT left quadrant to the prepared input's size, centred
t0, l0 = (H - h) // 2, (W - w) // 2
n = min(len(gtl), fl.shape[0])
src_in = fl[:n].permute(0, 2, 3, 1).numpy().astype(np.float64)
ref = gtl[:n, t0:t0 + h, l0:l0 + w] if (H >= h and W >= w) else None
if ref is None or ref.shape != src_in.shape:
    print(f"[{clip}] (b) SKIPPED: prepared input {src_in.shape} vs GT quadrant crop {None if ref is None else ref.shape}")
else:
    mse2 = ((src_in - ref) ** 2).mean()
    print(f"[{clip}] (b) splatting input vs train-GT left quadrant, same crop, NO MODEL: "
          f"mean {src_in.mean():.5f} vs {ref.mean():.5f} = {(src_in.mean()-ref.mean())*255:+.3f}/255   "
          f"PSNR {10*math.log10(1/max(mse2,1e-18)):.2f} dB")
