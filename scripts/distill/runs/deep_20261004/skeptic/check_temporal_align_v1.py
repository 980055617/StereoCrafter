#!/usr/bin/env python
"""Data-integrity check (no quality score): is the model input (splatting TL = left eye used for the splat)
frame-aligned with left_eye_v2, and is the scoring GT (train TR) frame-aligned with right_eye_v2?
For each clip: MAD (0-255, luma-ish mean over RGB) between quadrant frame i and v2 frame i+k, k in [-4,4],
on the deployed 576x1024 window (crop to /128, centre), frames 8..8+NF.  CPU only.
usage: python check_temporal_align_v1.py <clip> [<clip> ...]
"""
import os, sys
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
TH, TW = 576, 1024
NF, F0 = 12, 8
KS = range(-4, 5)

def win(Hq, Wq):
    return (Hq // 128 * 128 - TH) // 2, (Wq // 128 * 128 - TW) // 2

def quad_frames(path, which, idx):
    vr = VideoReader(path, ctx=cpu(0))
    f0 = vr[0].asnumpy(); H, W = f0.shape[0] // 2, f0.shape[1] // 2
    t0, l0 = win(H, W)
    oy = 0 if which in ("TL", "TR") else H
    ox = 0 if which in ("TL", "BL") else W
    b = vr.get_batch([i for i in idx if i < len(vr)]).asnumpy()
    return b[:, oy + t0:oy + t0 + TH, ox + l0:ox + l0 + TW].astype(np.float32), len(vr), (H, W)

def v2_frames(path, idx, Hq, Wq):
    vr = VideoReader(path, ctx=cpu(0))
    f0 = vr[0].asnumpy()
    assert f0.shape[:2] == (Hq, Wq), (path, f0.shape, Hq, Wq)
    t0, l0 = win(Hq, Wq)
    b = vr.get_batch([i for i in idx if 0 <= i < len(vr)]).asnumpy()
    return b[:, t0:t0 + TH, l0:l0 + TW].astype(np.float32), len(vr)

for clip in sys.argv[1:]:
    print(f"=== {clip}  train -> {os.path.realpath(f'video_data/train/{clip}_train.mp4')}", flush=True)
    rng = list(range(F0, F0 + NF))
    ext = list(range(F0 - 4, F0 + NF + 4))
    for tag, path, which, v2 in (("splat TL vs left_v2", f"video_data/splatting/{clip}_splatting_results.mp4", "TL", f"video_data/left_eye_v2/{clip}.mp4"),
                                 ("train TL vs left_v2", f"video_data/train/{clip}_train.mp4", "TL", f"video_data/left_eye_v2/{clip}.mp4"),
                                 ("train TR vs right_v2", f"video_data/train/{clip}_train.mp4", "TR", f"video_data/right_eye_v2/{clip}.mp4"),
                                 ("train TR vs left_v2", f"video_data/train/{clip}_train.mp4", "TR", f"video_data/left_eye_v2/{clip}.mp4")):
        if not os.path.exists(path):
            print(f"  {tag:22s} MISSING {path}"); continue
        Q, nq, (H, W) = quad_frames(path, which, rng)
        V, nv = v2_frames(v2, ext, H, W)
        out = []
        for k in KS:
            m = []
            for j, i in enumerate(rng):
                jj = i + k - ext[0]
                if 0 <= jj < len(V):
                    d = Q[j] - V[jj]
                    m.append(np.abs(d - d.mean(axis=(0, 1))).mean())   # DC-removed MAD (colour-pipeline offsets)
            out.append((k, float(np.mean(m))))
        best = min(out, key=lambda t: t[1])
        print(f"  {tag:22s} nq={nq} nv={nv}  " + " ".join(f"{k:+d}:{v:5.2f}" for k, v in out) + f"   best k={best[0]:+d}", flush=True)
