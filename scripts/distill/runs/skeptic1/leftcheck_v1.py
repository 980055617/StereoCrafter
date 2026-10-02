#!/usr/bin/env python
"""Adversarial check: is the LEFT half (pass-through) of two configs of the same clip identical?

score_clip.py's 'leftPSNR' column is PSNR of the output's left half against the GT left half,
printed to 2 decimals on a 6-frame subsample -- a weak proxy for "no bits were stolen".
This compares the two outputs' left halves DIRECTLY, every frame, every pixel.

usage: leftcheck_v1.py <clip> <ref.mp4> <label>=<path> [<label>=<path> ...]
"""
import sys, math, hashlib
import numpy as np
from decord import VideoReader, cpu

def dec(p):
    vr = VideoReader(p, ctx=cpu(0))
    return vr.get_batch(list(range(len(vr)))).asnumpy()

clip, refp = sys.argv[1], sys.argv[2]
ref = dec(refp)
T, H, W, _ = ref.shape
half = W // 2
refL = np.ascontiguousarray(ref[:, :, :half])
print(f"[{clip}] ref={refp}")
print(f"[{clip}] ref frames={T} shape={ref.shape} leftmd5={hashlib.md5(refL.tobytes()).hexdigest()}")
for spec in sys.argv[3:]:
    lab, p = spec.split('=', 1)
    a = dec(p)
    n = min(len(a), T)
    L = np.ascontiguousarray(a[:n, :, :half])
    same_shape = a.shape[1:] == ref.shape[1:]
    md5 = hashlib.md5(L.tobytes()).hexdigest()
    if same_shape:
        d = L.astype(np.int32) - refL[:n].astype(np.int32)
        nd = int((d != 0).sum()); tot = d.size
        mse = float((d.astype(np.float64) ** 2).mean())
        psnr = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
        # right half too, for orientation
        R = a[:n, :, half:].astype(np.int32); Rr = ref[:n, :, half:].astype(np.int32)
        rmse = float(((R - Rr).astype(np.float64) ** 2).mean())
        rpsnr = 10 * math.log10(255.0 ** 2 / rmse) if rmse > 0 else float('inf')
        print(f"[{clip}] {lab:22s} frames={len(a):4d} shape={a.shape[1:]} leftmd5={md5} "
              f"LEFT: bytesdiff={nd}/{tot} maxabs={int(np.abs(d).max())} psnr={psnr:.2f} "
              f"| RIGHT: psnr={rpsnr:.2f}  {'LEFT_BIT_IDENTICAL' if nd==0 else 'LEFT_DIFFERS'}")
    else:
        print(f"[{clip}] {lab:22s} frames={len(a):4d} shape={a.shape[1:]} SHAPE_MISMATCH vs ref {ref.shape[1:]}")
