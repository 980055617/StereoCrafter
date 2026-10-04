#!/usr/bin/env python
"""Distances between decoded RIGHT halves of lossless SBS renders (CPU only).  Not a quality score (no GT):
used for the smoke alarm and for X2 (are the higher-order solvers closer to each other than to s25?).
usage: psnr_dist_v1.py OUT.txt NFRAMES(0=all common) A.mkv B.mkv [C.mkv D.mkv ...]   (pairs)"""
import subprocess, sys, math
import numpy as np
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
FFPROBE = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffprobe"

def frames(path, nmax):
    w, h = [int(x) for x in subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height", "-of", "csv=p=0", path]).decode().strip().split(",")[:2]]
    p = subprocess.Popen([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out = []
    while nmax <= 0 or len(out) < nmax:
        b = p.stdout.read(w * h * 3)
        if not b:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(h, w, 3)[:, w // 2:].copy())
    p.stdout.close(); p.wait()
    return np.stack(out)

outf, n = sys.argv[1], int(sys.argv[2])
paths = sys.argv[3:]
assert len(paths) % 2 == 0
lines = []
for a, b in zip(paths[::2], paths[1::2]):
    A, B = frames(a, n), frames(b, n)
    m = min(len(A), len(B))
    d = A[:m].astype(np.float64) - B[:m].astype(np.float64)
    mse = float((d ** 2).mean())
    ps = 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))
    lines.append(f"PSNR_right {ps:7.3f} dB  meanAbs {float(np.abs(d).mean()):.4f}  frames {m}  A={a}  B={b}")
with open(outf, "a") as fh:
    for ln in lines:
        fh.write(ln + "\n"); print(ln)
