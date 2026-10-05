#!/usr/bin/env python
"""blur_diag HIRES_A (CONTEXT row): crop the shared 576x1024 window out of the EXISTING origin 1024x1792 renders
(outputs/finalcheck_20261004/validate/clips/<c>_origin_ll_1024x1792, lossless FFV1; read-only).
Inner offset (224, 384) = deployed /128-crop + centre-crop math for both formats.  Gate G4: the cropped LEFT half must be
byte-identical to the native origin_ll render's left half on ALL frames, else the clip is not written.
Writes outputs/deep_20261004/blur_diag/hiresA_crop_r2/<c>_origin_hiresA_1024x1792/<c>_inpainting_results_sbs.mkv
(576x2048: cropped left | cropped right).  CPU only.
usage: python crop_hiresA_r2.py <clip> [...]"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
_spec = importlib.util.spec_from_file_location("ffv1", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
SRC = "outputs/finalcheck_20261004/validate/clips"
NAT = "outputs/beyond4_lossless/clips"
OUT = "outputs/deep_20261004/blur_diag/hiresA_crop_r2"
Y0, X0, TH, TW = 224, 384, 576, 1024


def ffv1_write(arr, fps, path):
    import subprocess
    cmd = ["/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{arr.shape[2]}x{arr.shape[1]}", "-r", f"{float(fps):.6f}", "-i", "-", "-an",
           "-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1", "-threads", "8", "-pix_fmt", "bgr0", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in arr:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0


for clip in sys.argv[1:]:
    src = f"{SRC}/{clip}_origin_ll_1024x1792/{clip}_inpainting_results_sbs.mkv"
    nat = f"{NAT}/{clip}_origin_ll/{clip}_inpainting_results_sbs.mkv"
    od = f"{OUT}/{clip}_origin_hiresA_1024x1792"
    dst = f"{od}/{clip}_inpainting_results_sbs.mkv"
    if os.path.exists(dst):
        print(f"{clip}: exists -> skip")
        continue
    a, b = VideoReader(src, ctx=cpu(0)), VideoReader(nat, ctx=cpu(0))
    n = len(a)
    assert n == len(b), (n, len(b))
    fps = float(a.get_avg_fps())
    out = np.empty((n, TH, 2 * TW, 3), np.uint8)
    ok = True
    for s in range(0, n, 16):
        idx = list(range(s, min(n, s + 16)))
        A = a.get_batch(idx).asnumpy()
        Bn = b.get_batch(idx).asnumpy()
        assert A.shape[1:] == (1024, 3584, 3), A.shape
        L = A[:, Y0:Y0 + TH, X0:X0 + TW]
        R = A[:, Y0:Y0 + TH, 1792 + X0:1792 + X0 + TW]
        ok &= bool(np.array_equal(L, Bn[:, :, :TW]))
        out[s:s + len(idx)] = np.concatenate([L, R], axis=2)
    print(f"{clip}: G4 left-half byte identity over {n} frames: {'PASS' if ok else 'FAIL'}")
    if not ok:
        continue
    os.makedirs(od, exist_ok=True)
    ffv1_write(out, fps, dst)
    dig = hashlib.md5(out.tobytes()).hexdigest()
    json.dump(dict(src=src, native=nat, inner_offset=[Y0, X0], n=n, fps=fps, md5=dig, G4="PASS"),
              open(f"{od}/crop_meta.json", "w"), indent=1)
    print(f"{clip}: wrote {dst} md5 {dig}")
