#!/usr/bin/env python
"""more_20261004 / temporal lane: VIEWING AID ONLY (lossy H.264, never scored).  CPU only.

For one clip, decodes the lossless FFV1 renders exactly (ffmpeg rgb24) and writes
  <out>/<clip>_review_flicker.mp4 : top row    right eyes  [origin_ll | BASE (T5@1.00, decode 2) | T5@1.00 decode 14]
                                     bottom row |R_t - R_t-1| x GAIN of the same three (bright = changed since the
                                                previous frame; a steady video is dark except where things move)
  each panel 512x288 (2x downscale), 7 fps (the pipeline's fps) and a 2x slow-motion copy.
usage: make_review_v1.py OUTDIR CLIP [GAIN]
"""
import os
import subprocess
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
FFMPEG = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
FFPROBE = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffprobe"
out, clip = sys.argv[1], sys.argv[2]
GAIN = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
SRC = [("origin (deployed)", f"outputs/beyond4_lossless/clips/{clip}_origin_ll/{clip}_inpainting_results_sbs.mkv"),
       ("deliv T5@1.00 decode 2", f"outputs/finalcheck_20261004/speed/clips/{clip}_deliv_g100_T5nat/{clip}_inpainting_results_sbs.mkv"),
       ("deliv T5@1.00 decode 14", f"outputs/more_20261004/temporal/clips/{clip}_T5nat_dcs14/{clip}_inpainting_results_sbs.mkv")]


def read_right(path):
    w, h = [int(x) for x in subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         path]).decode().strip().split(",")[:2]]
    raw = subprocess.check_output([FFMPEG, "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    v = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return v[:, :, w // 2:]


vids = [read_right(p) for _, p in SRC]
n = min(len(v) for v in vids)
vids = [v[:n, ::2, ::2].astype(np.int16) for v in vids]          # 2x downscale by decimation (viewing only)
H, W = vids[0].shape[1:3]
frames = []
for t in range(n):
    top = [np.clip(v[t], 0, 255).astype(np.uint8) for v in vids]
    if t == 0:
        bot = [np.zeros_like(top[0]) for _ in vids]
    else:
        bot = [np.clip(np.abs(v[t] - v[t - 1]) * GAIN, 0, 255).astype(np.uint8) for v in vids]
    sep = np.full((H, 4, 3), 255, np.uint8)
    row1 = np.concatenate([top[0], sep, top[1], sep, top[2]], 1)
    row2 = np.concatenate([bot[0], sep, bot[1], sep, bot[2]], 1)
    frames.append(np.concatenate([row1, np.full((4, row1.shape[1], 3), 255, np.uint8), row2], 0))
fr = np.stack(frames)
os.makedirs(out, exist_ok=True)
for name, rate in ((f"{clip}_review_flicker.mp4", 7), (f"{clip}_review_flicker_slow2x.mp4", 3.5)):
    T, HH, WW, _ = fr.shape
    p = subprocess.Popen([FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WW}x{HH}",
                          "-r", str(rate), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                          os.path.join(out, name)], stdin=subprocess.PIPE)
    p.stdin.write(fr.tobytes())
    p.stdin.close()
    assert p.wait() == 0
with open(os.path.join(out, f"{clip}_review_README.txt"), "w") as fh:
    fh.write(f"VIEWING AID ONLY (H.264, lossy; never used for any number).  clip {clip}, {n} frames, 2x downscaled.\n"
             f"columns (left -> right): " + " | ".join(lbl for lbl, _ in SRC) + "\n"
             f"top row: right eye.  bottom row: |R_t - R_t-1| x {GAIN:g} (frame-to-frame change; flicker shows as\n"
             f"texture that lights up in static areas, pulsing every other frame for decode 2).\n"
             "sources:\n" + "".join(f"  {lbl}: {p}\n" for lbl, p in SRC))
print("wrote", out, clip, fr.shape)
