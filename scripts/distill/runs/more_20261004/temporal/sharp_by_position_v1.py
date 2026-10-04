#!/usr/bin/env python
"""more_20261004 / temporal lane, descriptive mechanism check (CPU only, lossless FFV1 decodes):
per-frame sharpness (score_clip_ll's statistic: mean |horizontal difference| of the right eye) grouped by the frame's
window-LOCAL position p (full-length windows k >= 1, kept positions 3..13), for origin_ll, BASE and T5nat_dcs14.
A decode_chunk_size-2 artefact would show as a sharpness pattern that alternates with p (pairs (2j, 2j+1)).
usage: sharp_by_position_v1.py OUT.txt CLIP [CLIP ...]"""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tlib import window_schedule
os.chdir("/home/kawa/master_project/StereoCrafter")
FF = "/home/kawa/miniconda3/envs/stereocrafter/bin/ffmpeg"
ROWS = [("origin", "outputs/beyond4_lossless/clips/{c}_origin_ll/{c}_inpainting_results_sbs.mkv"),
        ("BASE", "outputs/finalcheck_20261004/speed/clips/{c}_deliv_g100_T5nat/{c}_inpainting_results_sbs.mkv"),
        ("T5nat_dcs14", "outputs/more_20261004/temporal/clips/{c}_T5nat_dcs14/{c}_inpainting_results_sbs.mkv")]
out = sys.argv[1]; L = []
for c in sys.argv[2:]:
    L.append(f"clip {c}: mean sharpness by window-local position p (full windows k>=1); last col = std over p / mean")
    for lab, pat in ROWS:
        raw = subprocess.check_output([FF, "-v", "error", "-i", pat.format(c=c), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        v = np.frombuffer(raw, np.uint8).reshape(-1, 576, 2048, 3)[:, :, 1024:].astype(np.float32) / 255.
        sh = np.abs(v[:, :, 1:] - v[:, :, :-1]).mean(axis=(1, 2, 3))
        W = window_schedule(len(v), 14, 3)
        acc = {p: [] for p in range(3, 14)}
        for k, w in enumerate(W):
            if k == 0 or w["nf"] != 14:
                continue
            for p in range(3, 14):
                acc[p].append(sh[w["cur_i"] + p])
        m = np.array([np.mean(acc[p]) for p in range(3, 14)])
        L.append(f"  {lab:12s} " + " ".join(f"p{p}:{x:.5f}" for p, x in zip(range(3, 14), m)) + f"  cv {m.std() / m.mean():.4f}")
        print(L[-1], flush=True)
open(out, "w").write("\n".join(L) + "\n")
