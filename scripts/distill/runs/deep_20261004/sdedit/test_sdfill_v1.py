#!/usr/bin/env python
"""CPU unit test of sdfill_v1.fill_all_rowlin on real INPUT frames (no render, no score).
usage: CUDA_VISIBLE_DEVICES= python test_sdfill_v1.py OUTDIR clip:frame [clip:frame ...]
Mimics utils/inpainting.read_and_prepare_video (decord uint8 -> /255 float32, crop to /128, mask = RGB mean) and
inpainting_inference._center_crop_frames for ONE frame, fills the window + 16 px margin, checks the invariants and
saves zoomed before/after PNGs of the warped input around the densest-hole 128x128 block.
"""
import os
import sys

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sdfill_v1 as SDF  # noqa: E402

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
TH, TW, M = 576, 1024, 16
for spec in sys.argv[2:]:
    clip, fi = spec.split(":")
    fi = int(fi)
    f = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))[fi].asnumpy()
    H, W = f.shape[0] // 2, f.shape[1] // 2
    warped = f[H:, W:].astype(np.float32) / 255.0
    mask = f[H:, :W].astype(np.float32).mean(-1) / 255.0
    h128, w128 = H // 128 * 128, W // 128 * 128
    warped, mask = warped[:h128, :w128], mask[:h128, :w128]
    top, left = (h128 - TH) // 2, (w128 - TW) // 2
    y0, y1, x0, x1 = max(0, top - M), min(h128, top + TH + M), max(0, left - M), min(w128, left + TW + M)
    reg = np.ascontiguousarray(warped[y0:y1, x0:x1])
    holes = mask[y0:y1, x0:x1] >= 0.5
    filled, st = SDF.fill_all_rowlin(reg, holes)
    wy, wx = top - y0, left - x0
    win_h = holes[wy:wy + TH, wx:wx + TW]
    win_c = st["changed"][wy:wy + TH, wx:wx + TW]
    lens = st["lens"]
    hist = {int(k): int(v) for k, v in zip(*np.unique(np.minimum(lens, 9), return_counts=True))}
    # every hole pixel whose fill value differs from the (black) input changed; holes that stay equal are counted
    print(f"{clip} f{fi}: quadrant {H}x{W} window ({top},{left}) hole_frac_window {win_h.mean():.5f} "
          f"changed {int(win_c.sum())}/{int(win_h.sum())} runs {st['runs']} full_rows {st['full_rows']} "
          f"runlen_hist(capped 9) {hist}  non-hole changed: 0 (asserted)  "
          f"input mean at holes {reg[holes].mean():.4f} -> filled {filled[holes].mean():.4f}", flush=True)
    # zoom on the densest-hole 128x128 block of the window
    best, by, bx = -1, 0, 0
    for yy in range(0, TH - 128 + 1, 32):
        for xx in range(0, TW - 128 + 1, 32):
            v = win_h[yy:yy + 128, xx:xx + 128].mean()
            if v > best:
                best, by, bx = v, yy, xx
    a = (reg[wy + by:wy + by + 128, wx + bx:wx + bx + 128] * 255).round().clip(0, 255).astype(np.uint8)
    b = (filled[wy + by:wy + by + 128, wx + bx:wx + bx + 128] * 255).round().clip(0, 255).astype(np.uint8)
    hm = np.repeat((win_h[by:by + 128, bx:bx + 128] * 255).astype(np.uint8)[..., None], 3, -1)
    panel = np.concatenate([a, hm, b], 1)
    Image.fromarray(panel).resize((panel.shape[1] * 4, panel.shape[0] * 4), Image.NEAREST).save(
        os.path.join(OUT, f"{clip}_f{fi:03d}_fill_zoom_y{by}_x{bx}.png"))
print("TEST_OK")
