#!/usr/bin/env python
"""Magnified (3x nearest, no interpolation) sub-windows of each review crop, for close reading.

Writes to the scratchpad by default -- these are an inspection aid, not the deliverable.
usage: zoom_inspect.py <outdir> [sub=128] [zoom=3]
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reviewlib as R  # noqa: E402

OUTD = sys.argv[1]
SUB = int(sys.argv[2]) if len(sys.argv) > 2 else 128
Z = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(OUTD, exist_ok=True)
M = json.load(open("outputs/review_20261001/metrics.json"))
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception:
    FONT = ImageFont.load_default()

for clip, info in M.items():
    f = info["frame"]
    t0, l0 = info["t0"], info["l0"]
    _, TR, _, _, _, _ = R.tile_quadrants(clip, f)
    ppaths = [(lab, R.panel_path(clip, tag, roots)) for lab, tag, roots in R.PANELS]
    for rtype, c in info["crops"].items():
        y0, x0, S = c["y"], c["x"], c["size"]
        ddy, ddx = c["gtShift"]
        gt = TR[t0 + y0 + ddy:t0 + y0 + ddy + S, l0 + x0 + ddx:l0 + x0 + ddx + S]
        # sub-window: highest GT gradient energy, so the zoom lands on real structure
        g = gt.astype(np.float32).mean(axis=2)
        en = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
        ii = np.zeros((en.shape[0] + 1, en.shape[1] + 1), np.float64)
        ii[1:, 1:] = en.cumsum(0).cumsum(1)

        def box(y, x, s=SUB):
            return ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]

        rng = range(0, S - SUB, 8)
        _, sy, sx = max((box(y, x), y, x) for y in rng for x in rng)
        pan = [("GT", gt[sy:sy + SUB, sx:sx + SUB])]
        for lab, p in ppaths:
            a = R.grab(p, f)[:, R.TW:][y0:y0 + S, x0:x0 + S]
            pan.append((lab, a[sy:sy + SUB, sx:sx + SUB]))
        GAP, BAR = 8, 22
        W = SUB * Z * len(pan) + GAP * (len(pan) - 1)
        img = Image.new("RGB", (W, SUB * Z + BAR), (14, 14, 14))
        d = ImageDraw.Draw(img)
        for k, (lab, arr) in enumerate(pan):
            x = k * (SUB * Z + GAP)
            im = Image.fromarray(np.ascontiguousarray(arr)).resize((SUB * Z, SUB * Z), Image.NEAREST)
            img.paste(im, (x, BAR))
            d.text((x + 3, 4), f"{k}.{lab}", fill=(255, 220, 90) if lab == "THIS DELIVERABLE"
                   else (120, 255, 120) if k == 0 else (235, 235, 235), font=FONT)
        img.save(f"{OUTD}/{clip}_{rtype}_sub{SUB}x{Z}.png")
        print(clip, rtype, "sub at", (sy, sx), "->", f"{OUTD}/{clip}_{rtype}_sub{SUB}x{Z}.png", flush=True)
