#!/usr/bin/env python
"""Visual evidence for the stripes lane (CPU only).  For each (clip, frame, crop) writes one PNG grid, 100 % pixels:
  row 0: GT right eye (disparity-registered, local shift) | INPUT warped none | INPUT rowlin | INPUT telea | crack mask
  row 1: origin baseline | origin A rowlin | origin B telea | origin C rowlin+shrink | |A - baseline| x8
  row 2: deliv baseline  | deliv A rowlin  | deliv B telea  | deliv C rowlin+shrink  | |A - baseline| x8
usage: strips_v1.py OUTDIR
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decomp_v1 as D  # noqa: E402
import crackfill as CF  # noqa: E402
import torch  # noqa: E402
R = D.R

OUTD = sys.argv[1]
os.makedirs(OUTD, exist_ok=True)
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
except Exception:
    FONT = ImageFont.load_default()
S = 384
CROPS = [  # (clip, frame, y, x, local GT shift (ddy,ddx), name) -- 0301/0147 windows and shifts from review_20261001
    ("0301", 40, 184, 40, (0, -11), "hole"),
    ("0147", 8, 0, 400, (0, -41), "hole"),
    ("0301", 40, 0, 632, (1, -16), "texture"),
]
ROOT = "outputs/more_20261004/stripes/clips"
BASE = {"origin": "outputs/beyond4_lossless/clips/{c}_origin_ll/{c}_inpainting_results_sbs.mkv",
        "deliv": "outputs/finalcheck_20261004/speed/clips/{c}_deliv_g100_T5nat/{c}_inpainting_results_sbs.mkv"}
VAR = {"origin": "origin_g101_s8", "deliv": "deliv_g100_T5nat"}


def right(path, f):
    return R.grab(path, f)[:, R.TW:]


for clip, f, y, x, (ddy, ddx), name in CROPS:
    t0, l0, H, W = D.window(clip)
    TL, TR, BL, BR, _, _ = R.tile_quadrants(clip, f)
    gt = TR[t0 + y + ddy:t0 + y + ddy + S, l0 + x + ddx:l0 + x + ddx + S]
    m, wp = D.splat_quads(clip, f)
    ins = {}
    for mode in ("none", "rowlin", "telea"):
        w = torch.from_numpy(wp.astype(np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
        mm = torch.from_numpy(m)[None, None].contiguous()
        _, cw = CF.process(w, mm, 0, 0, R.TH, R.TW, mode, "keep", maxw=3, margin=0)
        ins[mode] = (w[0].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        crack = cw[0]
    rows = [[("GT right (registered)", gt), ("INPUT none", ins["none"][y:y + S, x:x + S]),
             ("INPUT rowlin", ins["rowlin"][y:y + S, x:x + S]), ("INPUT telea", ins["telea"][y:y + S, x:x + S]),
             ("crack px (<=3 wide)", np.repeat((crack[y:y + S, x:x + S] * 255).astype(np.uint8)[..., None], 3, -1))]]
    for cfg in ("origin", "deliv"):
        b = right(BASE[cfg].format(c=clip), f)[y:y + S, x:x + S]
        r = [(f"{cfg} baseline", b)]
        va = None
        for v, nm in (("A_rowlin", "A rowlin"), ("B_telea", "B telea"), ("C_rowlinshrink", "C rowlin+shrink")):
            p = f"{ROOT}/{clip}_{VAR[cfg]}_{v}/{clip}_inpainting_results_sbs.mkv"
            if os.path.exists(p):
                a = right(p, f)[y:y + S, x:x + S]
                if v == "A_rowlin":
                    va = a
            else:
                a = np.zeros_like(b)
            r.append((f"{cfg} {nm}", a))
        dif = np.zeros_like(b) if va is None else np.clip(np.abs(va.astype(np.int16) - b.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
        r.append(("|A - baseline| x8", dif))
        rows.append(r)
    GAP, BAR = 8, 20
    ncol = max(len(r) for r in rows)
    img = Image.new("RGB", (ncol * (S + GAP), len(rows) * (S + BAR + GAP)), (16, 16, 16))
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        for j, (lab, arr) in enumerate(r):
            X, Y = j * (S + GAP), i * (S + BAR + GAP)
            img.paste(Image.fromarray(np.ascontiguousarray(arr)), (X, Y + BAR))
            d.text((X + 4, Y + 3), f"{clip} f{f} {name} ({y},{x}) {lab}", fill=(255, 220, 90), font=FONT)
    p = f"{OUTD}/{clip}_f{f}_{name}_grid_100pct.png"
    img.save(p)
    print(p)
