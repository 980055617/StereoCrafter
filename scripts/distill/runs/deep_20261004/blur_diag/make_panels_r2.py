#!/usr/bin/env python
"""blur_diag -- inspection panels (CPU only, viewing aid, not a metric).  Per clip, at the scored frame nearest the
middle (frame 76), the 192x256 block of a 3x4 grid with the highest GT gradient energy (blocks with > 5 % holes
skipped), shown x2 nearest for every stage row that exists:  GT | VAE_GT | BR | VAE_BR | ORIGIN | DELIV | S25 | HIRES_B
(| HIRES_A).  Render-geometry rows can sit a few px off GT (residual multi-plane disparity), by design.
usage: python make_panels_r2.py <out_dir> <clip> [...]"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402

OUTD = sys.argv[1]
os.makedirs(OUTD, exist_ok=True)
ROWS = ["GT", "VAE_GT", "BR", "VAE_BR", "ORIGIN", "DELIV", "S25", "HIRES_B", "HIRES_A"]
BH, BW = 192, 256
for clip in sys.argv[2:]:
    M = B.meta(clip)
    fr = [76] if 76 in M["frames"] else [M["frames"][len(M["frames"]) // 2]]
    paths = B.row_paths(clip)
    ims = {r: B.load_row(clip, r, fr, paths)[0] for r in ROWS if r in ("GT", "BR") or r in paths}
    hole, _ = B.holes(clip, fr)
    g = ims["GT"].astype(np.float32).mean(-1)
    gm = np.abs(np.diff(g, axis=1))[:-1] + np.abs(np.diff(g, axis=0))[:, :-1]
    best, bb = -1, (0, 0)
    for by in range(3):
        for bx in range(4):
            if hole[0, by * BH:(by + 1) * BH, bx * BW:(bx + 1) * BW].mean() > 0.05:
                continue
            e = gm[by * BH:(by + 1) * BH - 1, bx * BW:(bx + 1) * BW - 1].mean()
            if e > best:
                best, bb = e, (by, bx)
    y, x = bb[0] * BH, bb[1] * BW
    tiles = []
    for r in ROWS:
        if r not in ims:
            continue
        t = cv2.resize(ims[r][y:y + BH, x:x + BW], (2 * BW, 2 * BH), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, r, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    panel = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
    p = f"{OUTD}/{clip}_f{fr[0]:03d}_stages_y{y}x{x}.png"
    assert not os.path.exists(p), p
    cv2.imwrite(p, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print("wrote", p)
