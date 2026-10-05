#!/usr/bin/env python
"""Visual stripe check near disocclusion holes (CPU, no metric).  For one clip, model and frame: the K densest-hole
96x96 blocks of the deployed window, each shown at 4x (nearest) for
  GT (REG_FRAME shift) | INPUT_warp | deployed | g1.00 control | sd31 | sd7 | sd1 | sd1fill | hole mask
usage: make_zoom_holes_v1.py <stage_dir> <outdir> <clip> <model origin|deliv> <frame> [K=3]
"""
import json
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

os.chdir("/home/kawa/master_project/StereoCrafter")
SD, OUT, clip, model, fi = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
K = int(sys.argv[6]) if len(sys.argv) > 6 else 3
os.makedirs(OUT, exist_ok=True)
TH, TW, B, Z = 576, 1024, 96, 4
rows = json.load(open(f"{SD}/ROWS.json"))["cells"][clip]
dep = {"origin": "origin_ll", "deliv": "mstudent2_step800_deliv_ll"}[model]
ctl = f"{model}_g100_s8"
labs = ["INPUT_warp", dep, ctl] + [f"{model}_g100_{v}" for v in ("sd31", "sd7", "sd1", "sd1fill")]
labs = [l for l in labs if l in rows]
rj = json.load(open(f"{SD}/reg/{clip}.json"))
t0, l0 = rj["window"]
ddy, ddx = int(rj["reg"]["smooth_ddy"][fi]), int(rj["reg"]["smooth_ddx"][fi])
tile = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))[fi].asnumpy()
H, W = tile.shape[0] // 2, tile.shape[1] // 2
gt = tile[t0 + ddy:t0 + ddy + TH, W + l0 + ddx:W + l0 + ddx + TW]
sp = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))[fi].asnumpy()
hs, ws = sp.shape[0] // 2, sp.shape[1] // 2
st0, sl0 = (hs // 128 * 128 - TH) // 2, (ws // 128 * 128 - TW) // 2
holes = sp[hs + st0:hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) / 255.0 >= 0.5
imgs = {l: VideoReader(rows[l]["path"], ctx=cpu(0))[fi].asnumpy()[:, TW:] for l in labs}
dens = cv2.blur(holes.astype(np.float32), (B, B))
blocks = []
d2 = dens.copy()
for _ in range(K):
    d2[:B // 2] = d2[-B // 2:] = -1
    d2[:, :B // 2] = d2[:, -B // 2:] = -1
    y, x = np.unravel_index(int(np.argmax(d2)), d2.shape)
    if d2[y, x] <= 0:
        break
    blocks.append((y - B // 2, x - B // 2, float(dens[y, x])))
    d2[max(0, y - B):y + B, max(0, x - B):x + B] = -1
out_rows = []
for (y, x, dn) in blocks:
    tiles = []
    for name, a in [("GT", gt)] + [(l, imgs[l]) for l in labs] + [("holes", np.repeat((holes * 255).astype(np.uint8)[..., None], 3, -1))]:
        c = cv2.resize(np.ascontiguousarray(a[y:y + B, x:x + B]), (B * Z, B * Z), interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, name.replace(f"{model}_g100_", ""), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        tiles.append(c)
    r = np.concatenate(tiles, 1)
    cv2.putText(r, f"y{y} x{x} hole density {dn:.3f}", (4, B * Z - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    out_rows.append(r)
p = os.path.join(OUT, f"{clip}_{model}_f{fi:03d}_holezoom.png")
cv2.imwrite(p, cv2.cvtColor(np.concatenate(out_rows, 0), cv2.COLOR_RGB2BGR))
print(p, blocks)
