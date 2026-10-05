"""Visual panel (CPU): real right eye (UNREGISTERED, deployed window) | inputs | renders, one frame, a crop.
usage: panel_v1.py <clip> <frame> <y0> <x0> <h> <w> <out.png> <label>=<input_variant_dir_or_sbs.mkv> ...
  a spec ending in .mkv shows the render's right half; otherwise the input variant's warped.npy."""
import os, sys
import numpy as np
import cv2
from decord import VideoReader, cpu
REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
import json
clip, fi, y0, x0, h, w, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
assert not os.path.exists(out), out
meta = json.load(open(f"/mnt/ssd_data/deep_20261004/input_side/prep_v2/{clip}/meta.json"))
top, lft = meta["top"], meta["lft"]
vt = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))
f = vt[fi].asnumpy(); H, W = f.shape[0] // 2, f.shape[1] // 2
tiles = [("GT right (unreg)", f[top:top + 576, W + lft:W + lft + 1024])]
for spec in sys.argv[8:]:
    lab, p = spec.split("=", 1)
    if p.endswith(".mkv"):
        im = VideoReader(p, ctx=cpu(0))[fi].asnumpy()[:, 1024:]
    else:
        im = np.load(os.path.join(p, "warped.npy"), mmap_mode="r")[fi]
    tiles.append((lab, np.asarray(im)))
crops = []
for lab, im in tiles:
    c = np.ascontiguousarray(im[y0:y0 + h, x0:x0 + w])
    c = cv2.resize(c, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    crops.append(c)
ncol = 3
rows = [np.concatenate(crops[i:i + ncol] + [np.zeros_like(crops[0])] * (ncol - len(crops[i:i + ncol])), 1) for i in range(0, len(crops), ncol)]
cv2.imwrite(out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
print("wrote", out)
