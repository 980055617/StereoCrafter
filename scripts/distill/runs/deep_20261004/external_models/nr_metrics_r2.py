#!/usr/bin/env python
"""NO-REFERENCE quality of the right eye (PREREG_r2.txt, NO-REF; descriptive, never gating).

pyiqa 0.1.14.1 (lane-local copy of blur_diag's install, offline weights in the lane torch_home): musiq (KonIQ, higher =
better), clipiqa (higher = better), niqe (lower = better).  Full 576x1024 right-eye frames as they are, frames 0,4,..
(score_clip_ll.py SCORE_STEP=4 sampling).  Rows: GT = real right eye at the UNREG window (origin's published offset),
then each label=render (right half of the SBS FFV1).  NR metrics can reward noise / fake texture: read next to LPIPS,
rightPSNR and the panels, never alone.
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python nr_metrics_r2.py <out_dir> <clip> label=path ...
"""
import json
import os
import sys
import time

import numpy as np
import pyiqa
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
PUB = "scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json"
STEP, TH, TW = 4, 576, 1024
OUT, CLIP = sys.argv[1], sys.argv[2]
specs = [a.split("=", 1) for a in sys.argv[3:]]
os.makedirs(OUT, exist_ok=True)
oj = os.path.join(OUT, f"{CLIP}.json")
assert not os.path.exists(oj), f"refusing to overwrite {oj}"
t0 = time.time()
pub = json.load(open(PUB))["cells"][CLIP]["origin_ll"]
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
f0 = vt[0].asnumpy()
H, W = f0.shape[0] // 2, f0.shape[1] // 2
r0, c0 = (H - TH) // 2 + pub["dy"], (W - TW) // 2 + pub["dx"]
idx = list(range(0, len(vt), STEP))
rows = {"GT": np.stack([vt[i].asnumpy()[r0:r0 + TH, W + c0:W + c0 + TW] for i in idx])}
for lab, p in specs:
    vr = VideoReader(p, ctx=cpu(0))
    ii = [i for i in idx if i < len(vr)]
    rows[lab] = vr.get_batch(ii).asnumpy()[:, :, TW:]
n = min(len(v) for v in rows.values())
dev = "cuda" if torch.cuda.is_available() else "cpu"
res = {}
for name in ("musiq", "clipiqa", "niqe"):
    m = pyiqa.create_metric(name, device=dev, as_loss=False)
    for lab, v in rows.items():
        vals = []
        with torch.no_grad():
            for j in range(n):
                x = torch.from_numpy(np.ascontiguousarray(v[j])).permute(2, 0, 1)[None].float().div(255.).to(dev)
                vals.append(float(m(x)))
        res.setdefault(lab, {})[name] = float(np.mean(vals))
        res[lab][name + "_frames"] = vals
    del m
    torch.cuda.empty_cache()
out = dict(clip=CLIP, n=n, step=STEP, gt_window=[r0, c0], specs=specs, means={k: {m: v[m] for m in
           ("musiq", "clipiqa", "niqe")} for k, v in res.items()}, rows=res, seconds=time.time() - t0)
json.dump(out, open(oj, "w"))
for lab in res:
    print(f"NR {CLIP} {lab:28s} musiq {res[lab]['musiq']:.3f} clipiqa {res[lab]['clipiqa']:.4f} niqe {res[lab]['niqe']:.3f}")
print("NR_DONE", CLIP, f"{time.time() - t0:.0f}s", flush=True)
