#!/usr/bin/env python
"""blur_diag (deep_20261004) -- no-reference IQA for every row of one clip (GPU, pyiqa offline).  PREREG.txt M4.

PRIMARY musiq (KonIQ ckpt), clipiqa, niqe; EXTRA topiq_nr, clipiqa+, arniqa.  Full 576x1024 RGB frames as they are
(float [0,1], batch 1), the 38 scored frames, clip value = frame mean.
env: PYTHONPATH=/mnt/ssd_data/deep_20261004/blur_diag/pylib TORCH_HOME=.../torch_home HF_HOME=.../hf_home
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python score_nr_r2.py <out_dir> <clip>
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402
import pyiqa  # noqa: E402

OUTD, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}.json"
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
T0 = time.time()
METRICS = ["musiq", "clipiqa", "niqe", "topiq_nr", "clipiqa+", "arniqa"]


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


M = B.meta(CLIP)
FR = M["frames"]
paths = B.row_paths(CLIP)
dev = torch.device("cuda")
mets = {m: pyiqa.create_metric(m, device=dev) for m in METRICS}
lower = {m: bool(mets[m].lower_better) for m in METRICS}
log(f"pyiqa {pyiqa.__version__}; lower_better {lower}")
ROWS = ["GT", "LEFT"] + [r for r in ["VAE_GT", "VAE_GT32", "VAE_GTx", "RS_GTx", "RS_GTx_L", "BR", "VAE_BR", "ORIGIN", "DELIV", "S25",
                                     "T5NAT", "T5PAD", "HIRES_A", "HIRES_B", "HIRES_B_L"] if r in paths or r == "BR"]
res = {}
with torch.no_grad():
    for row in ROWS:
        x = B.load_row(CLIP, row, FR, paths)
        per = {m: [] for m in METRICS}
        for f in range(len(x)):
            t = torch.from_numpy(x[f]).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(dev)
            for m in METRICS:
                per[m].append(float(mets[m](t)))
        res[row] = dict(perframe=per, clip={m: float(np.mean(v)) for m, v in per.items()})
        log(f"{row:9s} " + "  ".join(f"{m} {np.mean(v):.4f}" for m, v in per.items()))
json.dump(dict(clip=CLIP, frames=FR, metrics=METRICS, lower_better=lower, pyiqa=pyiqa.__version__, rows=res,
               seconds=time.time() - T0), open(OJ, "w"))
log(f"wrote {OJ}")
print("CLIP_DONE", CLIP, flush=True)
