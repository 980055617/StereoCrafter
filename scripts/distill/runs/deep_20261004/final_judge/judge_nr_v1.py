#!/usr/bin/env python
"""final_judge: no-reference IQA (PREREG.txt section A/F6) for every row of one clip + the REG_FRAME-registered GT.
NIQE (lower better) and MUSIQ-KonIQ (higher better), pyiqa read-only from blur_diag's lane-local install, offline.
Frames: the 38 scored frames (SCORE_STEP=4) of the judge score JSON; full 576x1024 RGB float [0,1], batch 1 (blur_diag
score_nr_r2.py settings).  GT crop = real right eye at the per-frame REG_FRAME shift recorded in <score_dir>/<clip>.json.
env: PYTHONPATH=/mnt/ssd_data/deep_20261004/blur_diag/pylib TORCH_HOME=.../torch_home HF_HOME=.../hf_home
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
usage: CUDA_VISIBLE_DEVICES=g flock /tmp/claude-gpu<g>.lock python judge_nr_v1.py <score_dir> <out_dir> <clip>"""
import json, os, sys, time
import numpy as np
import torch
from decord import VideoReader, cpu
import pyiqa
os.chdir("/home/kawa/master_project/StereoCrafter")
SD, OUTD, CLIP = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}.json"
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
T0 = time.time()
S = json.load(open(f"{SD}/{CLIP}.json"))
R = json.load(open("scripts/distill/runs/deep_20261004/final_judge/ROWS_v1.json"))
FR = S["frames"]
TH, TW = 576, 1024
t0, l0 = S["window"]
H, W = S["quadrant"]
dev = torch.device("cuda")
METRICS = ["niqe", "musiq"]
mets = {m: pyiqa.create_metric(m, device=dev) for m in METRICS}
lower = {m: bool(mets[m].lower_better) for m in METRICS}


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


log(f"pyiqa {pyiqa.__version__} lower_better {lower} frames {len(FR)}")
stacks = {}
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
tiles = vt.get_batch(FR).asnumpy()
sdy, sdx = S["reg"]["smooth_ddy"], S["reg"]["smooth_ddx"]
stacks["GT_REGFRAME"] = np.stack([tiles[j, t0 + sdy[fi]:t0 + sdy[fi] + TH, W + l0 + sdx[fi]:W + l0 + sdx[fi] + TW]
                                  for j, fi in enumerate(FR)])
del tiles
cells = R["cells"][CLIP]
for lab in [l for l in R["labels"] if l in cells]:
    p = cells[lab]["path"]
    virt = p.startswith("LEFTASRIGHT:")
    p = p.split("LEFTASRIGHT:")[-1]
    v = VideoReader(p, ctx=cpu(0)).get_batch(FR).asnumpy()
    stacks[lab] = v[:, :, :TW] if virt else v[:, :, TW:]
res = {}
with torch.no_grad():
    for row, x in stacks.items():
        per = {m: [] for m in METRICS}
        for f in range(len(x)):
            t = torch.from_numpy(np.ascontiguousarray(x[f])).permute(2, 0, 1).float().div(255.).unsqueeze(0).to(dev)
            for m in METRICS:
                per[m].append(float(mets[m](t)))
        res[row] = dict(perframe=per, clip={m: float(np.mean(v)) for m, v in per.items()})
        log(f"{row:28s} " + "  ".join(f"{m} {np.mean(v):.4f}" for m, v in per.items()))
json.dump(dict(clip=CLIP, frames=FR, metrics=METRICS, lower_better=lower, pyiqa=pyiqa.__version__, rows=res,
               seconds=time.time() - T0), open(OJ, "w"))
log(f"wrote {OJ}")
print("CLIP_DONE", CLIP, flush=True)
