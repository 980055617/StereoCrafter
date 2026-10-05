"""input_side lane, probe 2 step A (CPU, model-free): decomposition of the model's INPUT warped window for several
splat variants of one clip, all against the SAME GT regions: the REG_FRAME registration (median-5 per-frame shifts)
that score_input_v1.py computed for the R1 input (read from its JSON), at the same scored frames.
Operators: reviewlib.regions / decompose (imported) + score_input_v1.py's self_metrics (copied verbatim).
usage: input_decomp_v1.py <R1 scorer json> <out_json> <clip> <variant> [<variant> ...]
"""
import json
import os
import sys

import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, f"{REPO}/scripts/distill/runs/review_20261001")
import reviewlib as RL  # noqa: E402
os.chdir(REPO)
TH, TW = 576, 1024
RJ, OUTJ, CLIP = sys.argv[1], sys.argv[2], sys.argv[3]
VARS = sys.argv[4:]
assert not os.path.exists(OUTJ), f"refusing to overwrite {OUTJ}"
rj = json.load(open(RJ))
assert rj["clip"] == CLIP and rj["input_dir"].endswith("/R1"), (rj["clip"], rj["input_dir"])
frames = rj["decomp_frames"]
t0, l0 = rj["window"]
sm_ddy, sm_ddx = rj["reg"]["smooth_ddy"], rj["reg"]["smooth_ddx"]


def self_metrics(y):
    g = np.zeros_like(y)
    g[:, :, :-1] += np.abs(np.diff(y, axis=2))
    g[:, :-1, :] += np.abs(np.diff(y, axis=1))
    lo = g <= np.quantile(g, 0.50)
    hi = g >= np.quantile(g, 0.90)
    lap = RL._lap(y)
    col = np.abs(np.diff(y, axis=2))
    return dict(sharp=float(col.mean()), selfFlatHF=float(lap[lo[:, 1:-1, 1:-1]].mean()),
                selfFlatStripe=float(col[lo[:, :, :-1]].mean()), selfEdgeHF=float(lap[hi[:, 1:-1, 1:-1]].mean()))


vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
gts = []
for s in range(0, len(frames), 16):
    part = frames[s:s + 16]
    b = vt.get_batch(part).asnumpy()
    H, W = b.shape[1] // 2, b.shape[2] // 2
    for k, fi in enumerate(part):
        dy, dx = int(sm_ddy[fi]), int(sm_ddx[fi])
        gts.append(RL.gray(b[k, t0 + dy:t0 + dy + TH, W + l0 + dx:W + l0 + dx + TW]))
gts = np.stack(gts).astype(np.float32)
REGS = RL.regions(gts)
GTD = RL.decompose(REGS, gts)
GTS = self_metrics(gts)
out = dict(clip=CLIP, registration_from=RJ, frames=frames, gt=GTD, gt_self=GTS, variants={})
for v in VARS:
    d = f"/mnt/ssd_data/deep_20261004/input_side/inputs_v1/{CLIP}/{v}"
    Wp = np.load(f"{d}/warped.npy", mmap_mode="r")
    M = np.load(f"{d}/mask.npy", mmap_mode="r")
    y = np.stack([RL.gray(np.asarray(Wp[fi])) for fi in frames]).astype(np.float32)
    hole = float(np.mean([np.asarray(M[fi]).astype(np.float32).mean(-1).__gt__(127.5).mean() for fi in frames]))
    dd = RL.decompose(REGS, y)
    sm = self_metrics(y)
    out["variants"][v] = dict(decomp=dd, ratio={k: dd[k] / GTD[k] for k in ("flatHF", "edgeHF", "stripeE")},
                              self=sm, self_ratio={k: sm[k] / GTS[k] for k in sm}, hole_frac=hole,
                              params=json.load(open(f"{d}/params.json")))
    r = out["variants"][v]["ratio"]
    print(f"{CLIP} {v:8s} holes {hole*100:6.3f}%  stripeE/GT {r['stripeE']:.3f}  edgeHF/GT {r['edgeHF']:.3f}  "
          f"flatHF/GT {r['flatHF']:.3f}  halo {dd['haloFrac']:.2f}  selfFlatStripe {sm['selfFlatStripe']:.5f}  "
          f"selfEdgeHF {sm['selfEdgeHF']:.4f}  sharp {sm['sharp']:.4f}", flush=True)
# consistency: R1 row must equal the scorer's own input decomposition
if "R1" in out["variants"]:
    a, b = out["variants"]["R1"]["decomp"], rj["input_decomp"]
    out["check_R1_vs_scorer"] = {k: [a[k], b[k]] for k in ("flatHF", "edgeHF", "stripeE", "haloFrac")}
    ok = all(abs(a[k] - b[k]) <= 1e-6 * max(1.0, abs(b[k])) for k in ("flatHF", "edgeHF", "stripeE", "haloFrac"))
    out["check_R1_vs_scorer_pass"] = bool(ok)
    print(f"{CLIP} R1 row reproduces the scorer's input decomposition: {'PASS' if ok else 'FAIL'}", flush=True)
json.dump(out, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
