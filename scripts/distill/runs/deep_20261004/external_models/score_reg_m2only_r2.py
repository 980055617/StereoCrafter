#!/usr/bin/env python
"""REGISTERED-GT LPIPS on CPU from the PUBLISHED, model-independent registration (deviation D3, PREREG_r2.txt).

more_20261004/eval_robustness/score_registered_v1.py estimates the GT shifts from model-independent data (real right
eye vs the splatting BR, holes excluded) and stores them in outputs/more_20261004/eval_robustness/score_v1/<clip>.json
(reg.smooth_ddy/ddx = REG_FRAME, reg.clip_ddy/ddx = REG_CLIP, reg.raw_* = REG_FRAME_RAW, blk_local = BLK_LOCAL, window,
frames).  This script re-uses exactly those shifts and repeats the LPIPS part of that scorer (LPIPS-alex, inputs*2-1,
batches of 4 frames; block metric 3x4 blocks of 192x256, batches of 4 frames x 12 blocks) on CPU.
Gate (printed + stored): origin_ll and mstudent2_step800_deliv_ll must reproduce their published per-clip values
(all 5 variants) within 1e-5, else the clip's numbers for the new render are void.
usage: python score_reg_m2only_r2.py <out_dir> <clip> <label>=<render.mkv> [...]
VARIANT score_reg_m2only_r2.py: identical to score_reg_from_published_r2.py except env SKIP_REF=1 skips re-scoring
origin / deliverable (their PUBLISHED GPU values are then the reference; backend error measured on the regime set).
"""
import hashlib
import json
import math
import os
import sys
import time

import lpips
import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
torch.set_num_threads(int(os.environ.get("NTHREADS", "16")))
STEP, TH, TW = 4, 576, 1024
BY, BX, BH, BW = 3, 4, 192, 256
OUT, CLIP = sys.argv[1], sys.argv[2]
specs = [a.split("=", 1) for a in sys.argv[3:]]
os.makedirs(OUT, exist_ok=True)
oj = os.path.join(OUT, f"{CLIP}.json")
assert not os.path.exists(oj), f"refusing to overwrite {oj}"
T0 = time.time()
P = json.load(open(f"outputs/more_20261004/eval_robustness/score_v1/{CLIP}.json"))
assert P["step"] == STEP
t0, l0 = P["window"]
frames = P["frames"]
n = len(frames)
reg = P["reg"]
shifts = {"UNREG": [(0, 0)] * n,
          "REG_CLIP": [(reg["clip_ddy"], reg["clip_ddx"])] * n,
          "REG_FRAME": [(reg["smooth_ddy"][fi], reg["smooth_ddx"][fi]) for fi in frames],
          "REG_FRAME_RAW": [(reg["raw_ddy"][fi], reg["raw_ddx"][fi]) for fi in frames]}
VARIANTS = list(shifts)
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
GTQ = {}                                  # full right quadrant of the real right eye at the scored frames (uint8)
for fi in frames:
    f = vt[fi].asnumpy()
    H, W = f.shape[0] // 2, f.shape[1] // 2
    GTQ[fi] = f[:H, W:2 * W].copy()
assert [H, W] == P["quadrant"], (H, W, P["quadrant"])
net = lpips.LPIPS(net="alex").eval()


def crop(fi, ddy, ddx, by=0, bx=0, h=TH, w=TW):
    y, x = t0 + ddy + by, l0 + ddx + bx
    assert y >= 0 and x >= 0 and y + h <= H and x + w <= W
    return GTQ[fi][y:y + h, x:x + w]


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


res, gate = {}, {}
REFS = [] if os.environ.get("SKIP_REF") == "1" else [("origin_ll", P["configs"]["origin_ll"]["path"]),
                                                      ("mstudent2_step800_deliv_ll",
                                                       P["configs"]["mstudent2_step800_deliv_ll"]["path"])]
for lab, path in REFS + specs:
    vr = VideoReader(path, ctx=cpu(0))
    v = vr.get_batch([fi for fi in frames]).asnumpy()
    Lr, Rr = v[:, :, :TW], v[:, :, TW:]
    out = dict(path=path, n=n, md5_left=hashlib.md5(np.ascontiguousarray(Lr).tobytes()).hexdigest(),
               dy=P["configs"]["origin_ll"]["dy"], dx=P["configs"]["origin_ll"]["dx"], lpips_clip={}, rPSNR={},
               lpips={})
    R = t01(Rr)
    with torch.no_grad():
        for var in VARIANTS:
            t = t01(np.stack([crop(fi, *shifts[var][j]) for j, fi in enumerate(frames)]))
            tot, per = 0.0, []
            for i in range(0, n, 4):
                o = net(R[i:i + 4] * 2 - 1, t[i:i + 4] * 2 - 1)
                tot += float(o.sum())
                per += [float(x) for x in o.view(-1)]
            out["lpips_clip"][var] = tot / n
            out["lpips"][var] = per
            out["rPSNR"][var] = 10 * math.log10(1 / max(float((R - t).pow(2).mean(dim=(1, 2, 3)).mean()), 1e-12))
        per = []
        for j0 in range(0, n, 4):
            pr, pg = [], []
            for j in range(j0, min(j0 + 4, n)):
                fi = frames[j]
                for b_, (by, bx) in enumerate([(a, b) for a in range(BY) for b in range(BX)]):
                    sy, sx = P["blk_local"][str(fi)][b_][0], P["blk_local"][str(fi)][b_][1]
                    pr.append(Rr[j, by * BH:(by + 1) * BH, bx * BW:(bx + 1) * BW])
                    pg.append(crop(fi, sy, sx, by * BH, bx * BW, BH, BW))
            o = net(t01(np.stack(pr)) * 2 - 1, t01(np.stack(pg)) * 2 - 1).view(-1, BY * BX)
            per += [float(x) for x in o.mean(dim=1)]
        out["lpips_clip"]["BLK_LOCAL"] = float(np.mean(per))
        out["lpips"]["BLK_LOCAL"] = per
    if lab in P["configs"]:
        pub = P["configs"][lab]["lpips_clip"]
        d = {k: out["lpips_clip"][k] - pub[k] for k in VARIANTS + ["BLK_LOCAL"]}
        gate[lab] = dict(max_abs=max(abs(x) for x in d.values()), deltas=d,
                         pass_=max(abs(x) for x in d.values()) < 1e-5)
    res[lab] = out
    print(f"[{CLIP} {time.time() - T0:6.0f}s] {lab:28s} UNREG {out['lpips_clip']['UNREG']:.6f} REG_CLIP "
          f"{out['lpips_clip']['REG_CLIP']:.4f} REG_FRAME {out['lpips_clip']['REG_FRAME']:.6f} BLK_LOCAL "
          f"{out['lpips_clip']['BLK_LOCAL']:.4f}" + (f"  gate max|d| {gate[lab]['max_abs']:.1e}" if lab in gate else ""),
          flush=True)
ok = all(g["pass_"] for g in gate.values())
json.dump(dict(clip=CLIP, step=STEP, window=[t0, l0], frames=frames, quadrant=[H, W], source_registration=
               f"outputs/more_20261004/eval_robustness/score_v1/{CLIP}.json", gate=gate, gate_pass=ok, configs=res,
               seconds=time.time() - T0), open(oj, "w"))
print(f"REG_DONE {CLIP} gate_pass={ok} ({time.time() - T0:.0f}s)", flush=True)
