#!/usr/bin/env python
"""M5 supplement (labelled LOSSY): origin with three whole-clip seeds on 0042 (outputs/fulldata/seedfloor, cv2 mp4v),
pairwise LPIPS, plus a same-clip codec control: the lossless origin render of 0042 re-encoded with cv2 mp4v
(new file under outputs/deep_20261004/skeptic/codec_ctrl_v1/).  CPU only.
usage: python job_m5supp_v1.py <out_json>"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeplib as S  # noqa: E402

torch.set_num_threads(int(os.environ.get("SK_THREADS", "4")))
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
T0 = time.time()
CLIP = "0042"
js, jp = S.regjson(CLIP)
frames = list(js["frames"])
D = S.load_clip(CLIP, frames, want_splat=False)
GT = np.stack([S.box_crop(D["TR"][j], *S.reg_shift(js, fi, "REG_FRAME")) for j, fi in enumerate(frames)])
lp = S.Lp("alex")
seeds = {}
for s in (1, 2, 3):
    p = f"outputs/fulldata/seedfloor/{CLIP}_origin_s{s}/{CLIP}_inpainting_results_sbs.mp4"
    vr = VideoReader(p, ctx=cpu(0))
    b = vr.get_batch(frames).asnumpy()
    assert b.shape[1:3] == (S.TH, 2 * S.TW), b.shape
    seeds[s] = b[:, :, S.TW:].copy()
    if s == 1:
        LEFT1 = b[:, :, :S.TW].copy()
ll_path = S.row_path(CLIP, "origin")
vr = VideoReader(ll_path, ctx=cpu(0))
fps = vr.get_avg_fps()
allf = vr.get_batch(list(range(len(vr)))).asnumpy()
ORIG = allf[frames][:, :, S.TW:]
cdir = "outputs/deep_20261004/skeptic/codec_ctrl_v1"
os.makedirs(cdir, exist_ok=True)
cpath = os.path.join(cdir, f"{CLIP}_origin_ll_reencoded_mp4v.mp4")
assert not os.path.exists(cpath)
w = cv2.VideoWriter(cpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * S.TW, S.TH))
for f in allf:
    w.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
w.release()
cb = VideoReader(cpath, ctx=cpu(0)).get_batch(frames).asnumpy()[:, :, S.TW:]
res = dict(clip=CLIP, frames=frames, note="LOSSY supplement: seedfloor renders are cv2 mp4v; whole-clip seeds 1/2/3")
res["left_half_check_psnr_seed1_vs_ll"] = float(np.mean([S.psnr_u8(LEFT1[j], allf[fi][:, :S.TW]) for j, fi in enumerate(frames)]))
res["codec_ctrl_lpips"] = float(np.mean(lp(cb, ORIG)))
res["codec_ctrl_psnr"] = float(np.mean([S.psnr_u8(cb[j], ORIG[j]) for j in range(len(frames))]))
res["pair"] = {}
for a, b in ((1, 2), (1, 3), (2, 3)):
    res["pair"][f"s{a}_s{b}"] = dict(lpips=float(np.mean(lp(seeds[a], seeds[b]))),
                                    psnr=float(np.mean([S.psnr_u8(seeds[a][j], seeds[b][j]) for j in range(len(frames))])))
res["toGT_REG"] = {f"s{s}": float(np.mean(lp(seeds[s], GT))) for s in seeds}
res["toGT_REG"]["origin_ll"] = float(np.mean(lp(ORIG, GT)))
res["toGT_REG"]["origin_ll_mp4v"] = float(np.mean(lp(cb, GT)))
res["to_origin_ll"] = {f"s{s}": float(np.mean(lp(seeds[s], ORIG))) for s in seeds}
res["seconds"] = time.time() - T0
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
