#!/usr/bin/env python
"""EXPLORATORY (PREREG_ADDENDUM_3): does LPIPS reward outputs the user has rejected?  Rows built WITHOUT any model
change: COMP_origin = model input BR with origin's pixels in the holes (the user-rejected "mask-only compositing");
COMP_telea = BR with the holes filled by cv2 Telea inpainting (no generative model at all).  Scored like M6:
LPIPS-alex UNREG / REG_FRAME + PSNR (S38); LPIPS-vgg, DISTS, reviewlib flatHF / edgeHF / stripeE vs GT (S6).
One clip per call.  CPU only.   usage: python job_comp_v1.py <out_dir> <clip>"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeplib as S  # noqa: E402

sys.path.insert(0, os.path.join(S.REPO, "scripts/distill/runs/review_20261001"))
import reviewlib as RL  # noqa: E402

NT = int(os.environ.get("SK_THREADS", "4"))
torch.set_num_threads(NT)
cv2.setNumThreads(NT)
OUT, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
oj = os.path.join(OUT, f"{CLIP}.json")
assert not os.path.exists(oj), f"refusing to overwrite {oj}"
T0 = time.time()
TH, TW = S.TH, S.TW
js, jpath = S.regjson(CLIP)
frames = list(js["frames"])
S6 = frames[0::7][:6]
s6i = [frames.index(f) for f in S6]
D = S.load_clip(CLIP, frames, want_splat=True)
holes = D["BLext"][:, S.MY:S.MY + TH, S.MX:S.MX + TW].astype(np.float32).mean(-1) > 127.5
BR = np.ascontiguousarray(D["BRext"][:, S.MY:S.MY + TH, S.MX:S.MX + TW])
rows = {}
for r in ["origin", "s25"]:
    _, rows[r] = S.render_right(S.row_path(CLIP, r), frames)
V = {"origin": rows["origin"], "s25": rows["s25"],
     "COMP_origin": np.where(holes[..., None], rows["origin"], BR),
     "COMP_telea": np.stack([S.inpaint_holes(BR[j], holes[j]) for j in range(len(frames))])}
gt = {v: np.stack([S.box_crop(D["TR"][j], *S.reg_shift(js, fi, v)) for j, fi in enumerate(frames)]) for v in ("UNREG", "REG_FRAME")}
TRg6 = gt["REG_FRAME"][s6i]
lpA = S.Lp("alex")
import lpips  # noqa: E402
import piq  # noqa: E402
lpV = lpips.LPIPS(net="vgg", verbose=False).eval()
dists = piq.DISTS(reduction="none")
greg = RL.gray(TRg6)
reg = RL.regions(greg)
res = dict(clip=CLIP, frames=frames, S6=S6, hole_frac=float(holes.mean()), gt_decompose=RL.decompose(reg, greg),
           gtSharp_REG=S.sharp_score(gt["REG_FRAME"]), rows={})
for nm, X in V.items():
    e = {f"alex_{v}": float(np.mean(lpA(X, gt[v]))) for v in ("UNREG", "REG_FRAME")}
    e["psnr_REG"] = float(np.mean([S.psnr_u8(X[j], gt["REG_FRAME"][j]) for j in range(len(frames))]))
    e["sharp"] = S.sharp_score(X)
    with torch.no_grad():
        A6, G6 = S.t01(X[s6i]), S.t01(TRg6)
        e["vgg_REG"] = float(np.mean([float(lpV(A6[k:k + 1] * 2 - 1, G6[k:k + 1] * 2 - 1)) for k in range(len(s6i))]))
        e["dists_REG"] = float(dists(A6, G6).mean())
    e["decompose"] = RL.decompose(reg, RL.gray(X[s6i]))
    res["rows"][nm] = e
    print(f"[{CLIP} {time.time() - T0:6.1f}s] {nm:12s} alexU {e['alex_UNREG']:.4f} alexR {e['alex_REG_FRAME']:.4f} "
          f"psnrR {e['psnr_REG']:.2f} vgg {e['vgg_REG']:.4f} dists {e['dists_REG']:.4f} flatHF/GT "
          f"{e['decompose']['flatHF'] / res['gt_decompose']['flatHF']:.3f} stripeE/GT "
          f"{e['decompose']['stripeE'] / res['gt_decompose']['stripeE']:.3f}", flush=True)
res["seconds"] = time.time() - T0
json.dump(res, open(oj, "w"))
print("CLIP_DONE", CLIP, flush=True)
