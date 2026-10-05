#!/usr/bin/env python
"""deep_20261004 / sdedit lane: visual-check PNGs (CPU only).  One PNG per (clip, model):
  [ GT (real right eye, REG_FRAME shift of that frame) | deployed render | best SDEdit variant of that model ]
  row 1: whole 576x1024 window at half size
  row 2: 256x256 native crop (shown 2x, nearest) where |best - deployed| (64x64 box mean of the gray difference) is largest
  row 3: 256x256 native crop (shown 2x) where the disocclusion-hole density is largest (cracks / stripes live there)
best = the model's variant with the lowest stage-mean REG_FRAME (display choice only, PREREG.txt), read from TABLE json.
v2 (copy of make_panels_v1.py): env PANEL_VARIANT=<sd31|sd7|sd1|sd1fill> shows that variant instead of the best one
(used for the sd7 visual record of PREREG_ADDENDUM_2); the file name says which variant is shown.
frame = the decomposition frame (densest disocclusion inside the window, build_review.py rule) from decomp.json.
usage: make_panels_v1.py <stage_dir> <table.json> <outdir> clip [clip ...]
"""
import json
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

os.chdir("/home/kawa/master_project/StereoCrafter")
SD, TJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
tab = json.load(open(TJ))
rows = json.load(open(f"{SD}/ROWS.json"))
dec = json.load(open(f"{SD}/decomp/decomp.json"))
TH, TW = 576, 1024
DEP = {"origin": "origin_ll", "deliv": "mstudent2_step800_deliv_ll"}


def best_label(model):
    cand = [(np.mean([json.load(open(f"{SD}/reg/{c}.json"))["configs"][lab]["lpips_clip"]["REG_FRAME"]
                      for c in tab["clips"]]), lab)
            for lab, r in tab["results"].items() if r["model"] == model]
    return min(cand)


def box(a, k=64):
    return cv2.blur(a.astype(np.float32), (k, k))


def put(img, s, x, y, scale=0.6):
    cv2.putText(img, s, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1)


for clip in sys.argv[4:]:
    geo = dec[clip]["geometry"]
    fi = int(geo["frame"])
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
    for model, dep in DEP.items():
        mbest, blab = best_label(model)
        if os.environ.get("PANEL_VARIANT"):
            blab = f"{model}_g100_{os.environ['PANEL_VARIANT']}"
            mbest = float(np.mean([json.load(open(f"{SD}/reg/{c}.json"))["configs"][blab]["lpips_clip"]["REG_FRAME"]
                                   for c in tab["clips"]]))
        dpath = rows["cells"][clip][dep]["path"]
        bpath = rows["cells"][clip][blab]["path"]
        R = {k: VideoReader(p, ctx=cpu(0))[fi].asnumpy()[:, TW:] for k, p in (("dep", dpath), ("best", bpath))}
        diff = box(np.abs(R["best"].astype(np.float32).mean(-1) - R["dep"].astype(np.float32).mean(-1)))
        hd = box(holes.astype(np.float32))
        crops = []
        for m_ in (diff, hd):
            mm = m_.copy()
            mm[:128, :] = mm[-128:, :] = -1
            mm[:, :128] = mm[:, -128:] = -1
            y, x = np.unravel_index(int(np.argmax(mm)), mm.shape)
            crops.append((y - 128, x - 128))
        top = np.concatenate([cv2.resize(a, (TW // 2, TH // 2), interpolation=cv2.INTER_AREA) for a in (gt, R["dep"], R["best"])], 1)
        names = [f"GT real right eye (REG_FRAME shift {ddy:+d},{ddx:+d})", f"deployed {dep}", f"{'SDEdit' if os.environ.get('PANEL_VARIANT') else 'best SDEdit'} {blab}"]
        for q, s in enumerate(names):
            put(top, s, q * (TW // 2) + 6, 20, 0.5)
        put(top, f"{clip} frame {fi}  stage-mean REG_FRAME of the shown variant {mbest:.4f}", 6, TH // 2 - 8, 0.5)
        rws = [top]
        for (y, x), why in zip(crops, ("max |best - deployed|", "max hole density")):
            tiles = []
            for a in (gt, R["dep"], R["best"]):
                c = a[y:y + 256, x:x + 256]
                tiles.append(cv2.resize(c, (512, 512), interpolation=cv2.INTER_NEAREST))
            row = np.concatenate(tiles, 1)
            put(row, f"crop y{y} x{x} (256x256 native, 2x) at {why}", 6, 20, 0.5)
            rws.append(row)
        panel = np.concatenate(rws, 0)
        p = os.path.join(OUT, f"{clip}_{model}_GT_deployed_{blab}_f{fi:03d}.png")
        cv2.imwrite(p, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(p, flush=True)
print("DONE")
