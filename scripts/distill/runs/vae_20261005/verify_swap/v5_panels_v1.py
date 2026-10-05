#!/usr/bin/env python
"""vae_20261005 / verify_swap -- V5 visual panels (PREREG.txt).  CPU only.  Frame 76, deliverable latents (+ origin stock tile).
Crop positions (384x384) by decoder_swap's rule from the REGISTERED GT + hole mask only: 0268 = decoder_swap crops.json positions,
0184 computed here by the same rule (E = highest GT top-decile-gradient fraction, stride 32; F = highest GT flat fraction among
windows with >= 5 % hole, else overall).  All panels <= 1160 px wide, <= ~0.96 MP, no resampling except the explicit 2x
nearest-neighbour zooms.  Writes outputs/vae_20261005/verify_swap/panels_v5/:
  <clip>_<E|F>_unet.png      GT reg | origin stock | deliverable stock  /  deliverable ftmse | ftema | cd          (100 %)
  <clip>_<E|F>_unetdiff.png  |ftmse - stock| x8 | |ftema - stock| x8 | |cd - stock| x8  (deliverable)          (100 %)
  <clip>_<E|F>_unetzoom.png  centre 192x192 at 2x: GT reg | deliverable stock | deliverable ftmse
  <clip>_<E|F>_hr.png        HEADROOM (real-right-eye latents): GT exact | stock | ftmse  /  ftema | cd | |ftmse - stock| x8
  <clip>_<E|F>_hrzoom.png    centre 192x192 at 2x: GT exact | stock | ftmse
usage: python v5_panels_v1.py <out_dir> <clip,clip>
"""
import json
import os
import sys

import cv2
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
OUTD, CLIPS = sys.argv[1], sys.argv[2].split(",")
os.makedirs(OUTD, exist_ok=True)
CJ = f"{OUTD}/crops_v5.json"
assert not os.path.exists(CJ), CJ
FR, CS, ST, TH, TW = 76, 384, 32, 576, 1024
RED = "/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev"
HRD = "/mnt/ssd_data/vae_20261005/decoder_swap/headroom_dev"
LANE_CROPS = json.load(open(f"{REPO}/outputs/vae_20261005/decoder_swap/panels_dev_v1/crops.json"))


def right(path, fr=FR):
    a = VideoReader(path, ctx=cpu(0))[fr].asnumpy()
    return np.ascontiguousarray(a[:, a.shape[1] // 2:]) if a.shape[1] == 2048 else np.ascontiguousarray(a)


def lab(img, text):
    s = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(s, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return np.concatenate([s, img], 0)


def grid(rows):
    sepv = lambda h: np.full((h, 4, 3), 255, np.uint8)
    out = []
    for r in rows:
        line = r[0]
        for t in r[1:]:
            line = np.concatenate([line, sepv(line.shape[0]), t], 1)
        out.append(line)
    body = out[0]
    for o in out[1:]:
        body = np.concatenate([body, np.full((4, body.shape[1], 3), 255, np.uint8), o], 0)
    return body


def diff8(a, b):
    return np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 8, 0, 255).astype(np.uint8)


def z2(a):
    c = a[96:288, 96:288]
    return np.repeat(np.repeat(c, 2, 0), 2, 1)


def save(name, img):
    cv2.imwrite(f"{OUTD}/{name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return [int(img.shape[1]), int(img.shape[0])]


out = dict(frame=FR, crops={}, sizes={})
for clip in CLIPS:
    R = json.load(open(f"{REPO}/outputs/vae_20261005/decoder_swap/score_reg_dev_v1/{clip}.json"))
    t0, l0 = R["window"]
    H, W = R["quadrant"]
    ddy, ddx = int(R["reg"]["smooth_ddy"][FR]), int(R["reg"]["smooth_ddx"][FR])
    tile = VideoReader(f"video_data/train/{clip}_train.mp4", ctx=cpu(0))[FR].asnumpy()
    GT = np.ascontiguousarray(tile[t0 + ddy:t0 + ddy + TH, W + l0 + ddx:W + l0 + ddx + TW])
    s = VideoReader(f"video_data/splatting/{clip}_splatting_results.mp4", ctx=cpu(0))[FR].asnumpy()
    Hs, Ws = s.shape[0] // 2, s.shape[1] // 2
    st0, sl0 = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
    hole = s[Hs + st0:Hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) > 127.5
    if clip in LANE_CROPS["crops"]:
        E, F = tuple(LANE_CROPS["crops"][clip]["E"]["yx"]), tuple(LANE_CROPS["crops"][clip]["F"]["yx"])
        src = "decoder_swap crops.json"
    else:
        g = GT.astype(np.float32).mean(-1) / 255.
        gm = np.zeros_like(g)
        gm[:, :-1] += np.abs(np.diff(g, axis=1))
        gm[:-1, :] += np.abs(np.diff(g, axis=0))
        hi, lo = gm >= np.quantile(gm, 0.90), gm <= np.quantile(gm, 0.50)
        cand = [(y, x) for y in range(0, TH - CS + 1, ST) for x in range(0, TW - CS + 1, ST)]
        fr_ = lambda m, y, x: float(m[y:y + CS, x:x + CS].mean())
        E = max(cand, key=lambda p: fr_(hi, *p))
        wh = [p for p in cand if fr_(hole, *p) >= 0.05]
        F = max(wh or cand, key=lambda p: fr_(lo, *p))
        src = "computed here (same rule)" + ("" if wh else "; no window with >= 5 % hole -> flat overall")
    out["crops"][clip] = dict(E=list(E), F=list(F), reg_shift=[ddy, ddx], source=src,
                              hole_frac={"E": float(hole[E[0]:E[0] + CS, E[1]:E[1] + CS].mean()),
                                         "F": float(hole[F[0]:F[0] + CS, F[1]:F[1] + CS].mean())})
    U = {d: right(f"{RED}/{clip}_deliv_cap__{d}/{clip}_inpainting_results_sbs.mkv") for d in ("stock", "ftmse", "ftema", "cd")}
    U["origin_stock"] = right(f"{RED}/{clip}_origin_cap__stock/{clip}_inpainting_results_sbs.mkv")
    GTX = np.load(f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{clip}_TR.npy", mmap_mode="r")[FR]
    HRr = {d: right(f"{HRD}/{clip}__{d}.mkv") for d in ("stock", "ftmse", "ftema", "cd")}
    for nm, (y, x) in (("E", E), ("F", F)):
        c = lambda a: np.ascontiguousarray(a[y:y + CS, x:x + CS])
        out["sizes"][f"{clip}_{nm}_unet"] = save(f"{clip}_{nm}_unet.png", grid([
            [lab(c(GT), f"GT reg ({ddy:+d},{ddx:+d})"), lab(c(U["origin_stock"]), "origin + stock dec"),
             lab(c(U["stock"]), "deliverable + stock dec")],
            [lab(c(U["ftmse"]), "deliverable + ft-mse"), lab(c(U["ftema"]), "deliverable + ft-ema"),
             lab(c(U["cd"]), "deliverable + consistency")]]))
        out["sizes"][f"{clip}_{nm}_unetdiff"] = save(f"{clip}_{nm}_unetdiff.png", grid([[
            lab(diff8(c(U["ftmse"]), c(U["stock"])), "|ft-mse - stock| x8"),
            lab(diff8(c(U["ftema"]), c(U["stock"])), "|ft-ema - stock| x8"),
            lab(diff8(c(U["cd"]), c(U["stock"])), "|consistency - stock| x8")]]))
        out["sizes"][f"{clip}_{nm}_unetzoom"] = save(f"{clip}_{nm}_unetzoom.png", grid([[
            lab(z2(c(GT)), "GT reg, 2x"), lab(z2(c(U["stock"])), "deliverable + stock, 2x"),
            lab(z2(c(U["ftmse"])), "deliverable + ft-mse, 2x")]]))
        out["sizes"][f"{clip}_{nm}_hr"] = save(f"{clip}_{nm}_hr.png", grid([
            [lab(c(GTX), "GT exact (encoded frame)"), lab(c(HRr["stock"]), "round trip: stock dec"),
             lab(c(HRr["ftmse"]), "round trip: ft-mse")],
            [lab(c(HRr["ftema"]), "round trip: ft-ema"), lab(c(HRr["cd"]), "round trip: consistency"),
             lab(diff8(c(HRr["ftmse"]), c(HRr["stock"])), "|ft-mse - stock| x8")]]))
        out["sizes"][f"{clip}_{nm}_hrzoom"] = save(f"{clip}_{nm}_hrzoom.png", grid([[
            lab(z2(c(GTX)), "GT exact, 2x"), lab(z2(c(HRr["stock"])), "round trip stock, 2x"),
            lab(z2(c(HRr["ftmse"])), "round trip ft-mse, 2x")]]))
    print(clip, out["crops"][clip], flush=True)
json.dump(out, open(CJ, "w"), indent=1)
print("PANELS_V5_DONE", flush=True)
