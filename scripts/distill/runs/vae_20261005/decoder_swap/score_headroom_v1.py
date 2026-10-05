#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- STEP 3 HEADROOM scores for one dev clip (PREREG.txt section 2).  GPU (LPIPS/DISTS/NR).

Reference = the encoded real-right-eye frames themselves (decoder_ft gt_dev/<clip>_TR.npy; registration-free).
Frames 0,4,8,... (score_clip_ll grid).  Per decoder row (headroom_dev/<clip>__<dec>.mkv):
  lpips      LPIPS-Alex, batches of 4, sum / n (roundtrip_v1.py / score_clip_ll.py accumulation)
  psnr       mean over frames of per-frame PSNR (roundtrip_v1.py definition)
  dists      piq.DISTS, mean over frames
  decompose  reviewlib.decompose with regions from the GT frames (flatHF, edgeHF, stripeE, haloFrac, haloMean); /GT ratios
  niqe musiq pyiqa, per frame (full 576x1024 RGB), mean   (also for GT itself)
  sharp      mean |horizontal diff| over RGB in [0,1] (score_clip_ll statistic)
HR0 gate: the stock row's md5 in roundtrip_v1.py's layout (uint8 [n,3,H,W], all frames) vs decoder_ft ROUNDTRIP_DEV_main_v1.json,
          and its LPIPS-Alex vs that file's (must agree within 1e-4 if the md5 differs).
HR-SEED (if <clip>__cd@1.npy / cd@2.npy exist): mean |a-b| (RGB, [0,1]) overall, on GT-edge (top decile gradient) and
          GT-flat (bottom half) pixels for cd vs cd@1, cd vs cd@2, cd@1 vs cd@2, and for context cd vs stock, ftmse vs stock.
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python score_headroom_v1.py <out_dir> <clip>
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/review_20261001"))
import reviewlib as RL  # noqa: E402
import lpips  # noqa: E402
import piq  # noqa: E402
import pyiqa  # noqa: E402

HR = "/mnt/ssd_data/vae_20261005/decoder_swap/headroom_dev"
RT = "scripts/distill/runs/deep_20261004/decoder_ft/ROUNDTRIP_DEV_main_v1.json"
ROWS = ["stock", "stock32", "ftmse", "ftema", "cd", "sd15"]
OUTD, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}.json"
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
T0 = time.time()
dev = torch.device("cuda")


def log(*a):
    print(f"[hr {CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


X = np.load(f"/mnt/ssd_data/deep_20261004/decoder_ft/gt_dev/{CLIP}_TR.npy", mmap_mode="r")
n = X.shape[0]
fr = list(range(0, n, 4))
GT = np.ascontiguousarray(X[fr])
greg = RL.gray(GT)
reg = RL.regions(greg)
net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
dists = piq.DISTS(reduction="none").to(dev)
niqe = pyiqa.create_metric("niqe", device=dev)
musiq = pyiqa.create_metric("musiq", device=dev)


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


@torch.no_grad()
def nr(x):
    a, b = [], []
    for f in range(len(x)):
        t = t01(x[f:f + 1]).to(dev)
        a.append(float(niqe(t)))
        b.append(float(musiq(t)))
    return a, b


def sharp(x):
    return float(np.abs(x[:, :, 1:].astype(np.float32) / 255. - x[:, :, :-1].astype(np.float32) / 255.).mean())


G = t01(GT)
gn, gm = nr(GT)
res = dict(clip=CLIP, n=n, frames=fr, GT=dict(niqe=float(np.mean(gn)), musiq=float(np.mean(gm)),
                                             decompose=RL.decompose(reg, greg), sharp=sharp(GT)), rows={})
rt = json.load(open(RT))["per_clip"].get(CLIP, {}).get("stock")
cache = {}
with torch.no_grad():
    for r in ROWS:
        p = f"{HR}/{CLIP}__{r}.mkv"
        if not os.path.exists(p):
            log(f"missing {p}")
            continue
        vr = VideoReader(p, ctx=cpu(0))
        assert len(vr) == n, (p, len(vr), n)
        if r == "stock":
            allf = vr.get_batch(list(range(n))).asnumpy()
            md5_cf = hashlib.md5(np.ascontiguousarray(allf.transpose(0, 3, 1, 2)).tobytes()).hexdigest()
            Y = np.ascontiguousarray(allf[fr])
            del allf
        else:
            Y = vr.get_batch(fr).asnumpy()
        cache[r] = Y
        A = t01(Y)
        tot, dl = 0.0, []
        for s in range(0, len(fr), 4):
            tot += float(net(A[s:s + 4].to(dev) * 2 - 1, G[s:s + 4].to(dev) * 2 - 1).sum())
            dl += [float(v) for v in dists(A[s:s + 4].to(dev), G[s:s + 4].to(dev)).view(-1)]
        mse = [float(((A[j] - G[j]) ** 2).mean()) for j in range(len(fr))]
        nn_, mm_ = nr(Y)
        d = RL.decompose(reg, RL.gray(Y))
        e = dict(path=p, lpips=tot / len(fr), psnr=float(np.mean([10 * math.log10(1 / max(m, 1e-12)) for m in mse])),
                 dists=float(np.mean(dl)), niqe=float(np.mean(nn_)), musiq=float(np.mean(mm_)), decompose=d,
                 ratio_to_GT={k: (d[k] / res["GT"]["decompose"][k] if res["GT"]["decompose"][k] > 0 else float("nan"))
                              for k in ("flatHF", "edgeHF", "stripeE")},   # NaN when the GT region value is exactly 0 (0082)
                 sharp=sharp(Y), decode=json.load(open(f"{HR}/{CLIP}__{r}.json")))
        if r == "stock":
            e["HR0"] = dict(md5_channel_first=md5_cf, decoder_ft_md5=rt["md5"] if rt else None,
                            md5_equal=bool(rt and rt["md5"] == md5_cf), decoder_ft_lpips=rt["lpips"] if rt else None,
                            lpips_diff=(e["lpips"] - rt["lpips"]) if rt else None)
            log(f"HR0: md5 {'EQUAL' if e['HR0']['md5_equal'] else 'DIFFERENT'} to decoder_ft roundtrip; LPIPS diff "
                f"{e['HR0']['lpips_diff']:+.2e}")
        res["rows"][r] = e
        log(f"{r:8s} LPIPS {e['lpips']:.4f} PSNR {e['psnr']:.3f} DISTS {e['dists']:.4f} NIQE {e['niqe']:.3f} MUSIQ {e['musiq']:.2f} "
            f"edgeHF/GT {e['ratio_to_GT']['edgeHF']:.3f} flatHF/GT {e['ratio_to_GT']['flatHF']:.3f} "
            f"stripeE/GT {e['ratio_to_GT']['stripeE']:.3f} halo% {d['haloFrac']:.3f}")
# HR-SEED
hi, lo = reg["hi"], reg["lo"]


def mad(a, b):
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1) / 255.
    return dict(all=float(d.mean()), edge=float(d[hi].mean()), flat=float(d[lo].mean()))


seeds = {s: f"{HR}/{CLIP}__cd@{s}.npy" for s in (1, 2)}
if all(os.path.exists(p) for p in seeds.values()) and "cd" in cache:
    c1, c2 = np.load(seeds[1]), np.load(seeds[2])
    assert c1.shape == cache["cd"].shape
    res["HR_SEED"] = {"cd vs cd@1": mad(cache["cd"], c1), "cd vs cd@2": mad(cache["cd"], c2), "cd@1 vs cd@2": mad(c1, c2),
                      "cd vs stock": mad(cache["cd"], cache["stock"]),
                      "ftmse vs stock": mad(cache["ftmse"], cache["stock"]) if "ftmse" in cache else None,
                      "cd vs GT": mad(cache["cd"], GT), "stock vs GT": mad(cache["stock"], GT)}
    for k, v in res["HR_SEED"].items():
        if v:
            log(f"HR-SEED {k:16s} mean|d| all {v['all']:.5f} edge {v['edge']:.5f} flat {v['flat']:.5f}")
res["seconds"] = time.time() - T0
json.dump(res, open(OJ, "w"), indent=1)
log(f"wrote {OJ}")
print("HEADROOM_SCORE_DONE", CLIP, flush=True)
