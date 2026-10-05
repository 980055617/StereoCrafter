#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- AUXILIARY metrics per clip for every label of a rows json (GPU).  PREREG.txt.

Reads the registered scorer's own output json (score_registered_df_v1.py: frames, window, per-frame REG_FRAME shifts
reg.smooth_ddy/smooth_ddx) so the GT crop is IDENTICAL to the one REG_FRAME LPIPS-Alex used:
  GT_REG[j] = TR quadrant of video_data/train/<clip>_train.mp4 at frame frames[j], rows t0+ddy.., cols W+l0+ddx..
Per label (right half of the SBS render at frames[j], j = 0..n-1, frames = 0,4,8,... as score_clip_ll.py):
  dists      piq.DISTS vs GT_REG (lower better)                      [held-out perceptual metric]
  lpips_vgg  LPIPS-VGG vs GT_REG (lower better)                      [the decoder's TRAINING loss family: diagnostic only]
  psnr_reg   PSNR vs GT_REG (dB; = the registered scorer's REG_FRAME rPSNR definition up to frame-vs-pooled averaging)
  niqe, musiq  pyiqa no-reference (blur_diag calibration: both monotone in blur at origin's level; NIQE lower better)
  decompose  reviewlib.decompose on gray frames, regions from GT_REG (flatHF / edgeHF / stripeE / haloFrac)
  sharp      score_clip_ll.py statistic (mean |horizontal diff| over RGB in [0,1])
plus the same NR / decompose values for GT_REG itself (reference level).
env: PYTHONPATH must include /mnt/ssd_data/deep_20261004/blur_diag/pylib (pyiqa) and /mnt/ssd_data/deep_20261004/skeptic/pylib
     (piq); TORCH_HOME /mnt/ssd_data/deep_20261004/decoder_ft/torch_home; HF_HOME .../blur_diag/hf_home, offline.
usage: CUDA_VISIBLE_DEVICES=1 flock /tmp/claude-gpu1.lock python score_aux_v1.py <out_dir> <rows.json> <regscore_dir> <clip>
"""
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

TH, TW = 576, 1024
OUTD, ROWS, REGD, CLIP = sys.argv[1:5]
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}.json"
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
T0 = time.time()
dev = torch.device("cuda")


def log(*a):
    print(f"[aux {CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


rows = json.load(open(ROWS))
labels = rows["labels"]
cells = rows["cells"][CLIP]
R = json.load(open(f"{REGD}/{CLIP}.json"))
frames = R["frames"]
t0, l0 = R["window"]
H, W = R["quadrant"]
sdy, sdx = R["reg"]["smooth_ddy"], R["reg"]["smooth_ddx"]
vt = VideoReader(f"video_data/train/{CLIP}_train.mp4", ctx=cpu(0))
tiles = vt.get_batch(frames).asnumpy()
GT = np.stack([tiles[j, t0 + sdy[fi]:t0 + sdy[fi] + TH, W + l0 + sdx[fi]:W + l0 + sdx[fi] + TW]
               for j, fi in enumerate(frames)])
del tiles, vt
assert GT.shape == (len(frames), TH, TW, 3), GT.shape


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


def psnr(a, b):
    e = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(e, 1e-12))


net_v = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
dists = piq.DISTS(reduction="none").to(dev)
niqe = pyiqa.create_metric("niqe", device=dev)
musiq = pyiqa.create_metric("musiq", device=dev)
greg = RL.gray(GT)
reg = RL.regions(greg)


@torch.no_grad()
def nr(x):
    n_, m_ = [], []
    for f in range(len(x)):
        t = t01(x[f:f + 1]).to(dev)
        n_.append(float(niqe(t)))
        m_.append(float(musiq(t)))
    return n_, m_


res = dict(clip=CLIP, frames=frames, rows_json=ROWS, regscore=f"{REGD}/{CLIP}.json", labels={})
gn, gm = nr(GT)
res["GT"] = dict(niqe=float(np.mean(gn)), musiq=float(np.mean(gm)), decompose=RL.decompose(reg, greg),
                 sharp=float(np.abs(GT[:, :, 1:].astype(np.float32) / 255. - GT[:, :, :-1].astype(np.float32) / 255.).mean()))
log(f"GT niqe {res['GT']['niqe']:.3f} musiq {res['GT']['musiq']:.2f}")
G = t01(GT)
with torch.no_grad():
    for lab in labels:
        p = cells[lab]["path"]
        vr = VideoReader(p, ctx=cpu(0))
        n = min(len(frames), len(range(0, len(vr), 4)))
        assert n == len(frames), (lab, n, len(frames))
        v = vr.get_batch([4 * j for j in range(n)]).asnumpy()
        assert [4 * j for j in range(n)] == frames, "frame grid mismatch"
        Rr = np.ascontiguousarray(v[:, :, v.shape[2] // 2:])
        A = t01(Rr)
        dl, vl = [], []
        for i in range(0, n, 4):
            a, g = A[i:i + 4].to(dev), G[i:i + 4].to(dev)
            dl += [float(x) for x in dists(a, g).view(-1)]
            vl += [float(x) for x in net_v(a * 2 - 1, g * 2 - 1).view(-1)]
        nn_, mm_ = nr(Rr)
        e = dict(path=p, dists=float(np.mean(dl)), lpips_vgg=float(np.mean(vl)),
                 psnr_reg=float(np.mean([psnr(Rr[j], GT[j]) for j in range(n)])),
                 niqe=float(np.mean(nn_)), musiq=float(np.mean(mm_)), decompose=RL.decompose(reg, RL.gray(Rr)),
                 sharp=float(np.abs(Rr[:, :, 1:].astype(np.float32) / 255. - Rr[:, :, :-1].astype(np.float32) / 255.).mean()),
                 dists_frames=dl, lpips_vgg_frames=vl, niqe_frames=nn_, musiq_frames=mm_)
        res["labels"][lab] = e
        d = e["decompose"]
        log(f"{lab:40s} dists {e['dists']:.4f} vgg {e['lpips_vgg']:.4f} psnrR {e['psnr_reg']:.3f} niqe {e['niqe']:.3f} "
            f"musiq {e['musiq']:.2f} flatHF {d['flatHF']:.5f} edgeHF {d['edgeHF']:.5f} stripeE {d['stripeE']:.5f} "
            f"sharp {e['sharp']:.4f}")
res["seconds"] = time.time() - T0
json.dump(res, open(OJ, "w"))
log(f"wrote {OJ}")
print("AUX_DONE", CLIP, flush=True)
