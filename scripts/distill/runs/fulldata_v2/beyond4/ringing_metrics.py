"""Objective companions to the visual read: halo/overshoot, flat-region HF, edge HF, stripe energy.

Regions are defined on the GT (so they are identical for every config compared) using the same
crop math as score_clip.py.  Reported on the SCORE_STEP-subsampled frames of the lossless videos.

usage: ringing_metrics.py <clip> <dy> <dx> <label>=<sbs video> ...
"""
import os, sys
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
STEP = int(os.environ.get("SCORE_STEP", "4"))
clip, dy, dx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
specs = [s.split("=", 1) for s in sys.argv[4:]]

def load(p):
    vr = VideoReader(p, ctx=cpu(0))
    return vr.get_batch(list(range(0, len(vr), STEP))).asnumpy().astype(np.float32) / 255.0

tile = load(f"video_data/train/{clip}_train.mp4")
H, W = tile.shape[1] // 2, tile.shape[2] // 2
vids = {lab: load(p) for lab, p in specs}
first = next(iter(vids.values()))
h, w = first.shape[1], first.shape[2] // 2
t0, l0 = (H - h) // 2 + dy, (W - w) // 2 + dx
n = min(min(len(v) for v in vids.values()), len(tile))
gt = tile[:n, t0:t0 + h, l0:l0 + w].mean(axis=3)
print(f"[{clip}] n={n} crop=({t0},{l0}) {h}x{w}")

gmag = np.zeros_like(gt)
gmag[:, :, :-1] += np.abs(np.diff(gt, axis=2)); gmag[:, :-1, :] += np.abs(np.diff(gt, axis=1))
hi = gmag >= np.quantile(gmag, 0.90)
lo = gmag <= np.quantile(gmag, 0.50)
# GT local min/max over a 3x3 neighbourhood, for the halo (out-of-range) test
def nb(a, f):
    s = [a]
    for ax in (1, 2):
        for k in (-1, 1):
            s.append(np.roll(a, k, axis=ax))
    return f(np.stack(s), axis=0)
gmin, gmax = nb(gt, np.min), nb(gt, np.max)
lap = lambda a: np.abs(4 * a[:, 1:-1, 1:-1] - a[:, :-2, 1:-1] - a[:, 2:, 1:-1] - a[:, 1:-1, :-2] - a[:, 1:-1, 2:])
gl = lap(gt); hi_i, lo_i = hi[:, 1:-1, 1:-1], lo[:, 1:-1, 1:-1]
print(f"{'label':14s} {'haloFrac%':>10s} {'haloMean':>9s} {'flatHF':>9s} {'flatHF/GT':>10s} {'edgeHF':>9s} {'edgeHF/GT':>10s} {'stripeE':>9s} {'stripeE/GT':>11s}")
def rep(lab, y):
    over = np.maximum(y - gmax, 0) + np.maximum(gmin - y, 0)
    hf = lap(y)
    flat = hf[lo_i].mean(); edge = hf[hi_i].mean()
    # stripe energy: column-direction HF inside GT-flat regions (splatting stripes are vertical)
    col = np.abs(np.diff(y, axis=2))
    m = lo[:, :, :-1]
    stripe = col[m].mean()
    return over, hf, flat, edge, stripe
_, _, gf, ge, gs = rep("GT", gt)
print(f"{'GT':14s} {0.0:10.3f} {0.0:9.5f} {gf:9.5f} {1.0:10.3f} {ge:9.5f} {1.0:10.3f} {gs:9.5f} {1.0:11.3f}")
for lab, _p in specs:
    y = vids[lab][:n, :, w:, :].mean(axis=3)
    over, hf, flat, edge, stripe = rep(lab, y)
    hmask = over[hi] > 0.04
    print(f"{lab:14s} {100*hmask.mean():10.3f} {over[hi].mean():9.5f} {flat:9.5f} {flat/gf:10.3f} "
          f"{edge:9.5f} {edge/ge:10.3f} {stripe:9.5f} {stripe/gs:11.3f}")
print("  haloFrac% = % of GT-edge pixels whose value falls outside the GT's own 3x3 [min,max] by >0.04 (halo/ringing)")
print("  flatHF    = mean |Laplacian| inside GT-flat regions (noise / amplified splatting texture)")
print("  edgeHF    = mean |Laplacian| inside the GT's top-decile gradient regions (detail at edges)")
print("  stripeE   = mean |horizontal difference| inside GT-flat regions (splatting-stripe energy)")
