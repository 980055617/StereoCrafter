#!/usr/bin/env python
"""blur_diag (deep_20261004) -- CPU-only extraction cache.  Definitions: PREREG.txt (same dir).

Per clip writes /mnt/ssd_data/deep_20261004/blur_diag/cache_r2/<clip>/ :
  GTreg.npy   (n_s,576,1024,3) uint8  real right eye TR at the REG_FRAME per-frame shift (eval_robustness JSON),
                                      frames >= n_g padded with the last GT frame (VAE window grid only, never scored)
  BR.npy      (n_s,576,1024,3) uint8  model input = splatting BR quadrant at the deployed window
  HOLE.npy    (n_s,576,1024)   bool   splatting BL quadrant channel mean > 127.5 at the window
  GTunreg_sc.npy (38,576,1024,3) uint8 TR at zero shift, scored frames (UNREG gate)
  meta.json   shifts, frames, geometry, G1 validity numbers, md5 of every array
Nothing tracked is modified; videos are only read.
usage: python extract_cache_r2.py <clip> [<clip> ...]
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
CACHE = "/mnt/ssd_data/deep_20261004/blur_diag/cache_r2"
TH, TW = 576, 1024
ER = "outputs/more_20261004/eval_robustness"


def er_json(clip):
    p = f"{ER}/score_v1_wide/{clip}.json" if clip == "0125" else f"{ER}/score_v1/{clip}.json"
    return p, json.load(open(p))


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def decode_all(vr, idxs, fn, chunk=8):
    for s in range(0, len(idxs), chunk):
        part = idxs[s:s + chunk]
        b = vr.get_batch(part).asnumpy()
        for k, fi in enumerate(part):
            fn(fi, b[k])
        del b


def run(clip):
    T0 = time.time()
    out = f"{CACHE}/{clip}"
    if os.path.exists(f"{out}/meta.json"):
        print(f"[{clip}] cache exists -> skip (never overwrite)")
        return
    os.makedirs(out, exist_ok=True)
    jp, J = er_json(clip)
    train = f"video_data/train/{clip}_train.mp4"
    splat = f"video_data/splatting/{clip}_splatting_results.mp4"
    real_train = os.path.realpath(train)
    assert "train_leftGT_broken" not in real_train, f"{clip}: train tile resolves into the broken dir: {real_train}"
    assert int(clip) < 310, f"{clip}: >= 0310 has no valid GT"
    vt = VideoReader(train, ctx=cpu(0))
    vs = VideoReader(splat, ctx=cpu(0))
    n_t, n_s = len(vt), len(vs)
    f0 = vt[0].asnumpy()
    H, W = f0.shape[0] // 2, f0.shape[1] // 2
    s0 = vs[0].asnumpy()
    Hs, Ws = s0.shape[0] // 2, s0.shape[1] // 2
    st0, sl0 = (Hs // 128 * 128 - TH) // 2, (Ws // 128 * 128 - TW) // 2
    t0, l0 = J["window"]
    assert (H, W) == (Hs, Ws) == tuple(J["quadrant"]), ((H, W), (Hs, Ws), J["quadrant"])
    assert (t0, l0) == (st0, sl0), ((t0, l0), (st0, sl0))
    assert n_t == J["n_train"] and n_s == J["n_splat"], (n_t, n_s, J["n_train"], J["n_splat"])
    n_g = min(n_t, n_s)
    assert n_g == J["n_g"]
    frames = J["frames"]
    sdy, sdx = J["reg"]["smooth_ddy"], J["reg"]["smooth_ddx"]
    assert len(sdy) == n_g and len(sdx) == n_g
    print(f"[{clip}] train {n_t} splat {n_s} n_g {n_g} quadrant {H}x{W} window ({t0},{l0}) json {jp}", flush=True)

    GTreg = np.empty((n_s, TH, TW, 3), np.uint8)
    GTun = {}
    TLw = {}
    fset = set(frames)

    def _train(fi, f):
        if fi < n_g:
            dy, dx = int(sdy[fi]), int(sdx[fi])
            y, x = t0 + dy, l0 + dx
            assert 0 <= y and y + TH <= H and 0 <= x and x + TW <= W, (fi, dy, dx)
            GTreg[fi] = f[y:y + TH, W + x:W + x + TW]
        if fi in fset:
            GTun[fi] = f[t0:t0 + TH, W + l0:W + l0 + TW].copy()
            TLw[fi] = f[t0:t0 + TH, l0:l0 + TW].copy()

    decode_all(vt, list(range(n_t)), _train)
    for fi in range(n_g, n_s):
        GTreg[fi] = GTreg[n_g - 1]
    BR = np.empty((n_s, TH, TW, 3), np.uint8)
    HOLE = np.empty((n_s, TH, TW), bool)

    def _splat(fi, f):
        BR[fi] = f[Hs + st0:Hs + st0 + TH, Ws + sl0:Ws + sl0 + TW]
        HOLE[fi] = f[Hs + st0:Hs + st0 + TH, sl0:sl0 + TW].astype(np.float32).mean(-1) > 127.5

    decode_all(vs, list(range(n_s)), _splat)

    # G1 validity: real right eye must differ from the left eye (old broken bundles had TR == TL)
    ps = []
    for fi in frames:
        d = (TLw[fi].astype(np.float64) - GTun[fi].astype(np.float64)) / 255.0
        ps.append(10 * math.log10(1.0 / max(float((d * d).mean()), 1e-12)))
    g1 = dict(train_realpath=real_train, psnr_TL_TR_unreg_mean=float(np.mean(ps)), psnr_TL_TR_unreg_max=float(np.max(ps)),
              pass_=bool(np.mean(ps) < 30.0 and "train_leftGT_broken" not in real_train))
    GTun_sc = np.stack([GTun[fi] for fi in frames])
    np.save(f"{out}/GTreg.npy", GTreg)
    np.save(f"{out}/BR.npy", BR)
    np.save(f"{out}/HOLE.npy", HOLE)
    np.save(f"{out}/GTunreg_sc.npy", GTun_sc)
    meta = dict(clip=clip, er_json=jp, n_t=n_t, n_s=n_s, n_g=n_g, quadrant=[H, W], window=[t0, l0], frames=frames,
                smooth_ddy=sdy, smooth_ddx=sdx, G1=g1,
                hole_frac_scored=float(HOLE[frames].mean()), hole_frac_all=float(HOLE.mean()),
                md5=dict(GTreg=md5(GTreg), BR=md5(BR), HOLE=md5(HOLE), GTunreg_sc=md5(GTun_sc)),
                seconds=time.time() - T0)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"[{clip}] G1 psnr(TL,TR) mean {g1['psnr_TL_TR_unreg_mean']:.2f} max {g1['psnr_TL_TR_unreg_max']:.2f} "
          f"-> {'PASS' if g1['pass_'] else 'FAIL'}; holes {meta['hole_frac_scored']:.4f}; {meta['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    for c in sys.argv[1:]:
        run(c)
