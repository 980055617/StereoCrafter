"""more_20261004 / stripes lane: whole-window artefact/detail decomposition with DISPARITY-REGISTERED GT regions,
re-using scripts/distill/runs/review_20261001/reviewlib.py (operators imported, not copied: gray, regions,
decompose, grab, grab_many, psnr_u8, tile_quadrants) and the global-registration + whole-frame procedure of
scripts/distill/runs/review_20261001/build_review.py (copied verbatim below, marked).

The only extension: reviewlib.OFFSETS lacks 0052, so window()/splat_mask() are re-implemented here with an
extended offset table (every offset reviewlib has is asserted identical; 0052's (-12,-12) is the scorer's own
dy,dx from the ROW lines of scripts/distill/runs/finalcheck_20261004/speed/SCORES_STEP1.txt).
For the clips the review covered, the recomputed registration shift is asserted equal to the stored one in
outputs/review_20261001/metrics.json (and the frame choice too).

Extra (NOT in the review): decompose_input() applies the same decomposition to the model's own warped
right-eye INPUT (splatting video BR quadrant, deployed window) before/after a crack fill, and splits the
flat-region stripe energy by distance to the nearest hole pixel.
"""
import json
import os
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, f"{REPO}/scripts/distill/runs/review_20261001")
import reviewlib as R  # noqa: E402  (chdir REPO inside)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crackfill as CF  # noqa: E402

OFFSETS = dict(R.OFFSETS)
OFFSETS["0052"] = (-12, -12)
for _k, _v in R.OFFSETS.items():
    assert OFFSETS[_k] == _v
WHOLE_STEP = 8
REVIEW_METRICS = f"{REPO}/outputs/review_20261001/metrics.json"


def window(clip, h=R.TH, w=R.TW):
    tl = R.grab(f"video_data/train/{clip}_train.mp4", 0)
    H, W = tl.shape[0] // 2, tl.shape[1] // 2
    dy, dx = OFFSETS[clip]
    return (H - h) // 2 + dy, (W - w) // 2 + dx, H, W


def splat_quads(clip, idx):
    """(mask float [TH,TW] in [0,1] = RGB mean / 255 like read_and_prepare_video, warped uint8 [TH,TW,3]) in render coords."""
    f = R.grab(f"video_data/splatting/{clip}_splatting_results.mp4", idx)
    hh, ww = f.shape[0] // 2, f.shape[1] // 2
    h128, w128 = hh // 128 * 128, ww // 128 * 128
    top, left = (h128 - R.TH) // 2, (w128 - R.TW) // 2
    t0, l0, _, _ = window(clip)
    assert (top, left) == (t0, l0), f"{clip}: mask window {(top, left)} != GT window {(t0, l0)}"
    m = f[hh + top:hh + top + R.TH, left:left + R.TW].astype(np.float32).mean(-1) / 255.0
    wp = f[hh + top:hh + top + R.TH, ww + left:ww + left + R.TW]
    return m, wp


def splat_mask(clip, idx):
    """== reviewlib.splat_mask (channel-0 > 127) with the extended offset table."""
    f = R.grab(f"video_data/splatting/{clip}_splatting_results.mp4", idx)
    hh, ww = f.shape[0] // 2, f.shape[1] // 2
    h128, w128 = hh // 128 * 128, ww // 128 * 128
    top, left = (h128 - R.TH) // 2, (w128 - R.TW) // 2
    t0, l0, _, _ = window(clip)
    assert (top, left) == (t0, l0)
    return f[hh + top:hh + top + R.TH, left:left + R.TW, 0] > 127


def geometry(clip, render_paths=()):
    """frame choice + global GT registration, build_review.py code (verbatim apart from names)."""
    t0, l0, H, W = window(clip)
    ns = {"train": R.nframes(f"video_data/train/{clip}_train.mp4"),
          "splat": R.nframes(f"video_data/splatting/{clip}_splatting_results.mp4")}
    for p in render_paths:
        ns[p] = R.nframes(p)
    nvalid = min(ns.values())
    # ---- frame choice: densest disocclusion inside the deployed window, VALID for every source  [build_review.py]
    cand = [i for i in range(0, nvalid, 4) if 8 <= i <= nvalid - 9]
    mfr = np.array([splat_mask(clip, i).mean() for i in cand])
    fidx = int(cand[int(mfr.argmax())])
    TL, TR, BL, BR, _H, _W = R.tile_quadrants(clip, fidx)
    mask = splat_mask(clip, fidx)
    # ---- global registration of the real right eye over the whole deployed window ----  [build_review.py]
    tgt = BR[t0:t0 + R.TH, l0:l0 + R.TW]
    valid = ~mask
    gp0 = R.psnr_u8(TR[t0:t0 + R.TH, l0:l0 + R.TW], tgt, valid)
    best = (-1e9, 0, 0)
    for st in (2, 1):
        cy, cx = best[1], best[2]
        yr = range(-10, 11, 2) if st == 2 else range(cy - 2, cy + 3)
        xr = range(-140, 61, 2) if st == 2 else range(cx - 3, cx + 4)
        for a in yr:
            for b in xr:
                tt, ll = t0 + a, l0 + b
                if tt < 0 or ll < 0 or tt + R.TH > H or ll + R.TW > W:
                    continue
                pv = R.psnr_u8(TR[tt:tt + R.TH, ll:ll + R.TW], tgt, valid)
                if pv > best[0]:
                    best = (pv, a, b)
    gp, gddy, gddx = best
    geo = dict(clip=clip, t0=t0, l0=l0, H=H, W=W, nvalid=nvalid, nframes=ns, frame=fidx,
               gtShift=[gddy, gddx], gtWinPSNR_scorer=gp0, gtWinPSNR_best=gp,
               maskFracMean=float(mfr.mean()))
    if os.path.exists(REVIEW_METRICS):
        rv = json.load(open(REVIEW_METRICS))
        if clip in rv:
            assert rv[clip]["frame"] == fidx, (clip, rv[clip]["frame"], fidx)
            assert list(rv[clip]["gtGlobalShift"]) == [gddy, gddx], (clip, rv[clip]["gtGlobalShift"], [gddy, gddx])
            geo["review_check"] = "frame and gtGlobalShift identical to outputs/review_20261001/metrics.json"
    return geo


def gt_stack(clip, geo, frames):
    """registered GT right-eye window, channel-mean float [n,TH,TW]  (build_review.tr_crops semantics)."""
    t0, l0 = geo["t0"], geo["l0"]
    a, b = geo["gtShift"]
    out = []
    p = f"video_data/train/{clip}_train.mp4"
    for i in range(0, len(frames), 3):
        arr = R.grab_many(p, frames[i:i + 3])
        H, W = arr.shape[1] // 2, arr.shape[2] // 2
        TRb = arr[:, :H, W:2 * W]
        out.append(R.gray(TRb[:, t0 + a:t0 + a + R.TH, l0 + b:l0 + b + R.TW]))
        del arr, TRb
    return np.concatenate(out, 0)


def render_stack(path, frames):
    return np.stack([R.gray(R.grab(path, f)[:, R.TW:]) for f in frames])


def frames_for(geo):
    return list(range(0, geo["nvalid"], WHOLE_STEP))


def decompose_input(clip, reg, frames, mode, maxw=3, near_px=2):
    """Same decomposition on the model's own warped right-eye input (deployed window), after the crack fill
    `mode` (none|rowlin|telea).  The fill runs on the window only here (no margin) -- diagnostic only.
    Also: stripeE split into GT-flat pixels within `near_px` of a hole pixel (mask>=0.5) vs farther."""
    import torch
    ys = []
    near = []
    for f in frames:
        m, wp = splat_quads(clip, f)
        w = torch.from_numpy(wp.astype(np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
        mm = torch.from_numpy(m)[None, None].contiguous()
        CF.process(w, mm, 0, 0, R.TH, R.TW, mode, "keep", maxw=maxw, margin=0)
        ys.append(w[0].permute(1, 2, 0).numpy().mean(-1))
        hb = m >= 0.5
        d = hb.copy()
        for _ in range(near_px):  # 4-neighbour dilation
            d = d | np.roll(d, 1, 0) | np.roll(d, -1, 0) | np.roll(d, 1, 1) | np.roll(d, -1, 1)
        near.append(d)
    y = np.stack(ys).astype(np.float32)
    near = np.stack(near)
    dd = R.decompose(reg, y)
    col = np.abs(np.diff(y, axis=2))
    lo = reg["lo"][:, :, :-1]
    nr = near[:, :, :-1] | near[:, :, 1:]
    dd["stripeE_nearHole"] = float(col[lo & nr].mean()) if (lo & nr).any() else float("nan")
    dd["stripeE_farHole"] = float(col[lo & ~nr].mean())
    dd["flatFrac_nearHole"] = float((lo & nr).sum() / lo.sum())
    return dd, near


def split_stripe(reg, y, near):
    col = np.abs(np.diff(y, axis=2))
    lo = reg["lo"][:, :, :-1]
    nr = near[:, :, :-1] | near[:, :, 1:]
    return (float(col[lo & nr].mean()) if (lo & nr).any() else float("nan"), float(col[lo & ~nr].mean()))
