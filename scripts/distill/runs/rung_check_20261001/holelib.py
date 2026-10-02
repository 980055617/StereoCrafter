"""Helpers for the in-hole halo re-check (outputs/rung_check_20261001).

Reuses scripts/distill/runs/review_20261001/reviewlib.py for ALL geometry and ALL metric
operators (the review's CORRECTED right-eye geometry: tile[t0:t0+h, W+l0 : W+l0+w], plus the
per-clip horizontal disparity registration).  Two things are added here and nothing is changed:

  1. R.OFFSETS is extended from the review's 6 clips to all 12 TEST clips.  The values are
     harvested from the `ROW ... dy= dx=` lines of the already-published scoring logs, where
     they are identical across every config of a clip:
         (-12,-12) : 0042 0052 0125 0128 0141 0147
         (-28,  0) : 0170 0204 0225 0251 0259 0301
     R.window() / R.splat_mask() then work on all 12, and splat_mask's own assert
     (mask window == GT window) is the check that the extension is right.
  2. regions_restricted(): R.regions() with the GT gradient quantiles taken INSIDE a restriction
     mask and the hi/lo region sets intersected with it, so R.decompose() -- used verbatim --
     reports haloFrac / flatHF / edgeHF / stripeE over the disocclusion pixels only.

Panels: the review's four, plus the two conservative rungs rendered by this run.
"""
import glob
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, f"{REPO}/scripts/distill/runs/review_20261001")
import reviewlib as R  # noqa: E402  (does os.chdir(REPO) at import)

CLIPS12 = ["0042", "0052", "0125", "0128", "0141", "0147",
           "0170", "0204", "0225", "0251", "0259", "0301"]

R.OFFSETS.update({c: (-12, -12) for c in ("0042", "0052", "0125", "0128", "0141", "0147")})
R.OFFSETS.update({c: (-28, 0) for c in ("0170", "0204", "0225", "0251", "0259", "0301")})

# (label, render-dir tag).  The root is resolved by glob and asserted unique, so the 6 clips the
# review never touched need no hand-written root table.
PANELS6 = [
    ("origin (deployed)", "origin_ll"),
    ("shipped 5-slot Mamba", "mamba_ll"),
    ("mstudent2 step200", "mstudent2_step200_ll"),
    ("mstudent2 step400", "mstudent2_step400_ll"),
    ("step800 DELIVERABLE", "mstudent2_step800_deliv_ll"),
    ("origin+s25 (ceiling)", "s25_ll"),
]


def find_panel(clip, tag):
    g = sorted(glob.glob(f"{REPO}/outputs/*/clips/{clip}_{tag}/{clip}_inpainting_results_sbs.mkv"))
    if len(g) != 1:
        raise FileNotFoundError(f"{clip} {tag}: {len(g)} matches {g}")
    return os.path.relpath(g[0], REPO)


def panels(clip, tolerant=False):
    out = []
    for lab, tag in PANELS6:
        try:
            out.append((lab, find_panel(clip, tag)))
        except FileNotFoundError:
            if not tolerant:
                raise
    return out


def clear_readers():
    """reviewlib caches VideoReaders forever; drop them between clips."""
    R._VR.clear()


def nvalid(clip, paths):
    ns = [R.nframes(f"{REPO}/video_data/train/{clip}_train.mp4"),
          R.nframes(f"{REPO}/video_data/splatting/{clip}_splatting_results.mp4")]
    ns += [R.nframes(p) for p in paths]
    return min(ns)


# --------------------------------------------------------------------------------------------
# registration, build_review.py's recipe verbatim (BR = the model's own warped-right-eye INPUT,
# so the estimate is config-independent; disocclusion pixels excluded because BR is invalid there)
# --------------------------------------------------------------------------------------------
def global_shift_fast(clip, frame):
    """global_shift() on channel-mean GRAYSCALE instead of RGB -- same search grid, ~3x cheaper.

    Validated against GEOMETRY.txt's RGB-based shifts for the review's six clips.
    """
    t0, l0, H, W = R.window(clip)
    _, TR, _, BR, _, _ = R.tile_quadrants(clip, frame)
    mask = R.splat_mask(clip, frame)
    TRg = TR.astype(np.float32).mean(axis=2)
    tgt = BR[t0:t0 + R.TH, l0:l0 + R.TW].astype(np.float32).mean(axis=2)
    valid = (~mask).astype(np.float32)
    vs = float(valid.sum())

    def mse(tt, ll):
        d = TRg[tt:tt + R.TH, ll:ll + R.TW] - tgt
        return float((d * d * valid).sum() / vs)

    import math as _m

    def ps(tt, ll):
        return 10 * _m.log10(1.0 / max(mse(tt, ll) / (255.0 ** 2), 1e-12))

    p0 = ps(t0, l0)
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
                pv = ps(tt, ll)
                if pv > best[0]:
                    best = (pv, a, b)
    return best[1], best[2], best[0], p0


def refine_shift(clip, frame, ddy0, ddx0, ry=2, rx=12):
    """Per-frame refinement of a known global shift, grayscale, small window."""
    t0, l0, H, W = R.window(clip)
    _, TR, _, BR, _, _ = R.tile_quadrants(clip, frame)
    mask = R.splat_mask(clip, frame)
    TRg = TR.astype(np.float32).mean(axis=2)
    tgt = BR[t0:t0 + R.TH, l0:l0 + R.TW].astype(np.float32).mean(axis=2)
    valid = (~mask).astype(np.float32)
    vs = float(valid.sum())
    best = (-1e9, ddy0, ddx0)
    for a in range(ddy0 - ry, ddy0 + ry + 1):
        for b in range(ddx0 - rx, ddx0 + rx + 1):
            tt, ll = t0 + a, l0 + b
            if tt < 0 or ll < 0 or tt + R.TH > H or ll + R.TW > W:
                continue
            d = TRg[tt:tt + R.TH, ll:ll + R.TW] - tgt
            v = -float((d * d * valid).sum() / vs)
            if v > best[0]:
                best = (v, a, b)
    return best[1], best[2]


def global_shift(clip, frame, coarse=True):
    t0, l0, H, W = R.window(clip)
    _, TR, _, BR, _, _ = R.tile_quadrants(clip, frame)
    mask = R.splat_mask(clip, frame)
    tgt = BR[t0:t0 + R.TH, l0:l0 + R.TW]
    valid = ~mask
    p0 = R.psnr_u8(TR[t0:t0 + R.TH, l0:l0 + R.TW], tgt, valid)
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
    return best[1], best[2], best[0], p0


# --------------------------------------------------------------------------------------------
# restricted regions
# --------------------------------------------------------------------------------------------
def regions_restricted(gt, restrict):
    """R.regions() with the quantiles taken inside `restrict` and hi/lo intersected with it.

    gt, restrict: [n,H,W].  Returns None if the restriction has too few pixels to quantile.
    """
    if restrict.sum() < 200:
        return None
    gmag = np.zeros_like(gt)
    gmag[:, :, :-1] += np.abs(np.diff(gt, axis=2))
    gmag[:, :-1, :] += np.abs(np.diff(gt, axis=1))
    v = gmag[restrict]
    hi = (gmag >= np.quantile(v, 0.90)) & restrict
    lo = (gmag <= np.quantile(v, 0.50)) & restrict
    if hi[:, 1:-1, 1:-1].sum() < 50 or lo[:, 1:-1, 1:-1].sum() < 50 or lo[:, :, :-1].sum() < 50:
        return None
    gmin, gmax = R._nb(gt, np.min), R._nb(gt, np.max)
    return dict(hi=hi, lo=lo, gmin=gmin, gmax=gmax, gl=R._lap(gt),
                hi_i=hi[:, 1:-1, 1:-1], lo_i=lo[:, 1:-1, 1:-1])


def erode(mask2d, k):
    """Erode a 2-D bool mask by k pixels."""
    if k <= 0:
        return mask2d
    return ndimage.binary_erosion(mask2d, np.ones((2 * k + 1, 2 * k + 1), bool))


def dilate(mask2d, k):
    """Dilate a 2-D bool mask by k pixels.

    MEASURED: the disocclusion holes in this suite are 1-3 px wide stripes -- eroding by a single
    pixel deletes 97-99 % of them (0042: 9469 -> 145 px).  So "inside the hole" has almost no
    interior, and the statistic that can actually carry a rim/halo is the hole PLUS a few pixels
    of surround, which is what this builds.
    """
    if k <= 0:
        return mask2d
    return ndimage.binary_dilation(mask2d, np.ones((2 * k + 1, 2 * k + 1), bool))


# --------------------------------------------------------------------------------------------
# registration-free plateau overshoot, edge_profiles.py's definition, vectorised, centre-in-mask
# --------------------------------------------------------------------------------------------
PLAT = (6, 14)      # plateaus sampled 6..14 px either side of the edge  (edge_profiles.py)
SEG = PLAT[0]       # the +-5 px segment whose excursion beyond the plateaus is measured
MINCON = 0.12       # minimum edge contrast          (edge_profiles.py)
FLATTOL = 0.04      # plateau flatness tolerance = 0.02*1.5+0.01   (edge_profiles.py)


def _rowstats(img):
    """Per-pixel left/right plateau mean and std over the 8-wide windows edge_profiles.py uses."""
    c = np.concatenate([np.zeros((img.shape[0], 1), np.float64), np.cumsum(img, axis=1)], 1)
    c2 = np.concatenate([np.zeros((img.shape[0], 1), np.float64), np.cumsum(img ** 2, axis=1)], 1)
    n = PLAT[1] - PLAT[0]                                        # 8

    def win(s):   # mean/std of img[:, s : s+n] placed at column index 0..W-1
        m = (c[:, s + n:] - c[:, s:-n]) / n
        q = (c2[:, s + n:] - c2[:, s:-n]) / n
        return m, np.sqrt(np.maximum(q - m ** 2, 0))
    return win, c.shape[1] - 1


def overshoot_map(img):
    """Per-pixel (over, contrast, ok) for a vertical step edge centred at that pixel.

    `over` is the excursion of the +-5 px segment beyond the pixel's own two plateau levels,
    exactly edge_profiles.py's `over`, but returned for every pixel instead of 60 hand-picked
    edges, so it can be averaged over all in-hole edges of 150 frames.
    """
    H, W = img.shape
    win, _ = _rowstats(img)
    n = PLAT[1] - PLAT[0]
    mA, sA = win(0)    # window [x-14, x-7]  when placed at x -> needs shift
    # build aligned arrays with NaN padding
    a = np.full((H, W), np.nan)
    sa = np.full((H, W), np.nan)
    b = np.full((H, W), np.nan)
    sb = np.full((H, W), np.nan)
    # left plateau  = img[:, x-PLAT[1] : x-PLAT[0]]  -> window starting at x-14, length 8
    a[:, PLAT[1]:] = mA[:, :W - PLAT[1]]
    sa[:, PLAT[1]:] = sA[:, :W - PLAT[1]]
    # right plateau = img[:, x+PLAT[0]+1 : x+PLAT[1]+1] -> window starting at x+7, length 8
    lim = W - (PLAT[1] + 1)
    b[:, :lim] = mA[:, PLAT[0] + 1:PLAT[0] + 1 + lim]
    sb[:, :lim] = sA[:, PLAT[0] + 1:PLAT[0] + 1 + lim]

    p_lo = np.minimum(a, b)
    p_hi = np.maximum(a, b)
    contrast = np.abs(b - a)

    # segment img[:, x-5 : x+6]  (edge_profiles.py: row[xx-PLAT[0]+1 : xx+PLAT[0]])
    sw = sliding_window_view(img, 2 * SEG - 1, axis=1)          # width 11, centres 5..W-6
    smax = np.full((H, W), np.nan)
    smin = np.full((H, W), np.nan)
    smax[:, SEG - 1:W - SEG + 1] = sw.max(axis=2)
    smin[:, SEG - 1:W - SEG + 1] = sw.min(axis=2)

    over = np.maximum.reduce([smax - p_hi, p_lo - smin, np.zeros_like(p_hi)])
    ok = (contrast >= MINCON) & (sa <= FLATTOL) & (sb <= FLATTOL) & np.isfinite(over)
    return over, contrast, ok, p_lo, p_hi


def edge_candidates(org, extra_ok=None):
    """Pixels that qualify as a strong isolated step edge in the ORIGIN panel.

    gradient in the top 2 % of the frame + edge_profiles.py's contrast / plateau-flatness tests.
    """
    gx = np.abs(np.diff(org, axis=1))
    thr = np.quantile(gx, 0.98)
    g = np.zeros_like(org)
    g[:, :-1] = gx
    _, contrast, ok, p_lo, p_hi = overshoot_map(org)
    sel = ok & (g >= thr)
    if extra_ok is not None:
        sel &= extra_ok
    return sel, contrast, p_lo, p_hi


def panel_overshoot(img, sel, contrast_origin):
    """edge_profiles.py's overshoot, with ONE denominator (the ORIGIN edge contrast) for every
    panel, averaged over the selected pixels."""
    over, _, _, _, _ = overshoot_map(img)
    v = over[sel] / contrast_origin[sel]
    return float(np.nanmean(v)) if v.size else np.nan
