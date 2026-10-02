"""Shared geometry / panel / metric helpers for the 2026-10-01 visual review of
mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt.

NOTHING TRACKED IS MODIFIED.  All outputs go to outputs/review_20261001/.

GEOMETRY (verified empirically, see GEOMETRY.txt):
  video_data/train/<clip>_train.mp4 is a 2x2 tile of the HALF-size quadrants
      TL = left eye (real)        TR = right eye (real)  <- the LPIPS target
      BL = occlusion mask         BR = warped right eye   (the model's own input)
  H, W = tile.shape[0]//2, tile.shape[1]//2
  score_clip_ll.py slices the QUADRANT FIRST and then offsets inside it:
      gtR = tile[:, :H, W:2W]  ->  gtR[t0:t0+h, l0:l0+w],  t0=(H-h)//2+dy, l0=(W-w)//2+dx
  i.e. in raw-tile coordinates the right-eye window is  tile[t0:t0+h, W+l0 : W+l0+w].
  scripts/distill/runs/fulldata_v2/beyond4/{make_crops.py,ringing_metrics.py} both drop the
  `W +` term and therefore read the LEFT eye; that is corrected here.

  The occlusion mask that is pixel-registered with the render lives in the splatting video's
  bottom-left quadrant at exactly (t0, l0) as well (verified: (h128-576)//2 == t0 and
  (w128-1024)//2 == l0 for every clip in this suite).

METRIC DEFINITIONS are taken verbatim (operators, thresholds, quantiles, unweighted channel
mean) from scripts/distill/runs/fulldata_v2/beyond4/ringing_metrics.py, whose prior output is
scripts/distill/runs/skeptic1/RINGING_STUDENT.txt.  The only deviations, both stated in the
report: (1) the GT is the RIGHT eye (the prior file's regions were left-eye), and (2) for the
per-crop tables the 0.90/0.50 gradient quantiles are taken INSIDE the crop window, which is
what makes them a per-crop decomposition.
"""
import math
import os

import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)

TH, TW = 576, 1024
CROP = 384

# scorer-reported (dy, dx) for every row of every clip -- identical across all five configs
# (leftPSNR is identical on every row, which is the evidence that one offset registers all five).
OFFSETS = {
    "0042": (-12, -12), "0128": (-12, -12), "0141": (-12, -12), "0147": (-12, -12),
    "0204": (-28, 0), "0301": (-28, 0),
}

# panel order is the one the review prescribes; roots match the PROVENANCE block of
# scripts/distill/runs/beyond_distil_mamba_scaled/TABLE_HEADLINE_12CLIP.txt
PANELS = [
    ("origin (deployed)", "origin_ll", {"*": "outputs/beyond4_lossless"}),
    ("shipped 5-slot Mamba", "mamba_ll", {
        "0042": "outputs/beyond_distil_mamba", "0128": "outputs/beyond_distil_mamba",
        "0141": "outputs/beyond_distil_mamba", "0147": "outputs/skeptic1_stack",
        "0204": "outputs/skeptic1_stack", "0301": "outputs/skeptic1_stack"}),
    ("THIS DELIVERABLE", "mstudent2_step800_deliv_ll", {"*": "outputs/beyond_distil_mamba_scaled"}),
    ("origin+s25 (ceiling)", "s25_ll", {
        "0042": "outputs/skeptic1_stack", "0128": "outputs/skeptic1_stack",
        "0141": "outputs/skeptic1_stack", "0147": "outputs/beyond4_lossless",
        "0204": "outputs/beyond4_lossless", "0301": "outputs/beyond4_lossless"}),
]


def panel_path(clip, tag, roots):
    root = roots.get(clip, roots.get("*"))
    p = f"{root}/clips/{clip}_{tag}/{clip}_inpainting_results_sbs.mkv"
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


_VR = {}


def vr(path):
    if path not in _VR:
        _VR[path] = VideoReader(path, ctx=cpu(0))
    return _VR[path]


def nframes(path):
    return len(vr(path))


def grab(path, idx):
    """One frame, uint8 RGB.  Hard assert -- never silently clamp to a different frame."""
    v = vr(path)
    if not (0 <= idx < len(v)):
        raise IndexError(f"frame {idx} out of range for {path} (n={len(v)})")
    return v[idx].asnumpy()


def grab_many(path, idxs):
    v = vr(path)
    for i in idxs:
        if not (0 <= i < len(v)):
            raise IndexError(f"frame {i} out of range for {path} (n={len(v)})")
    return v.get_batch(list(idxs)).asnumpy()


def psnr_u8(a, b, m=None):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if m is None:
        e = ((a - b) ** 2).mean()
    else:
        mm = np.broadcast_to(m[..., None], a.shape)
        if mm.sum() == 0:
            return -1.0
        e = (((a - b) ** 2) * mm).sum() / mm.sum()
    return 10 * math.log10(1.0 / max(e / (255.0 ** 2), 1e-12))


def tile_quadrants(clip, idx):
    """TL/TR/BL/BR of the train tile at one frame, plus H, W."""
    t = grab(f"video_data/train/{clip}_train.mp4", idx)
    H, W = t.shape[0] // 2, t.shape[1] // 2
    return t[:H, :W], t[:H, W:2 * W], t[H:, :W], t[H:, W:2 * W], H, W


def window(clip, h=TH, w=TW):
    """(t0, l0) of the deployed window inside a quadrant, score_clip_ll.py math."""
    tl = grab(f"video_data/train/{clip}_train.mp4", 0)
    H, W = tl.shape[0] // 2, tl.shape[1] // 2
    dy, dx = OFFSETS[clip]
    return (H - h) // 2 + dy, (W - w) // 2 + dx, H, W


def splat_mask(clip, idx):
    """Occlusion mask (bool, True = disocclusion hole) in RENDER coordinates, 576x1024."""
    p = f"video_data/splatting/{clip}_splatting_results.mp4"
    f = grab(p, idx)
    hh, ww = f.shape[0] // 2, f.shape[1] // 2
    h128, w128 = hh // 128 * 128, ww // 128 * 128
    top, left = (h128 - TH) // 2, (w128 - TW) // 2
    t0, l0, _, _ = window(clip)
    assert (top, left) == (t0, l0), f"{clip}: mask window {(top,left)} != GT window {(t0,l0)}"
    return f[hh + top:hh + top + TH, left:left + TW, 0] > 127


def register_gt(TR, BR, maskbool, t0, l0, H, W, y0, x0, s=CROP,
                dy_rng=(-10, 11, 2), dx_rng=(-120, 41, 2)):
    """Integer window shift that registers the REAL right eye to the render frame.

    Target is the train tile's WARPED right-eye quadrant (BR) -- the pipeline's own model
    INPUT, hence pixel-registered with every render by construction and identical for all
    five configs, so this estimate cannot favour any panel.  Disocclusion-hole pixels are
    excluded because BR carries no valid data there.
    Returns (ddy, ddx, psnr_at_best, psnr_at_zero).
    """
    tgt = BR[t0 + y0:t0 + y0 + s, l0 + x0:l0 + x0 + s]
    valid = ~maskbool[y0:y0 + s, x0:x0 + s]
    if valid.sum() < 0.2 * valid.size:
        valid = np.ones_like(valid)
    p0 = psnr_u8(TR[t0 + y0:t0 + y0 + s, l0 + x0:l0 + x0 + s], tgt, valid)
    best = (-1e9, 0, 0)
    for st in (dy_rng[2], 1):
        cy, cx = best[1], best[2]
        yr = range(dy_rng[0], dy_rng[1], st) if st == dy_rng[2] else range(cy - 2, cy + 3)
        xr = range(dx_rng[0], dx_rng[1], st) if st == dy_rng[2] else range(cx - 3, cx + 4)
        for ddy in yr:
            for ddx in xr:
                tt, ll = t0 + y0 + ddy, l0 + x0 + ddx
                if tt < 0 or ll < 0 or tt + s > H or ll + s > W:
                    continue
                p = psnr_u8(TR[tt:tt + s, ll:ll + s], tgt, valid)
                if p > best[0]:
                    best = (p, ddy, ddx)
    return best[1], best[2], best[0], p0


# ---------------------------------------------------------------------------------------------
# ringing_metrics.py, verbatim operators.  `stack` arrays are float [n,H,W] in [0,1] (channel mean)
# ---------------------------------------------------------------------------------------------
def gray(a_u8):
    return a_u8.astype(np.float32).mean(axis=-1) / 255.0


def _lap(a):
    return np.abs(4 * a[:, 1:-1, 1:-1] - a[:, :-2, 1:-1] - a[:, 2:, 1:-1]
                  - a[:, 1:-1, :-2] - a[:, 1:-1, 2:])


def _nb(a, f):
    s = [a]
    for ax in (1, 2):
        for k in (-1, 1):
            s.append(np.roll(a, k, axis=ax))
    return f(np.stack(s), axis=0)


def regions(gt):
    """GT-defined regions, ringing_metrics.py definitions."""
    gmag = np.zeros_like(gt)
    gmag[:, :, :-1] += np.abs(np.diff(gt, axis=2))
    gmag[:, :-1, :] += np.abs(np.diff(gt, axis=1))
    hi = gmag >= np.quantile(gmag, 0.90)
    lo = gmag <= np.quantile(gmag, 0.50)
    gmin, gmax = _nb(gt, np.min), _nb(gt, np.max)
    return dict(hi=hi, lo=lo, gmin=gmin, gmax=gmax,
                gl=_lap(gt), hi_i=hi[:, 1:-1, 1:-1], lo_i=lo[:, 1:-1, 1:-1])


def decompose(reg, y):
    """haloFrac%, haloMean, flatHF, edgeHF, stripeE for one panel stack."""
    over = np.maximum(y - reg["gmax"], 0) + np.maximum(reg["gmin"] - y, 0)
    hf = _lap(y)
    flat = float(hf[reg["lo_i"]].mean())
    edge = float(hf[reg["hi_i"]].mean())
    col = np.abs(np.diff(y, axis=2))
    stripe = float(col[reg["lo"][:, :, :-1]].mean())
    oe = over[reg["hi"]]
    return dict(haloFrac=100.0 * float((oe > 0.04).mean()), haloMean=float(oe.mean()),
                flatHF=flat, edgeHF=edge, stripeE=stripe)


HDR = (f"{'label':24s} {'haloFrac%':>10s} {'haloMean':>9s} {'flatHF':>9s} {'flatHF/GT':>10s} "
       f"{'edgeHF':>9s} {'edgeHF/GT':>10s} {'stripeE':>9s} {'stripeE/GT':>11s}")


def fmt_row(lab, d, g):
    return (f"{lab:24s} {d['haloFrac']:10.3f} {d['haloMean']:9.5f} {d['flatHF']:9.5f} "
            f"{d['flatHF']/g['flatHF']:10.3f} {d['edgeHF']:9.5f} {d['edgeHF']/g['edgeHF']:10.3f} "
            f"{d['stripeE']:9.5f} {d['stripeE']/g['stripeE']:11.3f}")


LEGEND = """  haloFrac% = % of GT-edge pixels whose value falls outside the GT's own 3x3 [min,max] by >0.04 (halo/ringing)
  flatHF    = mean |Laplacian| inside GT-flat regions (noise / amplified splatting texture)   <- ARTEFACT
  edgeHF    = mean |Laplacian| inside the GT's top-decile gradient regions (detail at edges)  <- DETAIL
  stripeE   = mean |horizontal difference| inside GT-flat regions (splatting-stripe energy)   <- ARTEFACT
  definitions verbatim from scripts/distill/runs/fulldata_v2/beyond4/ringing_metrics.py
  prior 4-clip whole-frame version: scripts/distill/runs/skeptic1/RINGING_STUDENT.txt"""
