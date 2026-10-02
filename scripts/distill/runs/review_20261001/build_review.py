#!/usr/bin/env python
"""Build the visual-evidence package for the 2026-10-01 StereoCrafter deliverable.

Writes ONLY into outputs/review_20261001/.  Reuses the lossless FFV1 renders already on disk;
renders nothing.  No tracked file is modified.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reviewlib as R  # noqa: E402

OUT = "outputs/review_20261001"
for d in ("strips", "context"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

CLIPS = ["0147", "0141", "0042", "0128", "0301", "0204"]
NEIGH = (-8, -4, 0, 4, 8)
WHOLE_STEP = 8   # matches RINGING_STUDENT.txt's SCORE_STEP=8

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception:
    FONT = FONT_S = ImageFont.load_default()

geo = open(f"{OUT}/GEOMETRY.txt", "w")
mpc = open(f"{OUT}/METRICS_PER_CROP.txt", "w")
mwf = open(f"{OUT}/METRICS_WHOLEFRAME.txt", "w")
RESULT = {}


def tee(fh, *s):
    print(*s, flush=True)
    print(*s, file=fh, flush=True)


geo.write("""GEOMETRY AND REGISTRATION AUDIT -- outputs/review_20261001
=========================================================
Train tile quadrants (verified by PSNR against the renders' passthrough left half, and by a
near-binary test on the mask quadrant):
    TL = real left eye      TR = real right eye (the LPIPS target)
    BL = occlusion mask     BR = warped right eye (the model's own input)
score_clip_ll.py slices the quadrant FIRST, so the right-eye window in raw-tile coordinates is
    tile[t0:t0+h, W+l0 : W+l0+w]     with t0=(H-h)//2+dy, l0=(W-w)//2+dx
scripts/distill/runs/fulldata_v2/beyond4/{make_crops.py,ringing_metrics.py} omit the `W +`
term and therefore read the LEFT eye.  Corrected here.

gtWinPSNR = PSNR of the real right eye against the model-independent WARPED right eye over the
deployed 576x1024 window, disocclusion holes excluded, at the scorer's window and at the best
integer shift.  A large gain means the real right eye is NOT pixel-registered with the renders
at the scorer's window, so a GT panel / GT-defined region taken there is laterally displaced.
""")


def tr_crops(clip, frames, t0, l0, h, w, shifts, chunk=4):
    """TR-quadrant crops for several frames and several (ddy,ddx) shifts, batched decode.

    Returns {shift: float32 [n,h,w] channel-mean in [0,1]}.
    """
    out = {s: [] for s in shifts}
    p = f"video_data/train/{clip}_train.mp4"
    for i in range(0, len(frames), chunk):
        blk = frames[i:i + chunk]
        arr = R.grab_many(p, blk)
        H, W = arr.shape[1] // 2, arr.shape[2] // 2
        TRb = arr[:, :H, W:2 * W]
        for s in shifts:
            a, b = s
            out[s].append(R.gray(TRb[:, t0 + a:t0 + a + h, l0 + b:l0 + b + w]))
        del arr, TRb
    return {s: np.concatenate(v, 0) for s, v in out.items()}


for clip in CLIPS:
    t0, l0, H, W = R.window(clip)
    dy, dx = R.OFFSETS[clip]
    ppaths = [(lab, R.panel_path(clip, tag, roots)) for lab, tag, roots in R.PANELS]
    ns = {"train": R.nframes(f"video_data/train/{clip}_train.mp4"),
          "splat": R.nframes(f"video_data/splatting/{clip}_splatting_results.mp4")}
    for lab, p in ppaths:
        ns[lab] = R.nframes(p)
    nvalid = min(ns.values())

    # ---- frame choice: densest disocclusion inside the deployed window, VALID for every source
    cand = [i for i in range(0, nvalid, 4) if 8 <= i <= nvalid - 9]
    mfr = np.array([R.splat_mask(clip, i).mean() for i in cand])
    fidx = int(cand[int(mfr.argmax())])
    assert fidx < nvalid, (clip, fidx, nvalid)

    TL, TR, BL, BR, _H, _W = R.tile_quadrants(clip, fidx)
    mask = R.splat_mask(clip, fidx)

    # ---- global registration of the real right eye over the whole deployed window ----
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

    tee(geo, f"\n--- {clip} ---  chosen frame={fidx}   valid frame range 0..{nvalid-1}   nframes {ns}")
    tee(geo, f"  scorer offset (dy,dx)=({dy},{dx})   window (t0,l0)=({t0},{l0})   quadrant {H}x{W}")
    tee(geo, f"  mask fraction in deployed window: mean={mfr.mean()*100:.3f}%  chosen frame={mask.mean()*100:.3f}%")
    tee(geo, f"  gtWinPSNR scorer window = {gp0:.2f} dB ; best at (ddy,ddx)=({gddy},{gddx}) = {gp:.2f} dB"
             f"   -> residual GT disparity offset {gddx:+d} px")

    GTreg_full = TR[t0 + gddy:t0 + gddy + R.TH, l0 + gddx:l0 + gddx + R.TW]

    # cross-check: how well does each candidate GT window match the ORIGIN render's right half?
    _oR = R.grab(R.panel_path(clip, R.PANELS[0][1], R.PANELS[0][2]), fidx)[:, R.TW:]
    tee(geo, "  PSNR of the deployed ORIGIN render's right half against: "
             f"left-eye GT @scorer {R.psnr_u8(TL[t0:t0+R.TH, l0:l0+R.TW], _oR, valid):.2f} dB | "
             f"right-eye GT @scorer {R.psnr_u8(TR[t0:t0+R.TH, l0:l0+R.TW], _oR, valid):.2f} dB | "
             f"right-eye GT registered {R.psnr_u8(GTreg_full, _oR, valid):.2f} dB")

    # ---- window selection, in RENDER coordinates -------------------------------------------
    g = GTreg_full.astype(np.float32).mean(axis=2)
    en = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
    ii = np.zeros((en.shape[0] + 1, en.shape[1] + 1), np.float64)
    ii[1:, 1:] = en.cumsum(0).cumsum(1)
    mi = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.float64)
    mi[1:, 1:] = mask.astype(np.float64).cumsum(0).cumsum(1)
    S = R.CROP

    def box(a, y, x, s=S):
        return a[y + s, x + s] - a[y, x + s] - a[y + s, x] + a[y, x]

    # gradient integral image is (TH-1, TW-1); keep every window inside it
    ys = list(range(0, R.TH - S, 8))
    xs = list(range(0, R.TW - S, 8))
    hv, hy, hx = max((box(mi, y, x), y, x) for y in ys for x in xs)
    ranked = sorted(((box(ii, y, x), y, x) for y in ys for x in xs), reverse=True)
    tex, texthr = None, None
    # the texture window must be texture-rich and as hole-free as this clip allows; relax the
    # hole budget only as far as necessary so that it stays a DIFFERENT region from the hole crop
    for thr, ovmax in ((0.004, 0.25), (0.01, 0.25), (0.03, 0.25), (0.08, 0.25), (1.0, 0.25)):
        for v, y, x in ranked:
            if box(mi, y, x) / (S * S) > thr:
                continue
            ovl = max(0, min(y + S, hy + S) - max(y, hy)) * max(0, min(x + S, hx + S) - max(x, hx))
            if ovl > ovmax * S * S:
                continue
            tex, texthr = (v, y, x), thr
            break
        if tex is not None:
            break
    assert tex is not None, clip
    tv, ty, tx = tex
    tee(geo, f"  texture window (y,x)=({ty},{tx}) size={S} meanGrad={tv/S/S:.3f} maskCov={box(mi,ty,tx)/S/S*100:.3f}% (hole budget {texthr*100:.1f}%)")
    tee(geo, f"  hole    window (y,x)=({hy},{hx}) size={S} maskCov={hv/S/S*100:.3f}%")

    RESULT[clip] = dict(frame=fidx, t0=t0, l0=l0, H=H, W=W, dy=dy, dx=dx, nvalid=nvalid,
                        nframes=ns, maskFracFrame=float(mask.mean()), maskFracMean=float(mfr.mean()),
                        gtWinPSNR_scorer=gp0, gtWinPSNR_best=gp, gtGlobalShift=[gddy, gddx],
                        crops={})

    # ---- context sheet ---------------------------------------------------------------------
    orig_full = R.grab(ppaths[0][1], fidx)[:, R.TW:]
    ovr = np.array(Image.fromarray(np.ascontiguousarray(orig_full)).convert("RGB"))
    ovr[mask] = (0.45 * ovr[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    ctx = Image.fromarray(ovr)
    d = ImageDraw.Draw(ctx)
    d.rectangle([tx, ty, tx + S, ty + S], outline=(60, 255, 60), width=4)
    d.text((tx + 7, ty + 7), "TEXTURE", fill=(60, 255, 60), font=FONT)
    d.rectangle([hx, hy, hx + S, hy + S], outline=(80, 160, 255), width=4)
    d.text((hx + 7, hy + 7), "HOLE", fill=(80, 160, 255), font=FONT)
    d.rectangle([0, R.TH - 26, 560, R.TH], fill=(0, 0, 0))
    d.text((6, R.TH - 22), f"{clip} f{fidx} - origin render, disocclusion mask in red, crop windows",
           fill=(255, 255, 0), font=FONT_S)
    ctxp = f"{OUT}/context/{clip}_f{fidx}_context.png"
    ctx.save(ctxp)
    RESULT[clip]["context"] = ctxp

    # ---- per-region strips + metrics -------------------------------------------------------
    for rtype, (y0, x0) in (("texture", (ty, tx)), ("hole", (hy, hx))):
        lddy, lddx, lp, lp0 = R.register_gt(TR, BR, mask, t0, l0, H, W, y0, x0, s=S,
                                            dx_rng=(gddx - 48, gddx + 49, 2))
        covw = float(mask[y0:y0 + S, x0:x0 + S].mean())
        tee(geo, f"  [{rtype}] local GT registration (ddy,ddx)=({lddy},{lddx})"
                 f"  psnr {lp0:.2f} -> {lp:.2f} dB   maskCov={covw*100:.3f}%")

        gt_reg = TR[t0 + y0 + lddy:t0 + y0 + lddy + S, l0 + x0 + lddx:l0 + x0 + lddx + S]
        gt_sco = TR[t0 + y0:t0 + y0 + S, l0 + x0:l0 + x0 + S]
        pan = [("GT (real right eye)", gt_reg)]
        for lab, p in ppaths:
            pan.append((lab, R.grab(p, fidx)[:, R.TW:][y0:y0 + S, x0:x0 + S]))
        pan.append(("GT @ scorer window", gt_sco))

        GAP, BAR, FOOT = 12, 26, 26
        Wt = S * len(pan) + GAP * (len(pan) - 1)
        img = Image.new("RGB", (Wt, S + BAR + FOOT), (14, 14, 14))
        dd = ImageDraw.Draw(img)
        sharps = {}
        for k, (lab, arr) in enumerate(pan):
            x = k * (S + GAP)
            assert arr.shape[:2] == (S, S), (clip, rtype, lab, arr.shape)
            img.paste(Image.fromarray(np.ascontiguousarray(arr)), (x, BAR))
            col = ((120, 255, 120) if k == 0 else
                   (255, 210, 80) if lab == "THIS DELIVERABLE" else
                   (150, 200, 255) if lab.startswith("GT") else (235, 235, 235))
            dd.text((x + 4, 5), f"{k}. {lab}", fill=col, font=FONT)
            a = arr.astype(np.float64)
            sh = float(np.abs(a[:, 1:] - a[:, :-1]).mean() / 255.0)
            sharps[lab if k else "GT"] = sh
            dd.text((x + 4, S + BAR + 5), f"cropSharp(h)={sh:.5f}", fill=(190, 190, 190), font=FONT_S)
        sp = f"{OUT}/strips/{clip}_f{fidx}_{rtype}_5panel_100pct.png"
        img.save(sp)

        # ---- decomposition tables ----
        nb = sorted({min(max(fidx + o, 0), nvalid - 1) for o in NEIGH})
        pstk = {}
        for lab, p in ppaths:
            pstk[lab] = {f: R.gray(R.grab(p, f)[:, R.TW:][y0:y0 + S, x0:x0 + S]) for f in nb}
        gtc = tr_crops(clip, nb, t0 + y0, l0 + x0, S, S, [(lddy, lddx), (0, 0)])
        i0 = nb.index(fidx)
        tabs = {}
        for key, frames, shift in (("frame_reg", [i0], (lddy, lddx)),
                                   ("nb_reg", list(range(len(nb))), (lddy, lddx)),
                                   ("frame_scorerwin", [i0], (0, 0))):
            gs = gtc[shift][frames]
            reg = R.regions(gs)
            gd = R.decompose(reg, gs)
            rows = {"GT": gd}
            for lab, _ in ppaths:
                rows[lab] = R.decompose(reg, np.stack([pstk[lab][nb[j]] for j in frames]))
            tabs[key] = rows

        tee(mpc, f"\n### {clip}  frame {fidx}  region={rtype.upper()}  window(y,x)=({y0},{x0}) size={S}"
                 f"  maskCov={covw*100:.3f}%  GTshift=({lddy},{lddx})")
        tee(mpc, f"    strip: {sp}")
        for key, title in (
            ("frame_reg", "A. displayed frame, GT regions from the DISPARITY-REGISTERED real right eye   <- PRIMARY"),
            ("nb_reg", f"B. 5-frame neighbourhood {nb}, same registered GT regions (stability check)"),
            ("frame_scorerwin", "C. displayed frame, GT regions at the SCORER WINDOW, unregistered (convention of RINGING_STUDENT.txt)"),
        ):
            tee(mpc, f"  {title}")
            tee(mpc, "  " + R.HDR)
            g0 = tabs[key]["GT"]
            tee(mpc, "  " + R.fmt_row("GT", g0, g0))
            for lab, _ in ppaths:
                tee(mpc, "  " + R.fmt_row(lab, tabs[key][lab], g0))
        RESULT[clip]["crops"][rtype] = dict(y=y0, x=x0, size=S, maskCov=covw, strip=sp,
                                            gtShift=[lddy, lddx], regPSNR=[lp0, lp],
                                            cropSharp=sharps, tables=tabs, nb=nb)
        del pstk, gtc

    # ---- whole-frame decomposition (continuation of RINGING_STUDENT.txt) -------------------
    frames = list(range(0, nvalid, WHOLE_STEP))
    gtf = tr_crops(clip, frames, t0, l0, R.TH, R.TW, [(gddy, gddx), (0, 0)], chunk=3)
    pst = {lab: np.stack([R.gray(R.grab(p, f)[:, R.TW:]) for f in frames]) for lab, p in ppaths}
    wf = {}
    for key, shift in (("reg", (gddy, gddx)), ("scorerwin", (0, 0))):
        reg = R.regions(gtf[shift])
        gd = R.decompose(reg, gtf[shift])
        rows = {"GT": gd}
        for lab, _ in ppaths:
            rows[lab] = R.decompose(reg, pst[lab])
        wf[key] = rows
        tee(mwf, f"\n=== {clip}  WHOLE 576x1024 WINDOW, n={len(frames)} frames (step {WHOLE_STEP}), "
                 f"GT regions = {'DISPARITY-REGISTERED' if key == 'reg' else 'SCORER WINDOW (unregistered)'} ===")
        tee(mwf, R.HDR)
        tee(mwf, R.fmt_row("GT", gd, gd))
        for lab, _ in ppaths:
            tee(mwf, R.fmt_row(lab, rows[lab], gd))
    RESULT[clip]["wholeframe"] = wf
    del gtf, pst, TL, TR, BL, BR

for fh in (mpc, mwf, geo):
    print("\n" + R.LEGEND, file=fh)
    fh.close()
json.dump(RESULT, open(f"{OUT}/metrics.json", "w"), indent=1)
print("\nDONE")
