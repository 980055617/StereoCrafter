#!/usr/bin/env python
"""Phase A of the blind-rating build (PREREG.txt "FRAME RULE", "WINDOW RULES", "REGISTRATION").

Model-independent, rule-determined selection only -- reads the train tile (GT right eye TR, warped
input BR), the splatting video's mask quadrant and the frame counts.  NO render pixel is read here.

Per clip:
  * band [nv//3, 2nv//3), fc = nv//2
  * slot 2: densest-disocclusion (frame, window) over the band, or FALLBACK (< 0.25 %) -> texture2 at fc
  * clip-global registration shift at fc (and at the slot-2 frame)
  * slot 1: highest GT-gradient-energy window at fc (hole-budget relaxation, overlap cap vs slot 2)
  * a GT-ONLY context image of fc with a coordinate grid and the slot-1/slot-2 windows, for the hand
    choice of slot 3 (characteristic)

usage: python select_regions.py OUTDIR      (OUTDIR must not exist; CPU only)
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blindlib as B  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

R = B.R
S, TH, TW = B.S, B.TH, B.TW
FALLBACK_COV = 0.0025
CAP = 0.25
BUDGETS = (0.004, 0.01, 0.03, 0.08, 1.0)

OUT = B.new_dir(sys.argv[1])
os.makedirs(f"{OUT}/context")
log = B.Tee(f"{OUT}/select.log")
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def band_masks(clip, frames, t0, l0, chunk=6):
    """[n,TH,TW] bool masks for `frames`, sliced exactly as reviewlib.splat_mask does."""
    out = []
    p = B.splat_path(clip)
    for i in range(0, len(frames), chunk):
        arr = R.grab_many(p, frames[i:i + chunk])
        hh, ww = arr.shape[1] // 2, arr.shape[2] // 2
        h128, w128 = hh // 128 * 128, ww // 128 * 128
        top, left = (h128 - TH) // 2, (w128 - TW) // 2
        assert (top, left) == (t0, l0), (clip, top, left, t0, l0)
        out.append(arr[:, hh + top:hh + top + TH, left:left + TW, 0] > 127)
        del arr
    return np.concatenate(out, 0)


def pick_texture(en_ii, m_ii, avoid, exclude_eq=()):
    ranked = sorted(((B.box(en_ii, y, x), y, x) for y in B.GRID_Y for x in B.GRID_X), reverse=True)
    for thr in BUDGETS:
        for v, y, x in ranked:
            if (y, x) in exclude_eq:
                continue
            if B.box(m_ii, y, x) / (S * S) > thr:
                continue
            if any(B.overlap_frac((y, x), a) > CAP for a in avoid):
                continue
            return dict(y=y, x=x, energy=v / S / S, maskCov=B.box(m_ii, y, x) / S / S, hole_budget=thr)
    raise RuntimeError("no texture window satisfies the cap")


SEL = {}
for clip in B.CLIPS:
    tstart = time.time()
    nv, ns = B.nvalid(clip)
    b0, b1, fc = B.band(nv)
    t0, l0, H, W = R.window(clip)
    dy, dx = B.SCORER_OFFSETS[clip]
    # V3 (geometry): reviewlib's own assert, at fc
    m_fc_ref = R.splat_mask(clip, fc)

    frames = list(range(b0, b1))
    masks = band_masks(clip, frames, t0, l0)
    assert np.array_equal(masks[frames.index(fc)], m_fc_ref), clip

    per_frame, best = [], None
    for k, f in enumerate(frames):
        mi = B.integral(masks[k])
        bw = max((B.box(mi, y, x), -y, -x) for y in B.GRID_Y for x in B.GRID_X)
        cnt, y, x = bw[0], -bw[1], -bw[2]
        per_frame.append(dict(frame=f, wholeWindowMaskPct=float(masks[k].mean() * 100),
                              bestWindowCovPct=float(cnt / S / S * 100), y=y, x=x))
        key = (cnt, -abs(f - fc), -y, -x)
        if best is None or key > best[0]:
            best = (key, f, y, x, cnt)
    _, fD, yD, xD, cD = best
    covD = cD / S / S
    fallback = bool(covD < FALLBACK_COV)

    # ---- registration at fc (and at fD) -------------------------------------------------------
    TL, TR, BL, BR, _H, _W = R.tile_quadrants(clip, fc)
    assert (_H, _W) == (H, W)
    gddy, gddx, gp, gp0 = B.global_shift(TR, BR, m_fc_ref, t0, l0, H, W)
    reg = {str(fc): dict(ddy=gddy, ddx=gddx, psnr_reg=gp, psnr_unreg=gp0)}
    if not fallback and fD != fc:
        _TL, _TR, _BL, _BR, _, _ = R.tile_quadrants(clip, fD)
        mD = masks[frames.index(fD)]
        a, b, p1, p0 = B.global_shift(_TR, _BR, mD, t0, l0, H, W)
        reg[str(fD)] = dict(ddy=a, ddx=b, psnr_reg=p1, psnr_unreg=p0)
        del _TL, _TR, _BL, _BR

    GT = TR[t0 + gddy:t0 + gddy + TH, l0 + gddx:l0 + gddx + TW]
    assert GT.shape == (TH, TW, 3)
    en_ii = B.integral(B.grad_energy(GT))
    m_ii = B.integral(m_fc_ref)

    crops = []
    if not fallback:
        crops.append(dict(slot=2, region_type="disocclusion", frame=fD, y=yD, x=xD,
                          maskCovPct=covD * 100))
        tex = pick_texture(en_ii, m_ii, avoid=[(yD, xD)])
        crops.insert(0, dict(slot=1, region_type="texture", frame=fc, y=tex["y"], x=tex["x"],
                             energy=tex["energy"], maskCovPct=tex["maskCov"] * 100,
                             hole_budget=tex["hole_budget"]))
    else:
        tex = pick_texture(en_ii, m_ii, avoid=[])
        tex2 = pick_texture(en_ii, m_ii, avoid=[(tex["y"], tex["x"])], exclude_eq=((tex["y"], tex["x"]),))
        crops.append(dict(slot=1, region_type="texture", frame=fc, y=tex["y"], x=tex["x"],
                          energy=tex["energy"], maskCovPct=tex["maskCov"] * 100,
                          hole_budget=tex["hole_budget"]))
        crops.append(dict(slot=2, region_type="texture2", frame=fc, y=tex2["y"], x=tex2["x"],
                          energy=tex2["energy"], maskCovPct=tex2["maskCov"] * 100,
                          hole_budget=tex2["hole_budget"],
                          note=(f"FALLBACK: best disocclusion window over the band covers only "
                                f"{covD*100:.3f}% (< {FALLBACK_COV*100:.2f}%) -> second texture region")))

    SEL[clip] = dict(nvalid=nv, nframes=ns, band=[b0, b1], fc=fc, scorer_offset=[dy, dx],
                     window_t0l0=[t0, l0], quadrant_HW=[H, W],
                     disocclusion_best=dict(frame=fD, y=yD, x=xD, covPct=covD * 100),
                     fallback=fallback, global_registration=reg, crops=crops,
                     band_mask_profile=per_frame)

    log(f"\n--- {clip}  nvalid={nv} {ns}  band=[{b0},{b1})  fc={fc}  quadrant {H}x{W}  window (t0,l0)=({t0},{l0})")
    log(f"  mask: whole-window % over band mean={np.mean([p['wholeWindowMaskPct'] for p in per_frame]):.3f} "
        f"max={max(p['wholeWindowMaskPct'] for p in per_frame):.3f};  best window cov over band "
        f"= {covD*100:.3f}% at f{fD} (y={yD},x={xD})  -> {'FALLBACK (texture2)' if fallback else 'disocclusion crop'}")
    for fk, r in reg.items():
        log(f"  global GT registration @f{fk}: (ddy,ddx)=({r['ddy']},{r['ddx']})  "
            f"PSNR vs warped input {r['psnr_unreg']:.2f} -> {r['psnr_reg']:.2f} dB")
    for c in crops:
        extra = (f"energy={c['energy']:.3f} hole_budget={c['hole_budget']*100:.1f}%" if "energy" in c else "")
        log(f"  slot{c['slot']} {c['region_type']:13s} f{c['frame']} (y,x)=({c['y']},{c['x']}) "
            f"maskCov={c['maskCovPct']:.3f}% {extra}")
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            log(f"  overlap slot{crops[i]['slot']}/slot{crops[j]['slot']} = "
                f"{B.overlap_frac((crops[i]['y'], crops[i]['x']), (crops[j]['y'], crops[j]['x']))*100:.1f}%")

    # ---- GT-only context image of fc (for the hand choice of slot 3) ---------------------------------
    img = Image.fromarray(np.ascontiguousarray(GT)).convert("RGB")
    canvas = Image.new("RGB", (TW + 60, TH + 60), (20, 20, 20))
    canvas.paste(img, (40, 30))
    d = ImageDraw.Draw(canvas, "RGBA")
    for gx in range(0, TW + 1, 64):
        d.line([(40 + gx, 30), (40 + gx, 30 + TH)], fill=(255, 255, 255, 70 if gx % 128 else 130), width=1)
        if gx % 128 == 0:
            d.text((40 + gx - 8, 12), str(gx), fill=(255, 255, 0), font=FONT_S)
    for gy in range(0, TH + 1, 64):
        d.line([(40, 30 + gy), (40 + TW, 30 + gy)], fill=(255, 255, 255, 70 if gy % 128 else 130), width=1)
        if gy % 128 == 0:
            d.text((4, 30 + gy - 6), str(gy), fill=(255, 255, 0), font=FONT_S)
    colors = {"texture": (60, 255, 60), "texture2": (255, 160, 40), "disocclusion": (80, 160, 255)}
    for c in crops:
        col = colors[c["region_type"]]
        x0, y0 = 40 + c["x"], 30 + c["y"]
        d.rectangle([x0, y0, x0 + S, y0 + S], outline=col + (255,), width=3)
        lab = c["region_type"] + ("" if c["frame"] == fc else f" (f{c['frame']})")
        d.text((x0 + 6, y0 + 4), lab, fill=col + (255,), font=FONT)
    d.text((40, TH + 36), f"{clip} f{fc}  GT right eye (registered {gddy},{gddx})  -- GT only, no render pixels",
           fill=(255, 255, 0), font=FONT_S)
    canvas.save(f"{OUT}/context/{clip}_f{fc}_gt_context.png")
    log(f"  context: {OUT}/context/{clip}_f{fc}_gt_context.png   [{time.time()-tstart:.1f}s]")
    del TL, TR, BL, BR, masks

B.jdump(SEL, f"{OUT}/selection.json")
log(f"\nfallback clips: {[c for c in B.CLIPS if SEL[c]['fallback']]}")
log("done")
