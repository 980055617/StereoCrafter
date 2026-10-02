#!/usr/bin/env python
"""STAGE 2: the 0042 disocclusion halo, re-measured over ALL frames and restricted to the holes.

What the review measured (outputs/review_20261001): haloFrac on ONE frame (f32) of ONE 384x384
window whose mask coverage was 5.93 %, i.e. 94 % of the pixels it averaged were NOT holes.
Numbers: deliverable 50.9 %, origin 47.0 %, origin+s25 44.3 %.

What this measures, for 6 configs (origin / shipped Mamba / step200 / step400 / step800
DELIVERABLE / origin+s25):

  A  hole            every frame, haloFrac over GT-top-decile-gradient pixels INSIDE the mask
  B  hole_dil4       the same over the mask DILATED by 4 px (hole + 4 px surround).  MEASURED:
                     these holes are 1-3 px wide stripes -- eroding by one pixel deletes 97-99 %
                     of them -- so there is no hole "interior" to isolate, and the band that can
                     actually carry a re-drawn rim is the hole plus its immediate surround.
  C  holecrop384     the review's own 384x384 hole window, UNRESTRICTED -- the direct
                     all-frames scale-up of the published 50.9 %
  D  hole_pfshift    variant A with the GT re-registered PER FRAME, to separate a real effect
                     from residual misregistration (0042 only)
  E  wholewin        the whole 576x1024 window, every 8th frame -- reproduces
                     METRICS_WHOLEFRAME.txt's convention as a harness anchor

plus a registration-free cross-check: edge_profiles.py's PLATEAU OVERSHOOT, evaluated on the
strong step edges of the ORIGIN panel whose centre falls inside the mask.

Metric operators are reviewlib's, which are ringing_metrics.py's verbatim.  Regions are computed
PER FRAME (n=1 stacks), so the gradient quantile thresholds are per-frame; that shifts absolute
levels slightly against the published whole-stack tables, but every config in a frame is scored
against the same thresholds, which is what the config-vs-config claim rests on.

usage: inhole_halo.py [clip ...]        (default: every clip the survey marked analyse=True)
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holelib as HL  # noqa: E402
R = HL.R

OUT = f"{HL.REPO}/outputs/rung_check_20261001"
SURVEY = json.load(open(f"{OUT}/survey.json"))
REVIEW = json.load(open(f"{HL.REPO}/outputs/review_20261001/metrics.json"))
LABELS = [lab for lab, _ in HL.PANELS6]
CHUNK = 4
WHOLE_STEP = 8
KEYS = ("haloFrac", "haloMean", "flatHF", "edgeHF", "stripeE")

clips = sys.argv[1:] or [c for c in HL.CLIPS12 if SURVEY[c]["analyse"]]
rep = open(f"{OUT}/INHOLE_HALO.txt", "w")


def tee(*s):
    print(*s, flush=True)
    print(*s, file=rep, flush=True)


tee(__doc__.split("usage:")[0].rstrip())
tee("=" * 118)
tee(f"clips analysed: {clips}")
tee("")


def refine(TRg, BRg, mask, t0, l0, H, W, ddy0, ddx0, ry=2, rx=12):
    tgt = BRg[t0:t0 + R.TH, l0:l0 + R.TW]
    valid = (~mask).astype(np.float32)
    vs = float(valid.sum())
    best = (1e18, ddy0, ddx0)
    for a in range(ddy0 - ry, ddy0 + ry + 1):
        for b in range(ddx0 - rx, ddx0 + rx + 1):
            tt, ll = t0 + a, l0 + b
            if tt < 0 or ll < 0 or tt + R.TH > H or ll + R.TW > W:
                continue
            d = TRg[tt:tt + R.TH, ll:ll + R.TW] - tgt
            e = float((d * d * valid).sum() / vs)
            if e < best[0]:
                best = (e, a, b)
    return best[1], best[2]


def agg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, p10=np.nan, p90=np.nan,
                    min=np.nan, max=np.nan)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                min=float(v.min()), max=float(v.max()))


RES = {}
for clip in clips:
    HL.clear_readers()
    S = SURVEY[clip]
    ddy0, ddx0 = S["gtShift"]
    nv = S["nvalid"]
    nv = min(nv, int(os.environ.get("RC_MAXFRAMES", "999999")))
    t0, l0, H, W = R.window(clip)
    pans = HL.panels(clip)                      # strict: all six must exist
    assert len(pans) == 6, (clip, pans)
    paths = dict(pans)
    hc = REVIEW.get(clip, {}).get("crops", {}).get("hole")
    do_pf = (clip == "0042")

    variants = ["hole", "hole_dil4", "wholewin"]
    if hc:
        variants.append("holecrop384")
    if do_pf:
        variants.append("hole_pfshift")

    acc = {v: {lab: {k: [] for k in KEYS} for lab in ["GT"] + LABELS} for v in variants}
    ov_acc = {lab: [] for lab in ["GT (registered)"] + LABELS}
    perframe = []
    pf_shifts = []
    nsel_acc = []
    maskfrac = []
    npx = []

    for i0 in range(0, nv, CHUNK):
        idxs = list(range(i0, min(i0 + CHUNK, nv)))
        tr = R.grab_many(f"video_data/train/{clip}_train.mp4", idxs)
        TRg = tr[:, :H, W:2 * W].astype(np.float32).mean(axis=3) / 255.0
        BRg = tr[:, H:, W:2 * W].astype(np.float32).mean(axis=3) / 255.0
        del tr
        pst = {lab: R.gray(R.grab_many(p, idxs)[:, :, R.TW:]) for lab, p in pans}

        for j, fidx in enumerate(idxs):
            mask = R.splat_mask(clip, fidx)
            maskfrac.append(float(mask.mean()))
            npx.append((int(mask.sum()), int(HL.erode(mask, 1).sum()),
                        int(HL.dilate(mask, 4).sum())))
            gtfix = TRg[j][t0 + ddy0:t0 + ddy0 + R.TH, l0 + ddx0:l0 + ddx0 + R.TW][None]
            pan1 = {lab: pst[lab][j][None] for lab in LABELS}
            row = dict(frame=fidx, maskFrac=float(mask.mean()))

            def do(vkey, gt1, restrict):
                reg = HL.regions_restricted(gt1, restrict)
                if reg is None:
                    return
                g = R.decompose(reg, gt1)
                for k in KEYS:
                    acc[vkey]["GT"][k].append(g[k])
                for lab in LABELS:
                    d = R.decompose(reg, pan1[lab])
                    for k in KEYS:
                        acc[vkey][lab][k].append(d[k])
                    row[f"{vkey}|{lab}|haloFrac"] = d["haloFrac"]
                    row[f"{vkey}|{lab}|edgeHF"] = d["edgeHF"]
                    row[f"{vkey}|{lab}|flatHF"] = d["flatHF"]
                row[f"{vkey}|GT|edgeHF"] = g["edgeHF"]
                row[f"{vkey}|GT|flatHF"] = g["flatHF"]

            do("hole", gtfix, mask[None])
            do("hole_dil4", gtfix, HL.dilate(mask, 4)[None])
            if fidx % WHOLE_STEP == 0:
                do("wholewin", gtfix, np.ones_like(mask)[None])
            if hc:
                y0, x0, sz = hc["y"], hc["x"], hc["size"]
                gtc = TRg[j][t0 + ddy0 + y0:t0 + ddy0 + y0 + sz,
                             l0 + ddx0 + x0:l0 + ddx0 + x0 + sz][None]
                pan_save = dict(pan1)
                pan1 = {lab: pst[lab][j][y0:y0 + sz, x0:x0 + sz][None] for lab in LABELS}
                do("holecrop384", gtc, np.ones_like(gtc, bool))
                pan1 = pan_save
            if do_pf:
                a, b = refine(TRg[j], BRg[j], mask, t0, l0, H, W, ddy0, ddx0)
                pf_shifts.append((fidx, int(a), int(b)))
                gtpf = TRg[j][t0 + a:t0 + a + R.TH, l0 + b:l0 + b + R.TW][None]
                do("hole_pfshift", gtpf, mask[None])

            # ---- registration-free plateau overshoot, edges whose centre is inside the mask ----
            org = pst[LABELS[0]][j]
            sel, contrast, _, _ = HL.edge_candidates(org, extra_ok=mask)
            ns = int(sel.sum())
            nsel_acc.append(ns)
            row["nEdgesInHole"] = ns
            if ns >= 40:
                ov_acc["GT (registered)"].append(HL.panel_overshoot(gtfix[0], sel, contrast))
                for lab in LABELS:
                    v = HL.panel_overshoot(pst[lab][j], sel, contrast)
                    ov_acc[lab].append(v)
                    row[f"overshoot|{lab}"] = v
            perframe.append(row)
        del TRg, BRg, pst

    # ---------------------------------------------------------------------------------- report
    tee("=" * 118)
    tee(f"### {clip}   nframes={nv} (ALL valid frames)   mean maskFrac={np.mean(maskfrac)*100:.4f}%"
        f"   GT shift used (ddy,ddx)=({ddy0},{ddx0})")
    if do_pf:
        dxs = [s[2] for s in pf_shifts]
        tee(f"    per-frame refined shift: ddx {min(dxs)}..{max(dxs)} "
            f"(median {int(np.median(dxs))}, fixed {ddx0})")
    npa = np.array(npx)
    tee(f"    mask pixels/frame: raw {npa[:,0].mean():.0f}  eroded-1px {npa[:,1].mean():.0f}"
        f"  dilated-4px {npa[:,2].mean():.0f}   -> the holes are 1-3 px wide STRIPES, not areas")
    tee(f"    step edges with centre inside the mask, per frame: mean {np.mean(nsel_acc):.0f}  "
        f"min {min(nsel_acc)}  max {max(nsel_acc)}   (frames used for overshoot: "
        f"{len(ov_acc[LABELS[0]])})")
    RES[clip] = dict(nframes=nv, maskMean=float(np.mean(maskfrac)), gtShift=[ddy0, ddx0],
                     variants={}, overshoot={}, pfShifts=pf_shifts)

    for v in variants:
        nfr = len(acc[v]["GT"]["haloFrac"])
        tee("")
        tee(f"  [{v}]  n_frames_scored={nfr}")
        tee(f"    {'config':24s} {'haloFrac% mean':>14s} {'sd':>7s} {'min':>7s} "
            f"{'max(worst)':>11s} {'d vs origin':>11s} | {'edgeHF/GT':>9s} {'flatHF/GT':>9s} "
            f"{'stripeE/GT':>10s}")
        base = agg(acc[v]["origin (deployed)"]["haloFrac"])["mean"]
        gtm = {k: agg(acc[v]["GT"][k])["mean"] for k in KEYS}
        store = {}
        for lab in LABELS:
            a = agg(acc[v][lab]["haloFrac"])
            r = {k: agg(acc[v][lab][k])["mean"] for k in KEYS}
            tee(f"    {lab:24s} {a['mean']:14.3f} {a['sd']:7.3f} {a['min']:7.3f} "
                f"{a['max']:11.3f} {a['mean']-base:+11.3f} | "
                f"{r['edgeHF']/gtm['edgeHF']:9.3f} {r['flatHF']/gtm['flatHF']:9.3f} "
                f"{r['stripeE']/gtm['stripeE']:10.3f}")
            store[lab] = dict(halo=a, ratios={k: r[k] / gtm[k] for k in ("edgeHF", "flatHF", "stripeE")},
                              raw=r)
        tee(f"    {'GT (real right eye)':24s} {0.0:14.3f} {'':>7s} {'':>7s} {'':>11s} {'':>11s} | "
            f"{1.0:9.3f} {1.0:9.3f} {1.0:10.3f}")
        # per-frame paired comparison: how often is each config above origin?
        o = np.array(acc[v]["origin (deployed)"]["haloFrac"])
        tee(f"    paired vs origin, per frame (fraction of frames with HIGHER halo than origin):")
        for lab in LABELS[1:]:
            x = np.array(acc[v][lab]["haloFrac"])
            n = min(len(x), len(o))
            if n < 2:
                tee(f"      {lab:24s} n={n}, skipped")
                continue
            d = x[:n] - o[:n]
            tee(f"      {lab:24s} above on {100*float((d>0).mean()):5.1f}% of frames   "
                f"mean d {d.mean():+7.3f}  sd of d {d.std(ddof=1):6.3f}  "
                f"worst frame d {d.max():+7.3f}")
        RES[clip]["variants"][v] = dict(n=nfr, gtMean=gtm, configs=store)

    if len(ov_acc[LABELS[0]]):
        tee("")
        tee("  [plateau overshoot, REGISTRATION-FREE, edges with centre inside the mask]")
        tee(f"    {'config':24s} {'mean':>9s} {'sd':>8s} {'min':>8s} {'max(worst)':>11s}")
        for lab in ["GT (registered)"] + LABELS:
            a = agg(ov_acc[lab])
            tee(f"    {lab:24s} {a['mean']:9.4f} {a['sd']:8.4f} {a['min']:8.4f} {a['max']:11.4f}")
            RES[clip]["overshoot"][lab] = a

    with open(f"{OUT}/perframe_{clip}.csv", "w", newline="") as fh:
        cols = sorted({k for r in perframe for k in r})
        wr = csv.DictWriter(fh, fieldnames=["frame"] + [c for c in cols if c != "frame"])
        wr.writeheader()
        wr.writerows(perframe)
    tee(f"    per-frame CSV: {OUT}/perframe_{clip}.csv")

json.dump(RES, open(f"{OUT}/inhole_halo.json", "w"), indent=1)
tee("")
tee(R.LEGEND)
rep.close()
print("\nDONE inhole")
