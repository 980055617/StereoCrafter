#!/usr/bin/env python
"""STAGE 1: mean disocclusion fraction inside the deployed 576x1024 window, all 12 TEST clips,
plus the per-clip GT disparity-registration shift and a drift check over 10 frames.

Decides which clips Task 2 analyses (threshold: mean mask fraction > 0.3 %).
Writes outputs/rung_check_20261001/MASK_SURVEY.txt and survey.json.  Renders nothing.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holelib as HL  # noqa: E402
R = HL.R

OUT = f"{HL.REPO}/outputs/rung_check_20261001"
os.makedirs(OUT, exist_ok=True)
rep = open(f"{OUT}/MASK_SURVEY.txt", "w")


def tee(*s):
    print(*s, flush=True)
    print(*s, file=rep, flush=True)


tee("""DISOCCLUSION SURVEY -- all 12 TEST clips, deployed 576x1024 window
=================================================================
maskFrac  = fraction of the deployed window covered by the splatting video's occlusion-mask
            quadrant (bottom-left), in RENDER coordinates, averaged over every 4th valid frame.
            reviewlib.splat_mask asserts the mask window == the GT window, which is also the
            check that this file's extension of R.OFFSETS to 12 clips is correct.
gtShift   = integer (ddy,ddx) that registers the REAL right eye to the render grid, estimated
            against the model's own WARPED right eye (BR quadrant, config-independent), holes
            excluded -- build_review.py's recipe verbatim.
drift     = the same estimate repeated on 10 evenly spaced frames.
""")

SURVEY = {}
for clip in HL.CLIPS12:
    HL.clear_readers()
    pans = HL.panels(clip, tolerant=True)
    nv = HL.nvalid(clip, [p for _, p in pans])
    t0, l0, H, W = R.window(clip)
    cand = [i for i in range(0, nv, 4) if 8 <= i <= nv - 9]
    mfr = np.array([R.splat_mask(clip, i).mean() for i in cand])
    fdense = int(cand[int(mfr.argmax())])
    ddy, ddx, pbest, p0 = HL.global_shift_fast(clip, fdense)

    dframes = [int(round(x)) for x in np.linspace(8, nv - 9, 10)]
    drift = []
    for f in dframes:
        a, b, pb, _ = HL.global_shift_fast(clip, f)
        drift.append((f, a, b, round(pb, 2)))
    dxs = [d[2] for d in drift]
    dys = [d[1] for d in drift]

    tee(f"--- {clip} ---  nvalid={nv}  window(t0,l0)=({t0},{l0})  quadrant {H}x{W}  "
        f"scorer offset {R.OFFSETS[clip]}")
    tee(f"    maskFrac mean={mfr.mean()*100:.4f}%  median={np.median(mfr)*100:.4f}%  "
        f"max={mfr.max()*100:.4f}% (frame {fdense})  n_sampled={len(cand)}")
    tee(f"    gtShift at densest frame = ({ddy},{ddx})   PSNR {p0:.2f} -> {pbest:.2f} dB")
    tee(f"    drift over 10 frames: ddx {min(dxs)}..{max(dxs)} (median {int(np.median(dxs))}), "
        f"ddy {min(dys)}..{max(dys)}")
    tee(f"    drift detail: {drift}")
    tee(f"    ANALYSE: {'YES' if mfr.mean() > 0.003 else 'no (mean mask <= 0.3%)'}")
    SURVEY[clip] = dict(nvalid=nv, t0=t0, l0=l0, H=H, W=W, offset=list(R.OFFSETS[clip]),
                        maskMean=float(mfr.mean()), maskMedian=float(np.median(mfr)),
                        maskMax=float(mfr.max()), denseFrame=fdense,
                        gtShift=[int(ddy), int(ddx)], gtPSNR=[p0, pbest],
                        driftDdxMedian=int(np.median(dxs)), drift=drift,
                        analyse=bool(mfr.mean() > 0.003),
                        panels={lab: p for lab, p in pans})

tee("\n--- RANKED by mean mask fraction ---")
tee(f"  {'clip':6s} {'maskMean%':>10s} {'maskMax%':>9s} {'gtShift':>10s} {'analyse':>8s}")
for clip, s in sorted(SURVEY.items(), key=lambda kv: -kv[1]["maskMean"]):
    tee(f"  {clip:6s} {s['maskMean']*100:10.4f} {s['maskMax']*100:9.4f} "
        f"{str(tuple(s['gtShift'])):>10s} {'YES' if s['analyse'] else 'no':>8s}")

json.dump(SURVEY, open(f"{OUT}/survey.json", "w"), indent=1)
rep.close()
print("\nDONE survey")
