#!/usr/bin/env python
"""Supplement (descriptive, not pre-registered): per-clip deliverable delta vs a SEED-REALISATION YARDSTICK.

deliv_g100_T5nat and deliv_g100_T5pad are the same model, schedule and guidance; they differ ONLY in the initial noise
of sampler windows >= 1 (T5pad re-uses the 8-step renders' noise, T5nat draws its own), so |T5nat - T5pad| per clip
is a direct, if partial (window 0 shared), measurement of how much one noise realisation moves a clip's LPIPS.
A per-clip deliverable delta that is not clearly larger than this yardstick is not resolved by a single-seed render.
usage: python supplement_v1.py <score_dir> <wide_dir> <out.txt>
"""
import json
import os
import sys

import numpy as np

SCORE, WIDE, OUT = sys.argv[1:4]
assert not os.path.exists(OUT), OUT
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
L = []


def P(s=""):
    L.append(s)
    print(s)


def load(c):
    pw = os.path.join(WIDE, f"{c}.json")
    return json.load(open(pw if os.path.exists(pw) else os.path.join(SCORE, f"{c}.json")))


P("SEED-REALISATION YARDSTICK vs the deliverable's per-clip delta (descriptive supplement, not a PREREG criterion)")
P(f"scores: {SCORE} (+ widened {WIDE} for the clips it holds)")
P("yardstick s = |LPIPS(T5nat) - LPIPS(T5pad)| (same model/schedule/guidance, different noise in windows >= 1)")
P(f"{'clip':5s} {'GTshift':>8s} {'origin U':>9s} {'origin F':>9s} {'dDel U':>8s} {'dDel F':>8s} {'s U':>7s} {'s F':>7s} "
  f"{'|dDel F|/s F':>12s} {'rel dDel U':>10s} {'rel dDel F':>10s}")
ratios, rows = [], []
for c in CLIPS:
    d = load(c)
    cf = d["configs"]
    g = lambda r, v: cf[r]["lpips_clip"][v]
    dU = g("mstudent2_step800_deliv_ll", "UNREG") - g("origin_ll", "UNREG")
    dF = g("mstudent2_step800_deliv_ll", "REG_FRAME") - g("origin_ll", "REG_FRAME")
    sU = abs(g("deliv_g100_T5nat", "UNREG") - g("deliv_g100_T5pad", "UNREG"))
    sF = abs(g("deliv_g100_T5nat", "REG_FRAME") - g("deliv_g100_T5pad", "REG_FRAME"))
    ratio = abs(dF) / max(sF, 1e-6)
    ratios.append(ratio)
    rows.append(dict(clip=c, dU=dU, dF=dF, sU=sU, sF=sF, ratio=ratio))
    P(f"{c:5s} {d['reg']['clip_ddx']:+8d} {g('origin_ll', 'UNREG'):9.4f} {g('origin_ll', 'REG_FRAME'):9.4f} {dU:+8.4f} "
      f"{dF:+8.4f} {sU:7.4f} {sF:7.4f} {ratio:12.1f} {100 * dU / g('origin_ll', 'UNREG'):+9.1f}% "
      f"{100 * dF / g('origin_ll', 'REG_FRAME'):+9.1f}%")
sF_all = np.array([r["sF"] for r in rows])
P()
P(f"yardstick under REG_FRAME: median {np.median(sF_all):.4f}, max {sF_all.max():.4f} ({rows[int(np.argmax(sF_all))]['clip']}); "
  f"RMS {np.sqrt(np.mean(sF_all ** 2)):.4f}")
resolved = [r["clip"] for r in rows if r["dF"] < 0 and r["ratio"] >= 3]
unres = [r["clip"] for r in rows if r["ratio"] < 3]
P(f"clips whose REG_FRAME deliverable delta is negative and >= 3x their own yardstick: {len(resolved)}/12 {resolved}")
P(f"clips NOT resolved by one seed (|delta| < 3x yardstick): {unres}")
rel_u = np.mean([100 * r['dU'] / load(r['clip'])['configs']['origin_ll']['lpips_clip']['UNREG'] for r in rows])
rel_f = np.mean([100 * r['dF'] / load(r['clip'])['configs']['origin_ll']['lpips_clip']['REG_FRAME'] for r in rows])
P(f"mean relative change of the deliverable vs origin: UNREG {rel_u:+.2f}%  REG_FRAME {rel_f:+.2f}%  (absolute deltas are "
  f"nearly equal, the denominator shrinks under registration)")
open(OUT, "w").write("\n".join(L) + "\n")
