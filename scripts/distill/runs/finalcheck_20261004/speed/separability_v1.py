#!/usr/bin/env python
"""Mamba share vs inference-setting share, from score_clip_ll.py ROW lines ONLY.

For each sampler setting S (8x2@1.01, 8x1@1.00, T6 6x2@1.01, T5 5x2@1.01, T5 5x1@1.00):
  MAMBA SHARE at S     = LPIPS(deliverable @ S) - LPIPS(origin @ S)          (same sampler, different UNet)
  SETTING COST origin  = LPIPS(origin @ S)      - LPIPS(origin 8x2 @1.01)    (same UNet, different sampler)
  SETTING COST deliv   = LPIPS(deliverable @ S) - LPIPS(deliverable 8x2 @1.01)
  TOTAL vs origin 8x2  = LPIPS(deliverable @ S) - LPIPS(origin 8x2 @1.01) = MAMBA SHARE at S + SETTING COST origin
usage: separability_v1.py OUT.txt SCORES.txt [SCORES.txt ...]
"""
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=\S+ dx=\S+ leftPSNR=\S+ lpips=(\S+) sharp=(\S+) gtSharp=\S+ "
                 r"rightPSNR=\S+ n=\S+ path=(\S+)")
OUT = sys.argv[1]
D, SRC = defaultdict(dict), defaultdict(dict)
for f in sys.argv[2:]:
    for line in open(f):
        m = ROW.match(line.strip())
        if m:
            cl, tag = m.group(1), m.group(2)
            lab = tag[len(cl) + 1:]
            D[cl][lab] = (float(m.group(3)), float(m.group(4)))
            SRC[lab][cl] = f
CLIPS = ["0042", "0052", "0125", "0128", "0141", "0147", "0170", "0204", "0225", "0251", "0259", "0301"]
SETTINGS = [("8x2 @1.01 (deployed)", "mstudent2_step800_deliv_ll", "origin_ll"),
            ("8x1 @1.00", "deliv_g100_s8", "origin_g100_s8"),
            ("T6 6x2 @1.01", "deliv_g101_T6pad", "origin_g101_T6pad"),
            ("T5 5x2 @1.01", "deliv_g101_T5pad", "origin_g101_T5pad"),
            ("T5 5x1 @1.00", "deliv_g100_T5pad", "origin_g100_T5pad")]
DREF, OREF = "mstudent2_step800_deliv_ll", "origin_ll"
L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


def mean(x):
    return sum(x) / len(x)


P("=" * 132)
P("SEPARABILITY -- 12 test clips, LOSSLESS FFV1, real-GT LPIPS (score_clip_ll.py SCORE_STEP=4); T6/T5 rows RNG-padded")
P("MAMBA SHARE = deliverable - origin at the SAME sampler setting (negative = the Mamba deliverable is better)")
P("=" * 132)
P(f"  {'setting':22s}" + "".join(f"{c:>8s}" for c in CLIPS))
for name, dl, og in SETTINGS:
    P(f"  {name:22s}" + "".join(f"{D[c][dl][0] - D[c][og][0]:+8.4f}" for c in CLIPS))
P()
P(f"  {'setting':22s}{'deliv':>8s}{'origin':>8s}{'MAMBA':>9s}{'improved':>9s}{'worst (least gain)':>20s}"
  f"{'sh d/o':>8s}{'cost(origin)':>13s}{'cost(deliv)':>12s}{'TOTAL vs o8x2':>14s}")
for name, dl, og in SETTINGS:
    dm = mean([D[c][dl][0] for c in CLIPS])
    om = mean([D[c][og][0] for c in CLIPS])
    share = [D[c][dl][0] - D[c][og][0] for c in CLIPS]
    imp = sum(1 for s in share if s < 0)
    wc = max(CLIPS, key=lambda c: D[c][dl][0] - D[c][og][0])
    shr = mean([D[c][dl][1] / D[c][og][1] for c in CLIPS])
    co = om - mean([D[c][OREF][0] for c in CLIPS])
    cd = dm - mean([D[c][DREF][0] for c in CLIPS])
    tot = dm - mean([D[c][OREF][0] for c in CLIPS])
    P(f"  {name:22s}{dm:8.4f}{om:8.4f}{mean(share):+9.4f}{imp:>6d}/12{wc + ' ' + format(D[wc][dl][0] - D[wc][og][0], '+.4f'):>20s}"
      f"{shr:8.4f}{co:+13.4f}{cd:+12.4f}{tot:+14.4f}")
P()
P("Reading: TOTAL vs o8x2 = MAMBA (at that setting) + cost(origin).  'sh d/o' = mean per-clip sharpness ratio "
  "deliverable/origin at that setting.")
P()
P("SOURCES (score file per label)")
for name, dl, og in SETTINGS:
    for lab in (dl, og):
        P(f"  {lab:28s} {sorted(set(SRC[lab][c] for c in CLIPS))}")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"\nwrote {OUT}")
