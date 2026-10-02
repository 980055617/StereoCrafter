#!/usr/bin/env python
"""THE HEADLINE TABLE: 12 test clips, LOSSLESS FFV1, deployed config, five configurations.

    origin                  deployed 8-step origin UNet
    mamba 5-slot (SHIPPED)  the protected deliverable
    origin + s25            25-step origin sampling (2.4x the cost) -- the compute ceiling
    origin + student        the ORIGIN-side step-distillation student  [trained on 0301 + 0204]
    THIS DELIVERABLE        5-slot Mamba + Mamba-side step-distilled up_blocks.3 params

CONTAMINATION, stated in the table rather than footnoted.  0301 and 0204 are TEST clips and BOTH the
origin-side student and mstudent1 were trained on them, so their 12-clip means are not fully held out.
This deliverable trained only on TRAIN-split clips, so every one of its 12 numbers is held out.  The
n=10 column (test minus {0301,0204}) is therefore the only column on which all five rows are
comparable, and it is the number the write-up should quote when comparing against the origin-side
student.

usage: headline_table_v1.py <out.txt> <deliverable_label> <scores.txt> [more...]
env    HL_CLIPS (default the 12 test clips)
"""
import os
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(\S+) dx=(\S+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                 r"gtSharp=(\S+) rightPSNR=(\S+) n=(\S+) path=(\S+)")
OUT, DELIV = sys.argv[1], sys.argv[2]
CLIPS = os.environ.get("HL_CLIPS",
                       "0042,0052,0125,0128,0141,0147,0170,0204,0225,0251,0259,0301").split(",")
TRAINED_ON = ["0301", "0204"]          # the origin-side student's / mstudent1's training clips

D, GT, PATH = defaultdict(dict), {}, defaultdict(dict)
for f in sys.argv[3:]:
    for line in open(f):
        m = ROW.match(line.strip())
        if not m:
            continue
        cl, tag = m.group(1), m.group(2)
        lab = tag[len(cl) + 1:] if tag.startswith(cl + "_") else tag
        D[cl][lab] = dict(lpips=float(m.group(6)), sharp=float(m.group(7)), leftPSNR=float(m.group(5)),
                          rightPSNR=float(m.group(9)), n=int(m.group(10)), off=(m.group(3), m.group(4)))
        PATH[cl][lab] = m.group(11)
        GT[cl] = float(m.group(8))

ORDER = [("origin_ll", "origin (deployed 8 steps)", ""),
         ("mamba_ll", "mamba 5-slot (SHIPPED)", ""),
         ("s25_ll", "origin + s25 (2.4x cost)", ""),
         ("student_ll", "origin + student", "trained on 0301+0204"),
         ("mstudent1_step600_ll", "mamba + mstudent1 s600", "trained on 0301+0204"),
         (DELIV, "THIS DELIVERABLE (A)", "trained on 10 TRAIN clips")]
have = [c for c in CLIPS if c in D]
L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


P("=" * 126)
P("HEADLINE TABLE -- 12 TEST CLIPS, LOSSLESS FFV1 (never the cv2 mp4v writer), real-GT LPIPS,")
P("DEPLOYED CONFIG: 8 sampler steps, EulerDiscrete Karras, guidance 1.01, 14-frame windows overlap 3,")
P("576x1024 centre crop.  scorer scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=4")
P("=" * 126)
P()
P("--- PER CLIP: LPIPS (lower is better) ---")
P(f"  {'row':30s} " + "".join(f"{c:>8s}" for c in have))
P(f"  {'GT sharpness':30s} " + "".join(f"{GT[c]:8.4f}" for c in have))
for lab, name, note in ORDER:
    if not any(lab in D[c] for c in have):
        continue
    P(f"  {name:30s} " + "".join(f"{D[c][lab]['lpips']:8.4f}" if lab in D[c] else f"{'--':>8s}"
                                 for c in have) + (f"   [{note}]" if note else ""))
P()
P("--- PER CLIP: delta vs deployed ORIGIN (negative = better) ---")
for lab, name, note in ORDER[1:]:
    if not any(lab in D[c] for c in have):
        continue
    P(f"  {name:30s} " + "".join(
        f"{D[c][lab]['lpips'] - D[c]['origin_ll']['lpips']:+8.4f}" if lab in D[c] and "origin_ll" in D[c]
        else f"{'--':>8s}" for c in have))
P()
P("--- PER CLIP: sharpness / GT sharpness (1.0 = matches the real right eye) ---")
for lab, name, note in ORDER:
    if not any(lab in D[c] for c in have):
        continue
    P(f"  {name:30s} " + "".join(f"{D[c][lab]['sharp'] / GT[c]:8.3f}" if lab in D[c] else f"{'--':>8s}"
                                 for c in have))
P()


def block(sub, title):
    P("=" * 126)
    P(title)
    P("=" * 126)
    P(f"  {'row':30s}{'meanLPIPS':>10s}{'d vs origin':>12s}{'d vs mamba':>11s}{'improved':>10s}"
      f"{'sh/mamba':>9s}{'sh/GT':>7s}{'worst clip':>22s}{'rPSNR':>8s}")
    om = sum(D[c]["origin_ll"]["lpips"] for c in sub) / len(sub)
    mm = sum(D[c]["mamba_ll"]["lpips"] for c in sub) / len(sub)
    for lab, name, note in ORDER:
        v = [D[c][lab]["lpips"] for c in sub if lab in D[c]]
        if len(v) != len(sub):
            continue
        m = sum(v) / len(sub)
        nimp = sum(1 for c in sub if D[c][lab]["lpips"] < D[c]["origin_ll"]["lpips"])
        shm = sum(D[c][lab]["sharp"] / D[c]["mamba_ll"]["sharp"] for c in sub) / len(sub)
        shg = sum(D[c][lab]["sharp"] / GT[c] for c in sub) / len(sub)
        rp = sum(D[c][lab]["rightPSNR"] for c in sub) / len(sub)
        wc = max(sub, key=lambda c: D[c][lab]["lpips"] - D[c]["origin_ll"]["lpips"])
        wd = D[wc][lab]["lpips"] - D[wc]["origin_ll"]["lpips"]
        P(f"  {name:30s}{m:10.4f}{m - om:+12.4f}{m - mm:+11.4f}{nimp:>7d}/{len(sub)}"
          f"{shm:9.4f}{shg:7.3f}{wc + ' ' + format(wd, '+.4f'):>22s}{rp:8.3f}"
          + (f"   [{note}]" if note else ""))
    P()


block(have, f"MEANS over n={len(have)} (all test clips) -- rows marked 'trained on 0301+0204' are NOT "
            f"held out here")
sub10 = [c for c in have if c not in TRAINED_ON]
block(sub10, f"MEANS over n={len(sub10)} (test minus 0301,0204) -- the only set on which ALL FIVE ROWS "
             f"are held out")
P("=" * 126)
P("PROVENANCE of every scored file")
P("=" * 126)
for lab, name, note in ORDER:
    ps = sorted({os.path.dirname(PATH[c][lab]).rsplit("/clips/", 1)[0] for c in have if lab in PATH[c]})
    P(f"  {name:30s} label={lab:34s} roots={ps}")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"\nwrote {OUT}")
