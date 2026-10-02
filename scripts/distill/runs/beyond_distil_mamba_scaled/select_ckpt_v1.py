#!/usr/bin/env python
"""CHECKPOINT SELECTION on the DEV split, by SAMPLED LPIPS -- never by training loss.

The project's loss/quality anti-correlation has now been seen five times (mstudent1's loss fell
monotonically while its sampled LPIPS was not monotone, and selecting on loss would have shipped the
worse step800), so this script reads only the lossless sampled-LPIPS ROW lines and never touches
train_log.csv.

Selection set: the four dev-split clips fulldata_v1 designates for "model selection / stop rule"
(0040 0091 0184 0245), none of which any run trained on, and none of which appear in the 12-clip
headline table.  mstudent1 by contrast was selected on test clips, so its rungs are shown here too and
the ladder-vs-ladder comparison on these clips is the answer to "did more training clips help".

Tie-break rule, stated before the numbers are seen: if two rungs are within 0.0010 LPIPS of the best,
take the one whose mean sharpness-vs-GT ratio is closer to 1.0.  mstudent1 step600 already pushes the
most over-sharp test clips to ~1.7-1.8x GT sharpness, and more clips plus more steps could make that
worse while the LPIPS mean still improves.

usage: select_ckpt_v1.py <scores.txt> [more_scores.txt ...]
"""
import os
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(\S+) dx=(\S+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                 r"gtSharp=(\S+) rightPSNR=(\S+) n=(\S+) path=(\S+)")
DEV = os.environ.get("SEL_CLIPS", "0040,0091,0184,0245").split(",")
TIE = float(os.environ.get("SEL_TIE", "0.0010"))

D = defaultdict(dict)
GT = {}
for f in sys.argv[1:]:
    for line in open(f):
        m = ROW.match(line.strip())
        if not m:
            continue
        cl, tag = m.group(1), m.group(2)
        lab = tag[len(cl) + 1:] if tag.startswith(cl + "_") else tag
        D[cl][lab] = dict(lpips=float(m.group(6)), sharp=float(m.group(7)),
                          leftPSNR=float(m.group(5)), rightPSNR=float(m.group(9)), n=int(m.group(10)))
        GT[cl] = float(m.group(8))

have = [c for c in DEV if c in D]
assert have, "no dev rows found"
labs = sorted({l for c in have for l in D[c]},
              key=lambda s: (0 if s == "mamba_ll" else (1 if s.startswith("mstudent1") else 2),
                             int(re.search(r"step(\d+)", s).group(1)) if "step" in s else -1))

L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


P("=" * 118)
P("CHECKPOINT SELECTION -- DEV SPLIT, LOSSLESS FFV1, real-GT LPIPS, deployed config (8 steps, g 1.01)")
P(f"dev clips {have}   GT sharpness {[round(GT[c], 4) for c in have]}")
P("scorer scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py  SCORE_STEP=4")
P("=" * 118)
P(f"  {'row':30s} " + "".join(f"{c:>9s}" for c in have) + f"{'meanLPIPS':>11s}{'d vs mamba':>12s}"
  f"{'impr':>6s}{'sh/GT':>8s}{'sh/mamba':>10s}{'rPSNR':>8s}")
base = {c: D[c].get("mamba_ll") for c in have}
rows = {}
for lab in labs:
    v = [D[c][lab]["lpips"] for c in have if lab in D[c]]
    if len(v) != len(have):
        P(f"  {lab:30s} INCOMPLETE ({len(v)}/{len(have)} clips) -- excluded from selection")
        continue
    m = sum(v) / len(v)
    bm = sum(base[c]["lpips"] for c in have) / len(have) if all(base.values()) else None
    shgt = sum(D[c][lab]["sharp"] / GT[c] for c in have) / len(have)
    shmb = sum(D[c][lab]["sharp"] / base[c]["sharp"] for c in have) / len(have) if all(base.values()) else float("nan")
    rp = sum(D[c][lab]["rightPSNR"] for c in have) / len(have)
    nimp = sum(1 for c in have if base[c] and D[c][lab]["lpips"] < base[c]["lpips"])
    rows[lab] = dict(mean=m, shgt=shgt, shmb=shmb, nimp=nimp, per={c: D[c][lab]["lpips"] for c in have})
    P(f"  {lab:30s} " + "".join(f"{D[c][lab]['lpips']:9.4f}" for c in have)
      + f"{m:11.4f}{(m - bm) if bm else 0:+12.4f}{nimp:>4d}/{len(have)}{shgt:8.3f}{shmb:10.4f}{rp:8.3f}")
P()
cand = {k: v for k, v in rows.items() if k.startswith("mstudent2")}
if cand:
    best = min(cand, key=lambda k: cand[k]["mean"])
    near = [k for k in cand if cand[k]["mean"] - cand[best]["mean"] <= TIE]
    pick = min(near, key=lambda k: abs(cand[k]["shgt"] - 1.0)) if len(near) > 1 else best
    P(f"best by dev LPIPS      : {best}  mean {cand[best]['mean']:.4f}  sh/GT {cand[best]['shgt']:.3f}")
    P(f"within {TIE:.4f} of best : {sorted(near)}")
    P(f"SELECTED               : {pick}  mean {cand[pick]['mean']:.4f}  sh/GT {cand[pick]['shgt']:.3f}"
      + ("   (tie-break on sharpness-vs-GT)" if pick != best else ""))
    P(f"SELECTED_LABEL={pick}")
    pick_step = re.search(r"step(\d+)", pick).group(1)   # py3.11 forbids a backslash inside an f-string
    P(f"SELECTED_STEP={pick_step}")
m1 = {k: v for k, v in rows.items() if k.startswith("mstudent1")}
if m1 and cand:
    b1 = min(m1, key=lambda k: m1[k]["mean"])
    P()
    P("--- DID MORE TRAINING CLIPS HELP?  ladder vs ladder on the SAME dev clips, neither run trained "
      "on them ---")
    P(f"  mstudent1 (2 TEST clips, 26 windows)   best rung {b1}: {m1[b1]['mean']:.4f}")
    for k in sorted(m1):
        P(f"      {k:28s} {m1[k]['mean']:.4f}")
    P(f"  mstudent2 (10 TRAIN clips, 134 windows) best rung {best}: {cand[best]['mean']:.4f}")
    for k in sorted(cand, key=lambda s: int(re.search(r'step(\d+)', s).group(1))):
        P(f"      {k:28s} {cand[k]['mean']:.4f}")
    if "mstudent1_step800_ll" in m1 and "mstudent2_step800_ll" in cand:
        a, b = m1["mstudent1_step800_ll"]["mean"], cand["mstudent2_step800_ll"]["mean"]
        P(f"  MATCHED OPTIMISER BUDGET (both step800): 2 clips {a:.4f} vs 10 clips {b:.4f} = {b - a:+.4f}")
    P(f"  BEST vs BEST: {cand[best]['mean'] - m1[b1]['mean']:+.4f} "
      f"({'more clips HELPED' if cand[best]['mean'] < m1[b1]['mean'] else 'more clips did NOT help'})")
out = sys.argv[1].rsplit("/", 1)[0] + "/TABLE_DEV_SELECTION.txt"
open(out, "w").write("\n".join(L) + "\n")
print(f"\nwrote {out}")
