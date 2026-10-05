#!/usr/bin/env python
"""PREREG_ADDENDUM_2 B2/B3 lines (CPU, reads finished S2 outputs only; reported, never gating).
usage: p12_sensitivity_v1.py <stage_dir> <TABLE_S2_EXT12.json> OUT.txt
B2  per variant: REG_FRAME improved count (primary, as in the table), REG_CLIP improved count, improved count over the 11
    clips without 0125, the same without 0225 (ADDENDUM 3), and the 12-clip mean stripeE ratio, each against the same
    model's deployed render.
B3  the A1 round rule (mean REG_FRAME - origin_ll <= -0.005 and >= 9/12 better) for every variant AND for the no-model rows
    INPUT_warp / INPUT_fill (their registered LPIPS from the same S2 scorer runs, reg/<clip>.json; 0125 from reg_wide).
"""
import json
import os
import sys

os.chdir("/home/kawa/master_project/StereoCrafter")
SD, TJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}")
tab = json.load(open(TJ))
clips = tab["clips"]
L = []


def reg_of(c, lab, var):
    if c == "0125" and var == "REG_FRAME" and os.path.exists(f"{SD}/reg_wide/0125.json"):
        d = json.load(open(f"{SD}/reg_wide/0125.json"))
        if lab in d["configs"]:
            return d["configs"][lab]["lpips_clip"]["REG_FRAME"]
    d = json.load(open(f"{SD}/reg/{c}.json"))
    return d["configs"][lab]["lpips_clip"][var] if lab in d["configs"] else None


L.append(f"P12 sensitivity (PREREG_ADDENDUM_2 B2), stage {SD}, clips {' '.join(clips)}")
L.append(f"  {'variant':22s} {'REG_FRAME impr':>15s} {'REG_CLIP impr':>14s} {'w/o 0125 impr':>14s} {'w/o 0225 impr':>14s} {'stripeR mean':>13s} {'P12 (primary)':>14s}")
for lab, r in tab["results"].items():
    i_wo = [c for c, d in zip(clips, r["dREG_FRAME"]) if c != "0125" and d < 0]
    i_wo2 = [c for c, d in zip(clips, r["dREG_FRAME"]) if c != "0225" and d < 0]   # PREREG_ADDENDUM_3 (zero-hole clip)
    L.append(f"  {lab:22s} {r['n_improved_REG_FRAME']:>9d}/{len(clips):<5d} {r['n_improved_REG_CLIP']:>8d}/{len(clips):<5d} "
             f"{len(i_wo):>8d}/{len([c for c in clips if c != '0125']):<5d} "
             f"{len(i_wo2):>8d}/{len([c for c in clips if c != '0225']):<5d} {r['mean_stripe_ratio']:13.3f} "
             f"{('PASS' if r['verdict']['PASS'] else 'FAIL'):>14s}")
L.append("")
L.append("ROUND RULE vs ORIGIN (A1/B3): mean REG_FRAME(row) - REG_FRAME(origin_ll) <= -0.0050 AND better on >= 9/12")
rows = list(tab["results"].keys()) + ["mstudent2_step800_deliv_ll", "INPUT_warp", "INPUT_fill"]
for lab in rows:
    vals = []
    for c in clips:
        a, o = reg_of(c, lab, "REG_FRAME"), reg_of(c, "origin_ll", "REG_FRAME")
        if a is None:
            vals = None
            break
        vals.append(a - o)
    if vals is None:
        L.append(f"  {lab:28s} not scored on every clip of the stage")
        continue
    m = sum(vals) / len(vals)
    n = sum(v < 0 for v in vals)
    met = len(clips) == 12 and m <= -0.005 and n >= 9
    L.append(f"  {lab:28s} mean d {m:+.4f}  better {n:2d}/{len(clips)}  -> {'MET' if met else 'not met'}   per clip: "
             + " ".join(f"{c}:{v:+.4f}" for c, v in zip(clips, vals)))
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
