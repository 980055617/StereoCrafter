"""probe 2 step A table + the pre-registered P2A rule (CPU).  Reads input_decomp_v1/<clip>.json (input_decomp_v1.py).
P2A: V lowers the input stripe energy iff INPUT stripeE/GT is >= 10 % below R1's on >= 3/4 clips AND INPUT
     edgeHF/GT is never more than 10 % below R1's.  Step B renders every passing V, else the single V with the
     largest mean stripeE reduction.
usage: table_p2a_v1.py <out_txt>"""
import json, os, sys
import numpy as np
D = "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/deep_20261004/input_side/input_decomp_v1"
OUT = sys.argv[1]
assert not os.path.exists(OUT), OUT
CL = ["0301", "0204", "0052", "0147"]
VS = ["deployed", "R1", "NA", "B1", "B4", "ZB", "SS4", "SS4C"]
J = {c: json.load(open(f"{D}/{c}.json")) for c in CL if os.path.exists(f"{D}/{c}.json")}
L = []
L.append("PROBE 2 step A: the model's INPUT warped window, all variants against the SAME GT regions (R1's REG_FRAME")
L.append("registration), scored frames every 4th.  Sources: " + ", ".join(f"{D}/{c}.json" for c in J))
L.append(f"{'variant':8s} {'clip':5s} {'holes%':>7s} {'stripeE/GT':>10s} {'d vs R1':>8s} {'edgeHF/GT':>9s} {'d vs R1':>8s} "
         f"{'flatHF/GT':>9s} {'selfFlatStripe/GT':>17s} {'selfEdge/GT':>11s} {'sharp/GT':>8s}")
dS, dE = {}, {}
for v in VS:
    for c in CL:
        if c not in J or v not in J[c]["variants"]:
            continue
        r = J[c]["variants"][v]; r1 = J[c]["variants"]["R1"]
        ds = r["ratio"]["stripeE"] / r1["ratio"]["stripeE"] - 1
        de = r["ratio"]["edgeHF"] / r1["ratio"]["edgeHF"] - 1
        dS.setdefault(v, {})[c] = ds; dE.setdefault(v, {})[c] = de
        L.append(f"{v:8s} {c:5s} {100*r['hole_frac']:7.3f} {r['ratio']['stripeE']:10.3f} {100*ds:+7.1f}% {r['ratio']['edgeHF']:9.3f} "
                 f"{100*de:+7.1f}% {r['ratio']['flatHF']:9.3f} {r['self_ratio']['selfFlatStripe']:17.3f} "
                 f"{r['self_ratio']['selfEdgeHF']:11.3f} {r['self_ratio']['sharp']:8.3f}")
    L.append("")
L.append("P2A verdicts (n clips available = %d):" % len(J))
best, bestv = None, None
for v in VS:
    if v in ("deployed", "R1") or v not in dS:
        continue
    n_ok = sum(dS[v][c] <= -0.10 for c in dS[v])
    blur = min(dE[v].values())
    ok = n_ok >= 3 and blur >= -0.10 and len(dS[v]) == 4
    m = float(np.mean(list(dS[v].values())))
    if best is None or m < best:
        best, bestv = m, v
    L.append(f"  {v:6s} stripeE >=10% lower on {n_ok}/{len(dS[v])} clips, mean d {100*m:+.1f}%, worst edgeHF d {100*blur:+.1f}% -> "
             f"{'PASS' if ok else 'fail'}")
L.append(f"  largest mean stripeE reduction: {bestv} ({100*best:+.1f}%)")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
