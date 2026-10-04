#!/usr/bin/env python
"""ays_20261004 / robust -- PREREG ADDENDUM 1 (b): gate R0' after the fact.  CPU only, read-only.
My pass-2 UNREG LPIPS of the AYS5 row (outputs/ays_20261004/robust/score_reg_ays5_v1/<clip>.json) must equal the
published score_clip_ll.py ROW lpips of the same render within 1e-4 (and the same dy/dx/n), where the ROW line comes
from the ays5 lane's SCORES_AYS5_S2_<clip>.txt or the judge's SCORES_J3c_AYS_*.txt (the path must match).
usage: python check_r0p_ays5_v1.py <out.txt>
"""
import glob
import json
import os
import re
import sys

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
LAB = "AYS5pad8_origin_g100"
ROWS5 = json.load(open("scripts/distill/runs/ays_20261004/robust/ROWS_ays5.json"))
ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(\S+) dx=(\S+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                 r"gtSharp=(\S+) rightPSNR=(\S+) n=(\S+) path=(\S+)")
pub = {}
for f in sorted(glob.glob("scripts/distill/runs/ays_20261004/ays5/SCORES_AYS5_S2_*.txt")
                + glob.glob("scripts/distill/runs/more_20261004/judge/SCORES_J3c_AYS_*.txt")):
    for ln in open(f, errors="replace"):
        m = ROW.match(ln.strip())
        if m and m.group(2) == f"{m.group(1)}_{LAB}":
            pub.setdefault((m.group(1), m.group(11)), []).append((float(m.group(6)), int(m.group(3)), int(m.group(4)),
                                                                  int(m.group(10)), float(m.group(7)), f))
L, bad, unchecked, mx = [], [], [], 0.0
for c in ROWS5["clips"]:
    mine = json.load(open(f"outputs/ays_20261004/robust/score_reg_ays5_v1/{c}.json"))["configs"][LAB]
    recs = pub.get((c, mine["path"]), [])
    if not recs:
        unchecked.append(c)
        L.append(f"UNCHECKED {c}: no ROW line for {mine['path']}")
        continue
    vals = {(r[0], r[1], r[2], r[3]) for r in recs}
    if len(vals) > 1:
        bad.append(c)
        L.append(f"FAIL {c}: conflicting ROW lines {recs}")
        continue
    lp, dy, dx, n = vals.pop()
    dv = abs(mine["lpips_clip"]["UNREG"] - lp)
    mx = max(mx, dv)
    ok = dv <= 1e-4 and (mine["dy"], mine["dx"], mine["n"]) == (dy, dx, n)
    if not ok:
        bad.append(c)
    L.append(f"{'PASS' if ok else 'FAIL'} {c}: mine UNREG {mine['lpips_clip']['UNREG']:.6f} vs ROW {lp:.6f} (|d| {dv:.1e}) "
             f"offset mine ({mine['dy']},{mine['dx']}) n {mine['n']} vs ROW ({dy},{dx}) n {n}  sharp {recs[0][4]}  "
             f"[{', '.join(sorted({r[5] for r in recs}))}]")
L.append(f"R0' {'PASS' if not bad and not unchecked else ('FAIL' if bad else 'INCOMPLETE')}: {12 - len(bad) - len(unchecked)}/12 "
         f"checked within 1e-4 (max |d| {mx:.1e}); failing {bad or 'none'}; unchecked {unchecked or 'none'}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
