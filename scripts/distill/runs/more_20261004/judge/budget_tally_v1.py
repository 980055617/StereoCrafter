#!/usr/bin/env python
"""judge lane GPU tally (CPU): render seconds from the driver logs, scorer seconds from SCORE_START/SCORE_DONE and
TSCORE_START/TSCORE_DONE stamps (wall inside the scorer call, lock wait included).  usage: budget_tally_v1.py OUT.txt"""
import glob, os, re, sys
from datetime import datetime
os.chdir("/home/kawa/master_project/StereoCrafter")
O = "outputs/more_20261004/judge"; J = "scripts/distill/runs/more_20261004/judge"
L = []; tot = {"render": 0.0, "score": 0.0}
for lg in sorted(glob.glob(f"{O}/*/timing_gpu*.txt")):
    for ln in open(lg):
        if not ln.startswith("RUN "): continue
        m = re.search(r"(?:process_s|secs)=([0-9.]+)", ln); lab = " ".join(ln.split()[1:3])
        s = float(m.group(1)); tot["render"] += s; L.append(f"render {lg.split('/')[-2]:12s} {lab:45s} {s:8.1f} s")
def stamps(p, a, b):
    t0 = t1 = None
    for ln in open(p):
        if ln.startswith(a): t0 = datetime.strptime(ln.split()[1], "%Y-%m-%d_%H:%M:%S")
        if ln.startswith(b): t1 = datetime.strptime(ln.split()[-1], "%Y-%m-%d_%H:%M:%S")
    return (t1 - t0).total_seconds() if t0 and t1 else None
for p in sorted(glob.glob(f"{J}/SCORES_*.txt")):
    s = stamps(p, "SCORE_START", "SCORE_DONE")
    if s is not None: tot["score"] += s; L.append(f"score  {os.path.basename(p):58s} {s:8.1f} s (wall incl. lock wait)")
for p in sorted(glob.glob(f"{O}/rescore_temporal/*.log")):
    s = stamps(p, "TSCORE_START", "TSCORE_DONE")
    if s is not None: tot["score"] += s; L.append(f"score  {os.path.basename(p):58s} {s:8.1f} s (wall incl. lock wait)")
L.append(f"TOTAL renders {tot['render']/60:.1f} GPU-min, scoring <= {tot['score']/60:.1f} GPU-min (wall, lock waits included), sum <= {(tot['render']+tot['score'])/60:.1f}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L[-1:]))
