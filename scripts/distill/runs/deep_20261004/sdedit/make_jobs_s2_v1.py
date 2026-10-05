#!/usr/bin/env python
"""S2 trigger (PREREG.txt): jobs for every (variant, model) that PASSED P4 at S1, on the 8 non-regime test clips.
usage: make_jobs_s2_v1.py TABLE_S1_REGIME.json OUT_jobs.txt      (refuses to overwrite; writes nothing if none passed)
"""
import json
import os
import sys

T, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}")
tab = json.load(open(T))
assert tab["rule"] == "P4", tab["rule"]
SIG = {"sd31": ("s31", "warp"), "sd7": ("s7", "warp"), "sd1": ("s1", "warp"), "sd1fill": ("s1", "warpfill")}
CLIPS = "0042 0125 0128 0141 0170 0225 0251 0259".split()
passed = [(r["model"], lab.split("_")[-1]) for lab, r in tab["results"].items() if r["verdict"]["PASS"]]
print(f"P4 passed: {passed or 'NONE'}")
if not passed:
    sys.exit(0)
with open(OUT, "w") as fh:
    fh.write(f"# S2 renders for the P4 passes {passed} on the 8 non-regime test clips (generated from {T})\n")
    for c in CLIPS:
        for m, v in passed:
            s, mode = SIG[v]
            fh.write(f"{c} {m}_g100_{v} {m} 1.00 {s} 8 {mode}\n")
print(f"wrote {OUT}")
