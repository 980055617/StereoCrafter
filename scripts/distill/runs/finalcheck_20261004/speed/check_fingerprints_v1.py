#!/usr/bin/env python
"""G7 RNG-pairing check: per-window initial-latent fingerprints (speed_log.json "init_md5") of every render in
outputs/finalcheck_20261004/speed/clips must equal the clip's 8-step fingerprints.

Reference per clip = the fingerprints of its 8-step renders (labels containing "_s8" or "T8sig"), which must all
agree with each other (model and guidance do not change CUDA RNG consumption).  Expectations:
  *_s8*, *T8sig*, *pad*   -> every window identical to the reference
  *nat*                   -> window 0 identical, later windows DIFFERENT (unpadded schedule shifts the RNG)
usage: check_fingerprints_v1.py OUT.txt
"""
import glob
import json
import os
import sys
from collections import defaultdict

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
recs = defaultdict(dict)
for sp in sorted(glob.glob("outputs/finalcheck_20261004/speed/clips/*/speed_log.json")):
    d = os.path.basename(os.path.dirname(sp))
    clip, label = d.split("_", 1)
    recs[clip][label] = json.load(open(sp))["init_md5"]
L, nfail = [], 0
for clip in sorted(recs):
    refs = {lab: fp for lab, fp in recs[clip].items() if "_s8" in lab or "T8sig" in lab}
    uniq = {tuple(v) for v in refs.values()}
    if len(uniq) != 1:
        L.append(f"FAIL {clip}: the 8-step renders disagree among themselves: "
                 f"{ {k: v[:2] for k, v in refs.items()} }")
        nfail += 1
        continue
    ref = list(next(iter(uniq)))
    L.append(f"{clip}: reference = {len(refs)} eight-step renders agree ({len(ref)} windows; w0 {ref[0][:12]}..)")
    for lab, fp in sorted(recs[clip].items()):
        same = [a == b for a, b in zip(fp, ref)]
        nwin_ok = len(fp) == len(ref)
        if "nat" in lab:
            ok = nwin_ok and same[0] and not any(same[1:])
            exp = "w0 same, w1.. different"
        else:
            ok = nwin_ok and all(same)
            exp = "all windows same"
        nfail += (not ok)
        L.append(f"  {'PASS' if ok else 'FAIL'} {clip}_{lab:28s} windows {len(fp)}/{len(ref)}  identical={sum(same)}  "
                 f"(expected: {exp})")
L.append(f"SUMMARY clips={len(recs)} failed={nfail}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
