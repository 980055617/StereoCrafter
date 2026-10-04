#!/usr/bin/env python
"""C3 RNG-pairing check for the fewer_steps lane (adapted from finalcheck_20261004/speed/check_fingerprints_v1.py).

Reference per clip = the per-window initial-latent fingerprints ("init_md5" in speed_log.json) of the speed lane's 8-step
renders outputs/finalcheck_20261004/speed/clips/<clip>_{deliv,origin}_g100_s8 (which must agree with each other: model and
guidance do not change CUDA RNG consumption).  Every render of THIS lane whose label contains "pad" must match the reference
in every window (SK_RNG_PAD_TO=8 makes an N-step schedule consume the RNG exactly like the 8-step one).  The reference
deliv_g100_T5pad renders of the speed lane are listed too, as a sanity row.
usage: check_fingerprints_fs_v1.py OUT.txt
"""
import glob
import json
import os
import sys
from collections import defaultdict

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
ref = defaultdict(dict)
for sp in sorted(glob.glob("outputs/finalcheck_20261004/speed/clips/*_g100_s8/speed_log.json")
                 + glob.glob("outputs/finalcheck_20261004/speed/clips/*_deliv_g100_T5pad/speed_log.json")):
    d = os.path.basename(os.path.dirname(sp))
    clip, label = d.split("_", 1)
    ref[clip]["speed:" + label] = json.load(open(sp))["init_md5"]
mine = defaultdict(dict)
for sp in sorted(glob.glob("outputs/more_20261004/fewer_steps/clips/*/speed_log.json")):
    d = os.path.basename(os.path.dirname(sp))
    clip, label = d.split("_", 1)
    mine[clip][label] = json.load(open(sp))["init_md5"]
L, nfail = [], 0
for clip in sorted(mine):
    refs = {k: v for k, v in ref[clip].items() if k.endswith("_s8")}
    uniq = {tuple(v) for v in refs.values()}
    if len(uniq) != 1:
        L.append(f"FAIL {clip}: reference 8-step renders missing or disagree: {sorted(refs)}")
        nfail += 1
        continue
    r = list(next(iter(uniq)))
    t5 = ref[clip].get("speed:deliv_g100_T5pad")
    L.append(f"{clip}: reference = {len(refs)} speed-lane eight-step renders agree ({len(r)} windows; w0 {r[0][:12]}..); "
             f"speed-lane deliv_g100_T5pad matches it: {t5 == r if t5 else 'n/a'}")
    for lab, fp in sorted(mine[clip].items()):
        same = [a == b for a, b in zip(fp, r)]
        if "pad" in lab:
            ok = len(fp) == len(r) and all(same)
            exp = "all windows same"
        else:
            ok = len(fp) == len(r) and same[0]
            exp = "w0 same (unpadded: later windows may differ)"
        nfail += (not ok)
        L.append(f"  {'PASS' if ok else 'FAIL'} {clip}_{lab:32s} windows {len(fp)}/{len(r)} identical={sum(same)} "
                 f"(expected: {exp})")
L.append(f"SUMMARY clips={len(mine)} failed={nfail}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
