#!/usr/bin/env python
"""final_judge: input_side P1 reproduction (RF vs R1, both models, 4 clips) -- my re-run of that lane's own scorer vs its
published JSONs (outputs/deep_20261004/input_side/scores_v1/<clip>__<input>.json).  CPU.  usage: python compare_inputside_v1.py <out_txt>"""
import json, os, sys
import numpy as np
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
assert not os.path.exists(OUT)
CL = ["0301", "0204", "0052", "0147"]
L = []
mine = {(c, i): json.load(open(f"outputs/deep_20261004/final_judge/inputside_repro_v1/{c}__{i}.json")) for c in CL for i in ("R1", "RF")}
pub = {(c, i): json.load(open(f"outputs/deep_20261004/input_side/scores_v1/{c}__{i}.json")) for c in CL for i in ("R1", "RF")}
mx = 0.0
for k in mine:
    for lab in ("origin", "deliv"):
        for v in ("UNREG", "REG_FRAME", "REG_CLIP", "BLK_LOCAL"):
            a = mine[k]["configs"][lab]["lpips_clip"][v]; b = pub[k]["configs"][lab]["lpips_clip"][v]
            mx = max(mx, abs(a - b))
L.append(f"input_side reproduction: max |mine - published| over 4 clips x 2 inputs x 2 models x 4 variants = {mx:.2e}")
for lab in ("origin", "deliv"):
    for v in ("REG_FRAME", "UNREG", "BLK_LOCAL"):
        d = [mine[(c, "RF")]["configs"][lab]["lpips_clip"][v] - mine[(c, "R1")]["configs"][lab]["lpips_clip"][v] for c in CL]
        L.append(f"  {lab:6s} d{v:9s} (RF - R1) " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(CL, d)) +
                 f"  mean {np.mean(d):+.4f}  improved {sum(x < 0 for x in d)}/4")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
