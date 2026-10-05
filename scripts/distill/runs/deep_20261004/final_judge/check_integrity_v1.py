#!/usr/bin/env python
"""final_judge gate J1 (CPU only): for every render in ROWS_v1.json, full-decode md5 (decord, all frames, uint8, C order)
vs the sidecar <file>.mkv.md5 written by the lane, and md5 of the LEFT half (must be identical across rows of a clip).
usage: python check_integrity_v1.py <out_json (new)>"""
import hashlib, json, os, sys, time
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = sys.argv[1]
assert not os.path.exists(OUT), OUT
R = json.load(open("scripts/distill/runs/deep_20261004/final_judge/ROWS_v1.json"))
res = {}
T0 = time.time()
for c in R["clips"]:
    res[c] = {}
    for lab in R["labels"]:
        if lab not in R["cells"][c]:
            continue
        p = R["cells"][c][lab]["path"]
        if p.startswith("LEFTASRIGHT:"):
            continue
        vr = VideoReader(p, ctx=cpu(0))
        n = len(vr)
        hf, hl = hashlib.md5(), hashlib.md5()
        shape = None
        for s in range(0, n, 16):
            b = vr.get_batch(list(range(s, min(s + 16, n)))).asnumpy()
            shape = b.shape[1:]
            hf.update(np.ascontiguousarray(b).tobytes())
            hl.update(np.ascontiguousarray(b[:, :, : b.shape[2] // 2]).tobytes())
        side = p + ".md5"
        sidemd5 = open(side).read().split()[0] if os.path.exists(side) else None
        res[c][lab] = dict(path=p, n=n, shape=list(shape), md5_full=hf.hexdigest(), md5_left=hl.hexdigest(),
                           sidecar=sidemd5, sidecar_match=(sidemd5 == hf.hexdigest()) if sidemd5 else None)
        print(f"[{time.time()-T0:7.1f}s] {c} {lab:28s} n={n} md5 {hf.hexdigest()[:8]} side {str(sidemd5)[:8]} "
              f"match {res[c][lab]['sidecar_match']} left {hl.hexdigest()[:8]}", flush=True)
    lefts = {v["md5_left"] for v in res[c].values()}
    print(f"   {c}: distinct left-half md5 across {len(res[c])} renders: {len(lefts)}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("DONE", time.time() - T0)
