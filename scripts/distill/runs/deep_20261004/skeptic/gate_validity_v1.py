#!/usr/bin/env python
"""G0 validity gate (PREREG.txt): id < 0310, train bundle not a link into train_leftGT_broken, DC-removed
MAD(train TL, train TR) > 5.0 on the deployed window at frames 8..19.  CPU only.
usage: python gate_validity_v1.py <out_json> <clip> [<clip> ...]   (refuses to overwrite)"""
import json, os, sys
import numpy as np
from decord import VideoReader, cpu
os.chdir("/home/kawa/master_project/StereoCrafter")
TH, TW = 576, 1024
out = sys.argv[1]
assert not os.path.exists(out), f"refusing to overwrite {out}"
res = {}
for clip in sys.argv[2:]:
    p = f"video_data/train/{clip}_train.mp4"
    real = os.path.realpath(p)
    r = dict(real=real, id_ok=int(clip) < 310, link_ok="train_leftGT_broken" not in real, exists=os.path.exists(p))
    if r["exists"]:
        vr = VideoReader(p, ctx=cpu(0))
        f0 = vr[0].asnumpy(); H, W = f0.shape[0] // 2, f0.shape[1] // 2
        t0, l0 = (H // 128 * 128 - TH) // 2, (W // 128 * 128 - TW) // 2
        idx = [i for i in range(8, 20) if i < len(vr)]
        b = vr.get_batch(idx).asnumpy().astype(np.float32)
        TL = b[:, t0:t0 + TH, l0:l0 + TW]; TR = b[:, t0:t0 + TH, W + l0:W + l0 + TW]
        d = TL - TR
        mad = float(np.mean([np.abs(x - x.mean(axis=(0, 1))).mean() for x in d]))
        r.update(H=H, W=W, n=len(vr), mad_TL_TR=mad, mad_ok=mad > 5.0)
    r["valid"] = bool(r["id_ok"] and r["link_ok"] and r.get("mad_ok", False))
    res[clip] = r
    print(clip, json.dumps(r), flush=True)
json.dump(res, open(out, "w"), indent=1)
print("VALID", sum(v["valid"] for v in res.values()), "/", len(res))
