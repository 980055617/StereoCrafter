#!/usr/bin/env python
"""Independent re-verification of beyond4's lossless-writer faithfulness chain.
CLAIM A: decord's decode of the FFV1 .mkv equals the pre-encode array md5 recorded by the writer.
CLAIM B: the .mkv's LEFT half is bit-identical to the splatting input's TOP-LEFT quadrant.
usage: verify_ll_v1.py <clip> <mkv> <expected_md5>
"""
import sys, hashlib
import numpy as np
from decord import VideoReader, cpu

clip, mkv, exp = sys.argv[1], sys.argv[2], sys.argv[3]
vr = VideoReader(mkv, ctx=cpu(0))
a = vr.get_batch(list(range(len(vr)))).asnumpy()
a = np.ascontiguousarray(a)
got = hashlib.md5(a.tobytes()).hexdigest()
print(f"[{clip}] CLAIM_A decord({mkv}) md5={got} expected={exp} MATCH={got==exp} shape={a.shape}")

sp = f"video_data/splatting/{clip}_splatting_results.mp4"
vs = VideoReader(sp, ctx=cpu(0))
s = vs.get_batch(list(range(len(vs)))).asnumpy()
print(f"[{clip}] splatting tile shape={s.shape}")
T, H, W, _ = s.shape
h, w = H // 2, W // 2
tl = s[:, :h, :w]                      # top-left quadrant = the left eye
n = min(len(a), len(tl))
oh, ow = a.shape[1], a.shape[2] // 2
L = a[:n, :, :ow]
# the pipeline centre-crops/resizes; compare against a centre crop of the same size when shapes differ
if (oh, ow) == (h, w):
    ref = tl[:n]
else:
    t0 = (h - oh) // 2; l0 = (w - ow) // 2
    ref = tl[:n, t0:t0+oh, l0:l0+ow] if (h >= oh and w >= ow) else None
if ref is None or ref.shape != L.shape:
    print(f"[{clip}] CLAIM_B SKIPPED: out-left {L.shape} vs splatting-TL {tl.shape} not comparable by centre crop")
else:
    d = L.astype(np.int32) - ref.astype(np.int32)
    nd = int((d != 0).sum())
    print(f"[{clip}] CLAIM_B left-half vs splatting top-left: bytesdiff={nd}/{d.size} "
          f"maxabs={int(np.abs(d).max())} {'BIT_IDENTICAL' if nd==0 else 'DIFFERS'}")
