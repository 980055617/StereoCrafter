#!/usr/bin/env python
"""blur_diag G3: the identity-mode render (max_profile_chunks=1 -> 14 frames) must equal the first 14 frames of the
existing origin_ll render bit-exactly (both halves).  CPU only.
usage: python check_identity_r2.py <new_sbs.mkv> <reference_sbs.mkv>   -> prints G3_IDENTITY PASS|FAIL, exit 0|1"""
import sys

import numpy as np
from decord import VideoReader, cpu

new, ref = sys.argv[1], sys.argv[2]
a = VideoReader(new, ctx=cpu(0))
b = VideoReader(ref, ctx=cpu(0))
n = len(a)
A = a.get_batch(list(range(n))).asnumpy()
Bv = b.get_batch(list(range(n))).asnumpy()
assert A.shape == Bv.shape, (A.shape, Bv.shape)
d = np.abs(A.astype(np.int16) - Bv.astype(np.int16))
w = A.shape[2] // 2
ok = int(d.max()) == 0
print(f"frames {n} shape {A.shape}: left max|d| {d[:, :, :w].max()} right max|d| {d[:, :, w:].max()} "
      f"right mean|d| {d[:, :, w:].mean():.6f} right frac!=0 {(d[:, :, w:] > 0).mean():.6f}")
print(f"G3_IDENTITY {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
