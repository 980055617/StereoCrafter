#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- GATE D3 (PREREG.txt section 3): the consistency decoder with its fixed per-frame seed is
deterministic.  In a NEW process, window 0 of <clip>_<latlabel> (all 14 frames kept) is decoded twice; both uint8 results must
be identical to each other AND to frames 0..13 of the right half of this lane's cd re-decode render (written by an earlier
process, possibly on the other GPU).
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python cd_determinism_v1.py <out_txt> <clip> <latlabel>
"""
import hashlib, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dswap_lib as DL  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402
OUT, CLIP, LAT = sys.argv[1], sys.argv[2], sys.argv[3]
assert not os.path.exists(OUT), OUT
LD = f"/mnt/ssd_data/deep_20261004/decoder_ft/latents/{CLIP}_{LAT}"
REN = f"/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev/{CLIP}_{LAT}__cd/{CLIP}_inpainting_results_sbs.mkv"
n = len(VideoReader(f"video_data/splatting/{CLIP}_splatting_results.mp4", ctx=cpu(0)))
dec = DL.Decoder("cd")
fr = list(range(14))
a = DL.decode_frames(dec, LD, n, fr)
b = DL.decode_frames(dec, LD, n, fr)
vr = VideoReader(REN, ctx=cpu(0))
r = vr.get_batch(fr).asnumpy()
r = np.ascontiguousarray(r[:, :, r.shape[2] // 2:])
md = lambda x: hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
ab, ar = np.array_equal(a, b), np.array_equal(a, r)
lines = [f"cd seed {dec.seed}, {CLIP} {LAT} frames 0..13 (window 0), gpu {os.environ.get('CUDA_VISIBLE_DEVICES')}",
         f"run1 md5 {md(a)}  run2 md5 {md(b)}  render {REN} frames 0..13 md5 {md(r)}",
         f"run1 == run2: {ab}; run1 == render (other process): {ar}; max|run1-render| {int(np.abs(a.astype(int) - r.astype(int)).max())}",
         f"GATE D3 {'PASS' if (ab and ar) else 'FAIL'}"]
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
