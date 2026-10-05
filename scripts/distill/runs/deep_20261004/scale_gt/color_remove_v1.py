#!/usr/bin/env python
"""scale_gt DIAGNOSTIC (never used for selection): colour-removal sensitivity.
For a fine-tuned render <clip>_<label> and a reference render (origin at the same step count), subtract the per-clip global
mean-RGB difference of the RIGHT halves (fine-tuned minus reference, over ALL frames) from the fine-tuned right half, round,
clip to [0,255], and write a new lossless FFV1 sbs (left half byte-identical) to <out_root>/<clip>_<label>_colrm/.
The reference is the ORIGIN render, never the GT, so no GT information enters.  Then score it with score_clip_ll.py.
usage: python color_remove_v1.py <out_root> <clip>:<label_path>:<ref_path> [...]
"""
import hashlib, importlib.util, os, sys
import numpy as np
from decord import VideoReader, cpu
REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
spec = importlib.util.spec_from_file_location("il", f"{REPO}/scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py")
IL = importlib.util.module_from_spec(spec); spec.loader.exec_module(IL)
OUT = sys.argv[1]
for item in sys.argv[2:]:
    clip, p, ref = item.split(":")
    lab = os.path.basename(os.path.dirname(p))[len(clip) + 1:]
    od = os.path.join(OUT, f"{clip}_{lab}_colrm"); assert not os.path.exists(od), od
    v = VideoReader(p, ctx=cpu(0)); a = v.get_batch(list(range(len(v)))).asnumpy()
    r = VideoReader(ref, ctx=cpu(0)); b = r.get_batch(list(range(len(r)))).asnumpy()
    h = a.shape[2] // 2
    assert a.shape == b.shape and np.array_equal(a[:, :, :h], b[:, :, :h]), "left halves differ / shape mismatch"
    d = a[:, :, h:].reshape(-1, 3).astype(np.float64).mean(0) - b[:, :, h:].reshape(-1, 3).astype(np.float64).mean(0)
    out = a.copy()
    out[:, :, h:] = np.clip(np.rint(a[:, :, h:].astype(np.float64) - d), 0, 255).astype(np.uint8)
    os.makedirs(od)
    fn = os.path.join(od, f"{clip}_inpainting_results_sbs.mkv")
    IL._ffv1_write(out, float(v.get_avg_fps()), fn)
    with open(os.path.join(od, "colrm.txt"), "w") as fh:
        fh.write(f"src {p}\nref {ref}\nsubtracted_mean_rgb_diff {d.round(4).tolist()}\nmd5 {hashlib.md5(out.tobytes()).hexdigest()}\n")
    print(f"{clip} {lab}: subtracted {d.round(3).tolist()} -> {fn}", flush=True)
