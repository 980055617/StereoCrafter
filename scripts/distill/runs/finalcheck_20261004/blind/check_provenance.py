#!/usr/bin/env python
"""V1 + V2 of PREREG.txt: full-decode bit-identity of the 24 A/B source renders.

V1  md5 of the fully decoded SBS array == the sbs line of that render dir's writer_md5.txt
    (md5 of the pre-encode array the FFV1 writer received) and the shape matches.
V2  origin and deliverable LEFT halves are bit-identical on every frame of every clip
    (both are the splatting video's left-eye passthrough; proves the two renders are frame-aligned).

usage: python check_provenance.py OUTDIR   (OUTDIR must not exist)
CPU only.  Writes OUTDIR/provenance.json and OUTDIR/provenance.log; exit 1 on any failure.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blindlib as B  # noqa: E402
from decord import VideoReader, cpu  # noqa: E402

OUT = B.new_dir(sys.argv[1])
log = B.Tee(f"{OUT}/provenance.log")
res, fails = {}, []
for clip in B.CLIPS:
    arrs, row = {}, {}
    for lab in (B.LABEL_ORIGIN, B.LABEL_DELIV):
        p = B.render_path(clip, lab)
        t = time.time()
        vr = VideoReader(p, ctx=cpu(0))
        a = vr.get_batch(list(range(len(vr)))).asnumpy()
        del vr
        md5 = B.md5_array(a)
        rec = B.writer_md5_sbs(p)
        exp_md5, exp_shape = rec[-1] if rec else (None, None)
        ok = (md5 == exp_md5) and (str(tuple(a.shape)) == exp_shape)
        row[lab] = dict(path=p, decoded_md5=md5, decoded_shape=list(a.shape), writer_md5=exp_md5,
                        writer_shape=exp_shape, n_sbs_lines=len(rec), match=ok,
                        file_md5_sidecar=open(p + ".md5").read().split()[0] if os.path.exists(p + ".md5") else None)
        log(f"{clip} {lab:17s} decoded {a.shape} md5 {md5}  writer {exp_md5} {exp_shape} "
            f"(sbs lines={len(rec)})  {'MATCH' if ok else 'MISMATCH'}  [{time.time()-t:.1f}s]")
        if not ok:
            fails.append(f"V1 {clip} {lab}")
        arrs[lab] = a
    o, d = arrs[B.LABEL_ORIGIN], arrs[B.LABEL_DELIV]
    same_n = o.shape == d.shape
    left_eq = bool(same_n and (o[:, :, :B.TW] == d[:, :, :B.TW]).all())
    n_left_diff_frames = (int((o[:, :, :B.TW] != d[:, :, :B.TW]).reshape(o.shape[0], -1).any(1).sum())
                          if same_n else None)
    right_eq_frames = (int((o[:, :, B.TW:] == d[:, :, B.TW:]).reshape(o.shape[0], -1).all(1).sum())
                       if same_n else None)
    row["left_halves_bit_identical_all_frames"] = left_eq
    row["n_frames_left_differs"] = n_left_diff_frames
    row["n_frames_right_identical"] = right_eq_frames
    log(f"{clip} left halves bit-identical on all {o.shape[0]} frames: {left_eq}  "
        f"(frames differing {n_left_diff_frames});  right halves identical on {right_eq_frames} frames")
    if not left_eq:
        fails.append(f"V2 {clip}")
    res[clip] = row
    del arrs, o, d

res["_fails"] = fails
B.jdump(res, f"{OUT}/provenance.json")
log(f"V1/V2 {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
