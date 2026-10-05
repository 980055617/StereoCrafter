#!/usr/bin/env python
"""scale_gt lane -- GT validity check for every clip this lane touches (train pool, dev, test).

A clip is GT-VALID iff ALL of:
  (a) int(clip) < 310                                   (hard rule: 0310-0319 have no bundle, 0320-0365 are broken symlinks)
  (b) video_data/train/<clip>_train.mp4 exists and readlink -f does NOT point into train_leftGT_broken
  (c) the real right eye (TR quadrant) differs from the left eye (TL quadrant): MAD(TR,TL) > 2.0/255 at frame 0 AND at
      the middle frame (a broken bundle's TR is a codec copy of TL, MAD ~0-1)
  (d) video_data/splatting/<clip>_splatting_results.mp4 exists, same quadrant size as the train tile
Also records frame counts (train tile vs splatting) and the split role.
usage: python check_clips_v1.py <out_json>
"""
import json, os, sys, time
import numpy as np
from decord import VideoReader, cpu

REPO = "/home/kawa/master_project/StereoCrafter"; os.chdir(REPO)
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
split = json.load(open("scripts/distill/splits/fulldata_v1.json"))
dev, test, order, meta = split["dev"], split["test"], split["train_order"], split["clips"]
pool = [c for c in order if c not in dev and c not in test]
clips = sorted(set(pool) | set(dev) | set(test))
res = {}
t0 = time.time()
for c in clips:
    r = dict(clip=c, role=("dev" if c in dev else "test" if c in test else "train"), split_frames=meta[c]["frames"],
             split_gt=meta[c]["gt"], fmt=meta[c]["fmt"])
    p = f"video_data/train/{c}_train.mp4"
    r["lt310"] = int(c) < 310
    r["exists"] = os.path.exists(p)
    r["realpath"] = os.path.realpath(p) if r["exists"] else None
    r["broken_link"] = bool(r["realpath"] and "train_leftGT_broken" in r["realpath"])
    r["is_symlink"] = os.path.islink(p)
    ok = r["lt310"] and r["exists"] and not r["broken_link"]
    if r["exists"]:
        try:
            vr = VideoReader(p, ctx=cpu(0)); n = len(vr); r["train_frames"] = n
            mads = []
            for fi in (0, n // 2):
                f = vr[fi].asnumpy(); H, W = f.shape[0] // 2, f.shape[1] // 2
                mads.append(float(np.abs(f[:H, W:2 * W].astype(np.float32) - f[:H, :W].astype(np.float32)).mean()))
            r["tile"] = [int(f.shape[0]), int(f.shape[1])]; r["mad_tr_tl_255"] = mads
            r["right_differs"] = bool(min(mads) > 2.0)
            sp = f"video_data/splatting/{c}_splatting_results.mp4"
            r["splat_exists"] = os.path.exists(sp)
            if r["splat_exists"]:
                vs = VideoReader(sp, ctx=cpu(0)); s0 = vs[0].asnumpy()
                r["splat_frames"] = len(vs); r["splat_tile"] = [int(s0.shape[0]), int(s0.shape[1])]
                r["same_quadrant"] = (s0.shape[0] // 2, s0.shape[1] // 2) == (H, W)
            ok = ok and r["right_differs"] and r["splat_exists"] and r.get("same_quadrant", False)
        except Exception as e:  # unreadable
            r["error"] = repr(e); ok = False
    r["gt_valid"] = bool(ok)
    res[c] = r
    print(f"{c} {r['role']:5s} valid={r['gt_valid']} lt310={r['lt310']} link={r['is_symlink']} broken={r['broken_link']} "
          f"mad={r.get('mad_tr_tl_255')} frames t/s={r.get('train_frames')}/{r.get('splat_frames')} ({time.time()-t0:.0f}s)", flush=True)
summ = dict(
    n_checked=len(res),
    train_pool=len(pool), train_valid=[c for c in pool if res[c]["gt_valid"]],
    train_invalid=[c for c in pool if not res[c]["gt_valid"]],
    dev_valid=[c for c in dev if res[c]["gt_valid"]], dev_invalid=[c for c in dev if not res[c]["gt_valid"]],
    test_valid=[c for c in test if res[c]["gt_valid"]], test_invalid=[c for c in test if not res[c]["gt_valid"]],
    lt310_but_invalid=[c for c in res if res[c]["lt310"] and not res[c]["gt_valid"]],
    train_frames_ne_splat=[c for c in res if res[c].get("train_frames") != res[c].get("splat_frames")],
)
summ["n_train_valid"] = len(summ["train_valid"])
json.dump(dict(summary=summ, clips=res), open(OUT, "w"), indent=1)
print("SUMMARY", json.dumps({k: (v if not isinstance(v, list) or len(v) < 30 else f"{len(v)} items") for k, v in summ.items()}), flush=True)
