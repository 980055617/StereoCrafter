#!/usr/bin/env python
"""EXPLORATORY (PREREG_ADDENDUM_4): a stripe-specific statistic.  Forward-splat stripes are thin VERTICAL cracks, so in
regions that are flat in the GT they raise horizontal differences but not vertical ones.  Statistic per row, S6
frames, GT-flat mask = reviewlib.regions(GT)['lo'] (gradient <= 50% quantile, same as stripeE):
    aniso = mean|dY/dx| / mean|dY/dy|  over GT-flat pixels;   reported as aniso(row) / aniso(GT).
Rows: GT (REG_FRAME), origin, s25, deliverable, COMP_origin, COMP_telea, BR_raw (holes black).  CPU only.
usage: python job_aniso_v1.py <out_json> <clip> [<clip> ...]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeplib as S  # noqa: E402

sys.path.insert(0, os.path.join(S.REPO, "scripts/distill/runs/review_20261001"))
import reviewlib as RL  # noqa: E402

OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
res = {}
for CLIP in sys.argv[2:]:
    js, _ = S.regjson(CLIP)
    S6 = list(js["frames"])[0::7][:6]
    D = S.load_clip(CLIP, S6, want_splat=True)
    holes = D["BLext"][:, S.MY:S.MY + S.TH, S.MX:S.MX + S.TW].astype(np.float32).mean(-1) > 127.5
    BR = np.ascontiguousarray(D["BRext"][:, S.MY:S.MY + S.TH, S.MX:S.MX + S.TW])
    GT = np.stack([S.box_crop(D["TR"][j], *S.reg_shift(js, fi, "REG_FRAME")) for j, fi in enumerate(S6)])
    rows = {r: S.render_right(S.row_path(CLIP, r), S6)[1] for r in ("origin", "s25", "deliverable")}
    rows["COMP_origin"] = np.where(holes[..., None], rows["origin"], BR)
    rows["COMP_telea"] = np.stack([S.inpaint_holes(BR[j], holes[j]) for j in range(len(S6))])
    rows["BR_raw"] = BR
    g = RL.gray(GT)
    lo = RL.regions(g)["lo"] & ~holes                      # GT-flat, outside model holes

    def aniso(x):
        y = RL.gray(x)
        dx = np.abs(np.diff(y, axis=2))[:, :-1, :]
        dy = np.abs(np.diff(y, axis=1))[:, :, :-1]
        m = lo[:, :-1, :-1]
        return float(dx[m].mean() / max(dy[m].mean(), 1e-9)), float(dx[m].mean()), float(dy[m].mean())
    a_gt = aniso(GT)
    e = {"GT": dict(aniso=a_gt[0], dx=a_gt[1], dy=a_gt[2])}
    for r, x in rows.items():
        a = aniso(x)
        e[r] = dict(aniso=a[0], rel=a[0] / a_gt[0], dx=a[1], dy=a[2])
    res[CLIP] = e
    print(CLIP, " ".join(f"{r}:{v['rel']:.3f}" for r, v in e.items() if r != "GT"), f"(GT aniso {a_gt[0]:.3f})", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
