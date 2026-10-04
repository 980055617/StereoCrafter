"""Shared constants/helpers for the finalcheck_20261004 "blind" lane (blind A/B rating material).

Reuses scripts/distill/runs/review_20261001/reviewlib.py UNMODIFIED for frame access, the corrected
right-eye geometry (top-right quadrant), the splat mask and register_gt.  The only thing done to it is
an in-process update of its OFFSETS dict with the scorer's per-clip (dy,dx) for the six clips it lacks
(values = the ROW lines of score_clip_ll.py for origin_ll / mstudent2_step800_deliv_ll, identical for
both configs).  No tracked file is modified.
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
sys.path.insert(0, f"{REPO}/scripts/distill/runs/review_20261001")
import reviewlib as R  # noqa: E402  (chdir()s to REPO)

LANE_OUT = f"{REPO}/outputs/finalcheck_20261004/blind"
LANE_SCRIPTS = f"{REPO}/scripts/distill/runs/finalcheck_20261004/blind"

CLIPS = ["0042", "0052", "0125", "0128", "0141", "0147",
         "0170", "0204", "0225", "0251", "0259", "0301"]

# score_clip_ll.py ROW values (dy,dx) -- grep '^ROW clip=' in the beyond_distil_mamba_scaled / beyond4 logs
SCORER_OFFSETS = {
    "0042": (-12, -12), "0052": (-12, -12), "0125": (-12, -12), "0128": (-12, -12),
    "0141": (-12, -12), "0147": (-12, -12),
    "0170": (-28, 0), "0204": (-28, 0), "0225": (-28, 0), "0251": (-28, 0),
    "0259": (-28, 0), "0301": (-28, 0),
}
for _c, _o in SCORER_OFFSETS.items():
    if _c in R.OFFSETS:
        assert tuple(R.OFFSETS[_c]) == tuple(_o), (_c, R.OFFSETS[_c], _o)
R.OFFSETS.update(SCORER_OFFSETS)

TH, TW, S = R.TH, R.TW, R.CROP          # 576, 1024, 384
assert (TH, TW, S) == (576, 1024, 384)

LABEL_ORIGIN = "origin@1.01"
LABEL_DELIV = "deliverable@1.01"


def render_path(clip, which):
    if which == LABEL_ORIGIN:
        p = f"outputs/beyond4_lossless/clips/{clip}_origin_ll/{clip}_inpainting_results_sbs.mkv"
    elif which == LABEL_DELIV:
        p = (f"outputs/beyond_distil_mamba_scaled/clips/{clip}_mstudent2_step800_deliv_ll/"
             f"{clip}_inpainting_results_sbs.mkv")
    else:
        raise KeyError(which)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def writer_md5_sbs(render_file):
    """All sbs lines of the dir's writer_md5.txt, as [(md5, shape_str)] in file order."""
    d = os.path.dirname(render_file)
    out = []
    for ln in open(os.path.join(d, "writer_md5.txt")):
        parts = ln.split()
        if not parts:
            continue
        if parts[-1].endswith("_sbs.mp4") or parts[-1].endswith("_sbs.mkv"):
            shape = ln[ln.index("("):ln.index(")") + 1]
            out.append((parts[0], shape))
    return out


def md5_array(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def train_path(clip):
    return f"video_data/train/{clip}_train.mp4"


def splat_path(clip):
    return f"video_data/splatting/{clip}_splatting_results.mp4"


def nvalid(clip):
    ns = {"train": R.nframes(train_path(clip)), "splat": R.nframes(splat_path(clip)),
          LABEL_ORIGIN: R.nframes(render_path(clip, LABEL_ORIGIN)),
          LABEL_DELIV: R.nframes(render_path(clip, LABEL_DELIV))}
    return min(ns.values()), ns


def band(nv):
    return nv // 3, 2 * nv // 3, nv // 2


def global_shift(TR, BR, mask, t0, l0, H, W):
    """build_review.py's whole-window registration of the real right eye (TR) to the warped input (BR).

    Returns (ddy, ddx, psnr_best, psnr_at_zero)."""
    tgt = BR[t0:t0 + TH, l0:l0 + TW]
    valid = ~mask
    p0 = R.psnr_u8(TR[t0:t0 + TH, l0:l0 + TW], tgt, valid)
    best = (-1e9, 0, 0)
    for st in (2, 1):
        cy, cx = best[1], best[2]
        yr = range(-10, 11, 2) if st == 2 else range(cy - 2, cy + 3)
        xr = range(-140, 61, 2) if st == 2 else range(cx - 3, cx + 4)
        for a in yr:
            for b in xr:
                tt, ll = t0 + a, l0 + b
                if tt < 0 or ll < 0 or tt + TH > H or ll + TW > W:
                    continue
                pv = R.psnr_u8(TR[tt:tt + TH, ll:ll + TW], tgt, valid)
                if pv > best[0]:
                    best = (pv, a, b)
    return best[1], best[2], best[0], p0


def overlap_frac(a, b, s=S):
    (y1, x1), (y2, x2) = a, b
    oy = max(0, min(y1 + s, y2 + s) - max(y1, y2))
    ox = max(0, min(x1 + s, x2 + s) - max(x1, x2))
    return oy * ox / float(s * s)


def integral(a):
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    ii[1:, 1:] = a.astype(np.float64).cumsum(0).cumsum(1)
    return ii


def box(ii, y, x, s=S):
    return ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]


# the gradient integral image is (TH-1, TW-1) -> keep every window inside it (build_review.py)
GRID_Y = list(range(0, TH - S, 8))
GRID_X = list(range(0, TW - S, 8))


def grad_energy(gt_u8):
    g = gt_u8.astype(np.float32).mean(axis=2)
    return np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]


def psnr(a, b, m=None):
    return R.psnr_u8(a, b, m)


def _np_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def jdump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=_np_default)


def new_dir(path):
    """Every run writes a NEW directory -- refuse to reuse one."""
    if os.path.exists(path):
        raise SystemExit(f"refusing to write into existing directory {path}")
    os.makedirs(path)
    return path


class Tee:
    def __init__(self, path):
        self.fh = open(path, "w")

    def __call__(self, *s):
        print(*s, flush=True)
        print(*s, file=self.fh, flush=True)


def finite(x):
    return x if (isinstance(x, (int, float)) and math.isfinite(x)) else None
