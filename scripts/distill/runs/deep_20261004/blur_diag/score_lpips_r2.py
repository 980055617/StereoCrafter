#!/usr/bin/env python
"""blur_diag (deep_20261004) -- LPIPS for every row of one clip (GPU).  Definitions: PREREG.txt M1, M2, gate G0.

  REG       LPIPS-alex(row, GT registered REG_FRAME)  -- score_clip_ll.py call pattern (batches of 4, *2-1)
            (VAE_GT*, RS_GTx: GT is their own input -> "against itself")
  UNREG     LPIPS-alex(row, GT at zero shift) for render-geometry rows (published definition)
  VALID_GT  spatial LPIPS map of (composite(row, GT), GT), mean over valid (non-dilated-hole) pixels
  VALID_BR  spatial LPIPS map of (composite(row, BR), BR), mean over valid pixels (registration-free input chain)
G0: ORIGIN / DELIV / S25 REG and UNREG must equal the eval_robustness JSON values within 1e-4.
usage: CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock python score_lpips_r2.py <out_dir> <clip>
"""
import json
import os
import sys
import time

import lpips
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blurlib_r2 as B  # noqa: E402

OUTD, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUTD, exist_ok=True)
OJ = f"{OUTD}/{CLIP}.json"
assert not os.path.exists(OJ), f"refusing to overwrite {OJ}"
T0 = time.time()
dev = "cuda"


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


M = B.meta(CLIP)
FR = M["frames"]
jp, J = B.er_json(CLIP)
paths = B.row_paths(CLIP)
GT = B.load_row(CLIP, "GT", FR)
GTU = np.load(f"{B.CACHE}/{CLIP}/GTunreg_sc.npy")
BR = B.load_row(CLIP, "BR", FR)
hole, dil = B.holes(CLIP, FR)
valid = ~dil
log(f"rows {sorted(paths)}; holes {hole.mean():.4f} dilated {dil.mean():.4f}")

net = lpips.LPIPS(net="alex").to(dev).eval()
net_sp = lpips.LPIPS(net="alex", spatial=True).to(dev).eval()


def t01(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).float() / 255.


@torch.no_grad()
def lp_scalar(x, ref):
    R, T = t01(x), t01(ref)
    out = []
    for i in range(0, len(R), 4):                       # score_clip_ll.py batching, verbatim
        o = net((R[i:i + 4].cuda() * 2 - 1), (T[i:i + 4].cuda() * 2 - 1))
        out += [float(v) for v in o.view(-1)]
    return out


@torch.no_grad()
def lp_valid(x, ref):
    xc = B.composite(x, ref, dil)
    R, T = t01(xc), t01(ref)
    V = torch.from_numpy(valid).float()
    out = []
    for i in range(0, len(R), 4):
        m = net_sp((R[i:i + 4].cuda() * 2 - 1), (T[i:i + 4].cuda() * 2 - 1))[:, 0]
        v = V[i:i + 4].cuda()
        out += [float(a) for a in ((m * v).sum(dim=(1, 2)) / v.sum(dim=(1, 2)))]
    return out


res = {}
left_md5 = {}
ROWS = ["GT"] + [r for r in ["VAE_GT", "VAE_GT32", "VAE_GTx", "RS_GTx", "RS_GTx_L", "BR", "VAE_BR", "ORIGIN", "DELIV", "S25",
                             "T5NAT", "T5PAD", "HIRES_A", "HIRES_B", "HIRES_B_L"] if r in paths or r == "BR"]
for row in ROWS:
    x = GT if row == "GT" else (BR if row == "BR" else B.load_row(CLIP, row, FR, paths))
    assert x.shape == GT.shape, (row, x.shape)
    d = dict(REG=lp_scalar(x, GT), VALID_GT=lp_valid(x, GT))
    if row not in B.GT_GEOMETRY:
        d["UNREG"] = lp_scalar(x, GTU)
        if row != "BR":
            d["VALID_BR"] = lp_valid(x, BR)
    if row in paths and paths[row][0] == "sbs":
        import hashlib
        left_md5[row] = hashlib.md5(B.load_left(CLIP, row, FR, paths).tobytes()).hexdigest()
    res[row] = dict(path=paths[row][1] if row in paths else f"cache:{row}", perframe=d,
                    clip={k: float(np.mean(v)) for k, v in d.items()})
    log(f"{row:9s} " + "  ".join(f"{k} {np.mean(v):.4f}" for k, v in d.items()))

# ---------------------------------------------------------------------------------------- G0
g0 = {}
for row, lab in (("ORIGIN", "origin_ll"), ("DELIV", "mstudent2_step800_deliv_ll"), ("S25", "s25_ll")):
    pub = J["configs"][lab]["lpips_clip"]
    assert res[row]["path"] == J["configs"][lab]["path"]
    du = res[row]["clip"]["UNREG"] - pub["UNREG"]
    dr = res[row]["clip"]["REG"] - pub["REG_FRAME"]
    g0[row] = dict(unreg=res[row]["clip"]["UNREG"], unreg_pub=pub["UNREG"], d_unreg=du,
                   reg=res[row]["clip"]["REG"], reg_pub=pub["REG_FRAME"], d_reg=dr,
                   ok=bool(abs(du) <= 1e-4 and abs(dr) <= 1e-4))
G0 = all(v["ok"] for v in g0.values())
lefts = {k: v for k, v in left_md5.items() if k in ("ORIGIN", "DELIV", "S25", "T5NAT", "T5PAD", "HIRES_A", "HIRES_B", "HIRES_B_L")}
left_ident = len(set(lefts.values())) <= 1
log(f"G0 {'PASS' if G0 else 'FAIL'} " + "  ".join(f"{k} dU {v['d_unreg']:+.1e} dR {v['d_reg']:+.1e}" for k, v in g0.items())
    + f"; left halves identical across sbs rows: {left_ident}")
json.dump(dict(clip=CLIP, er_json=jp, frames=FR, n=len(FR), hole_frac=float(hole.mean()),
               dil_hole_frac=float(dil.mean()), rows=res, G0=dict(pass_=G0, rows=g0),
               left_md5=left_md5, left_identical=left_ident, seconds=time.time() - T0), open(OJ, "w"))
log(f"wrote {OJ}")
print("CLIP_DONE", CLIP, flush=True)
