#!/usr/bin/env python
"""S38 job (PREREG.txt): gate G2 (CPU LPIPS reproduces eval_robustness UNREG / REG_FRAME), M2 copy-left rows,
M5 seed spread (T5nat vs T5pad), and per-row sharpness.  One clip per call.  CPU only.
usage: python job_s38_v1.py <out_dir> <clip>"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeplib as S  # noqa: E402

torch.set_num_threads(int(os.environ.get("SK_THREADS", "6")))
OUT, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
oj = os.path.join(OUT, f"{CLIP}.json")
assert not os.path.exists(oj), f"refusing to overwrite {oj}"
T0 = time.time()


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


js, jpath = S.regjson(CLIP)
frames = list(js["frames"])
D = S.load_clip(CLIP, frames, want_splat=True)
log("decoded bundle boxes", D["TR"].shape)
lp = S.Lp("alex")

rows = {}
L_half = None
for r in ["origin", "AYS8", "deliverable", "s25", "T5nat", "T5pad"]:
    p = S.row_path(CLIP, r)
    Lh, Rh = S.render_right(p, frames)
    if L_half is None:
        L_half = Lh
    else:
        assert np.array_equal(L_half, Lh), f"left halves differ for {r}"
    rows[r] = Rh
log("renders loaded")

gt = {}
for var in ["UNREG", "REG_FRAME"]:
    gt[var] = np.stack([S.box_crop(D["TR"][j], *S.reg_shift(js, fi, var)) for j, fi in enumerate(frames)])

res = dict(clip=CLIP, family=S.family(CLIP), frames=frames, regjson=jpath, rows={})
# ---- G2 + reference rows
ays_json = f"outputs/ays_20261004/robust/score_reg_v1/{CLIP}.json"
ays_wide = "outputs/ays_20261004/robust/score_reg_v1_wide/0125.json"
ajs = json.load(open(ays_wide if (CLIP == "0125" and os.path.exists(ays_wide)) else ays_json)) \
    if os.path.exists(ays_json) else None
for r, Rh in rows.items():
    e = {}
    for var in ["UNREG", "REG_FRAME"]:
        v = lp(Rh, gt[var])
        e[f"lpips_{var}"] = float(np.mean(v))
        e[f"lpips_{var}_frames"] = v
        e[f"psnr_{var}"] = float(10 * np.log10(1.0 / max(np.mean([((Rh[j].astype(np.float64) - gt[var][j]) / 255.) .__pow__(2).mean() for j in range(len(frames))]), 1e-12)))
    lab = S.ROWLAB[r]
    pub = js["configs"].get(lab, {}).get("lpips_clip") if lab in js["configs"] else None
    if pub is None and ajs is not None and lab in ajs["configs"]:
        pub = ajs["configs"][lab]["lpips_clip"]
    if pub is not None:
        e["pub_UNREG"], e["pub_REG_FRAME"] = pub["UNREG"], pub["REG_FRAME"]
        e["dev_UNREG"] = e["lpips_UNREG"] - pub["UNREG"]
        e["dev_REG_FRAME"] = e["lpips_REG_FRAME"] - pub["REG_FRAME"]
    e["sharp"] = S.sharp_score(Rh)
    res["rows"][r] = e
    log(f"{r:12s} UNREG {e['lpips_UNREG']:.5f} REG {e['lpips_REG_FRAME']:.5f}  dev "
        f"{e.get('dev_UNREG', float('nan')):+.1e} {e.get('dev_REG_FRAME', float('nan')):+.1e}  sharp {e['sharp']:.5f}")
res["gtSharp_UNREG"] = S.sharp_score(gt["UNREG"])
res["gtSharp_REG"] = S.sharp_score(gt["REG_FRAME"])
res["leftSharp"] = S.sharp_score(L_half)
holes = (D["BLext"][:, S.MY:S.MY + S.TH, S.MX:S.MX + S.TW].astype(np.float32).mean(-1) > 127.5)
res["hole_frac"] = [float(h.mean()) for h in holes]
assert np.allclose(res["hole_frac"], js["hole_frac"], atol=2e-3), "hole fraction differs from the stored JSON"

# ---- M2 copy-left rows
cl = {}
for var in ["UNREG", "REG_FRAME"]:
    v = lp(L_half, gt[var])
    cl[f"lpips_{var}"] = float(np.mean(v)); cl[f"lpips_{var}_frames"] = v
opt_sh, opt_ps, gto = [], [], []
for j, fi in enumerate(frames):
    dy, dx, ps = S.best_shift_psnr(D["TR"][j], L_half[j], (-4, 4), (-80, 16))
    opt_sh.append((dy, dx)); opt_ps.append(ps)
    gto.append(S.box_crop(D["TR"][j], dy, dx))
gto = np.stack(gto)
v = lp(L_half, gto)
cl["lpips_OPT"] = float(np.mean(v)); cl["lpips_OPT_frames"] = v
cl["opt_shift"] = opt_sh; cl["opt_psnr"] = opt_ps
cl["psnr_OPT"] = float(np.mean(opt_ps))
# origin scored against the same left-registered GT (for a like-for-like contrast on near-mono clips)
v = lp(rows["origin"], gto)
cl["origin_vs_GTleftreg"] = float(np.mean(v))
res["copyleft"] = cl
log(f"copy-left UNREG {cl['lpips_UNREG']:.4f} REG {cl['lpips_REG_FRAME']:.4f} OPT {cl['lpips_OPT']:.4f} "
    f"(median shift {np.median([s[1] for s in opt_sh]):+.0f}px, PSNR {cl['psnr_OPT']:.2f}); origin REG "
    f"{res['rows']['origin']['lpips_REG_FRAME']:.4f}")

# ---- M5 seed spread (window_k >= 1 only)
wk = js["window_k"]
sel = [j for j in range(len(frames)) if wk[j] >= 1]
A, B = rows["T5nat"][sel], rows["T5pad"][sel]
v = lp(A, B)
smap = lp.spatial(A, B)
hs = holes[sel]
mass_hole = float((smap * hs).sum() / smap.sum())
ps = [S.psnr_u8(A[k], B[k]) for k in range(len(sel))]
res["seed"] = dict(n=len(sel), lpips_T5nat_T5pad=float(np.mean(v)), frames=v, psnr=float(np.mean(ps)),
                   spatial_mean=float(smap.mean()), mass_in_holes=mass_hole, hole_area=float(hs.mean()),
                   lpips_in_holes=float((smap * hs).sum() / max(hs.sum(), 1)),
                   lpips_out_holes=float((smap * ~hs).sum() / max((~hs).sum(), 1)),
                   dGT_REG=float(abs(np.mean([res["rows"]["T5nat"]["lpips_REG_FRAME_frames"][j] for j in sel]) -
                                     np.mean([res["rows"]["T5pad"]["lpips_REG_FRAME_frames"][j] for j in sel]))))
log(f"seed: LPIPS(T5nat,T5pad) {res['seed']['lpips_T5nat_T5pad']:.4f} PSNR {res['seed']['psnr']:.2f} on {len(sel)} frames; "
    f"mass in holes {mass_hole:.3f} (area {hs.mean():.3f})")

# ---- origin spatial decomposition (holes vs rest), REG_FRAME
om = lp.spatial(rows["origin"], gt["REG_FRAME"])
res["origin_spatial"] = dict(mean=float(om.mean()), mass_in_holes=float((om * holes).sum() / om.sum()),
                             hole_area=float(holes.mean()),
                             lpips_in_holes=float((om * holes).sum() / max(holes.sum(), 1)),
                             lpips_out_holes=float((om * ~holes).sum() / max((~holes).sum(), 1)))
res["seconds"] = time.time() - T0
json.dump(res, open(oj, "w"))
log("wrote", oj)
print("CLIP_DONE", CLIP, flush=True)
