#!/usr/bin/env python
"""M3 oracles + M1 camera differences (test clips) + M4 unobservable share, S10 frames, one clip per call.
v2 = v1 + M4 U3 (view-dependent proxy) + exploratory input-fidelity rows LPIPS(row, BR) (PREREG_ADDENDUM_2).
Definitions: PREREG.txt.  CPU only (RAFT-large on CPU).
usage: python job_m3_v1.py <out_dir> <clip>          env SK_NFR (default 10) limits frames for smoke tests"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeplib as S  # noqa: E402

torch.set_num_threads(int(os.environ.get("SK_THREADS", "6")))
cv2.setNumThreads(int(os.environ.get("SK_THREADS", "6")))
OUT, CLIP = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
oj = os.path.join(OUT, f"{CLIP}.json")
assert not os.path.exists(oj), f"refusing to overwrite {oj}"
T0 = time.time()
TH, TW, MY, MX = S.TH, S.TW, S.MY, S.MX


def log(*a):
    print(f"[{CLIP} {time.time() - T0:6.1f}s]", *a, flush=True)


js, jpath = S.regjson(CLIP)
s38 = list(js["frames"])
frames = s38[0::4][:int(os.environ.get("SK_NFR", "10"))]
D = S.load_clip(CLIP, frames, want_splat=True)
log("decoded", frames)
rows = {}
Lhalf = None
for r in ["origin", "AYS8", "deliverable", "s25"]:
    Lh, Rh = S.render_right(S.row_path(CLIP, r), frames)
    rows[r] = Rh
    Lhalf = Lh if Lhalf is None else Lhalf
# v2 sources (crf-12 cameras) at the deployed window
t0, l0 = D["t0"], D["l0"]
v2 = {}
for eye in ["left", "right"]:
    vr = VideoReader(f"video_data/{eye}_eye_v2/{CLIP}.mp4", ctx=cpu(0))
    b = vr.get_batch(frames).asnumpy()
    assert b.shape[1:3] == (D["H"], D["W"]), (b.shape, D["H"], D["W"])
    v2[eye] = b[:, t0:t0 + TH, l0:l0 + TW].copy()
    del b
log("renders + v2 loaded")

per = []
store = {k: [] for k in ["TRg", "A", "consA", "B", "C", "hole", "K1", "K2", "BR", "Yrt"]}
for j, fi in enumerate(frames):
    ddy, ddx = S.reg_shift(js, fi, "REG_FRAME")
    TRg_ext = np.ascontiguousarray(S.box_crop(D["TR"][j], ddy, ddx, my=MY, mx=MX))
    TL_ext = np.ascontiguousarray(S.box_crop(D["TLsplat"][j], 0, 0, my=MY, mx=MX))
    BR_ext = D["BRext"][j]
    hole_ext = D["BLext"][j].astype(np.float32).mean(-1) > 127.5
    BR_inp = S.inpaint_holes(BR_ext, hole_ext)
    # model's left input at the window must equal the render's left half
    lh_dev = float(np.abs(TL_ext[MY:MY + TH, MX:MX + TW].astype(np.int16) - Lhalf[j].astype(np.int16)).mean())
    F_GL = S.flow(TRg_ext, TL_ext); F_LG = S.flow(TL_ext, TRg_ext)
    F_BG = S.flow(BR_inp, TRg_ext); F_GB = S.flow(TRg_ext, BR_inp)
    cGL = S.consistency(F_GL, F_LG)          # on the TRg grid
    cBG = S.consistency(F_BG, F_GB)          # on the BR grid
    cGB = S.consistency(F_GB, F_BG)          # on the TRg grid
    c = (slice(MY, MY + TH), slice(MX, MX + TW))
    TRg = TRg_ext[c]
    A = S.q8(S.warp(TL_ext, F_GL))[c]
    consA = cGL[c]
    Afill = np.where(consA[..., None], A, TRg)
    Bw = S.q8(S.warp(TRg_ext, F_BG))[c]
    hole = hole_ext[c]
    okB = cBG[c] & ~hole
    Bfill = np.where(okB[..., None], Bw, TRg)
    BR = BR_ext[c]
    C = np.where(hole[..., None], Bfill, BR)
    K1 = S.shift_bicubic(TRg, 0.5, 0.5)
    X = S.warp(TRg_ext, F_BG)                     # TRg content on the BR grid (float)
    Yrt = S.q8(S.warp(S.q8(X), F_GB))[c]          # and back
    K2 = np.where(cGB[c][..., None], Yrt, TRg)
    # disparity statistics (horizontal) on consistent, textured pixels
    gy = S.luma(TRg)
    tex = (np.abs(np.diff(gy, axis=1, append=gy[:, -1:])) + np.abs(np.diff(gy, axis=0, append=gy[-1:]))) > 0.02
    rres = F_BG[c][..., 0][okB & tex]
    per.append(dict(fi=fi, shift=(ddy, ddx), leftHalf_vs_splatTL_mad=lh_dev,
                    true_occl=float(1 - consA.mean()), hole=float(hole.mean()),
                    geom_cons=float(cBG[c][~hole].mean()),
                    resid_abs_med=float(np.median(np.abs(rres))) if rres.size else None,
                    resid_abs_p90=float(np.percentile(np.abs(rres), 90)) if rres.size else None,
                    resid_gt2=float((np.abs(rres) > 2).mean()) if rres.size else None,
                    resid_gt4=float((np.abs(rres) > 4).mean()) if rres.size else None,
                    resid_gt8=float((np.abs(rres) > 8).mean()) if rres.size else None,
                    real_disp_med=float(np.median(F_GL[c][..., 0][consA & tex])) if (consA & tex).any() else None,
                    real_disp_p5_p95=[float(np.percentile(F_GL[c][..., 0][consA & tex], q)) for q in (5, 95)]
                    if (consA & tex).any() else None))
    for k, v in (("TRg", TRg), ("A", Afill), ("consA", consA), ("B", Bfill), ("C", C), ("hole", hole), ("K1", K1),
                 ("K2", K2), ("BR", BR), ("Yrt", Yrt)):
        store[k].append(v)
    log(f"f{fi}: shift ({ddy},{ddx}) trueOccl {per[-1]['true_occl']:.3f} hole {per[-1]['hole']:.3f} "
        f"resid|dx| med {per[-1]['resid_abs_med']} p90 {per[-1]['resid_abs_p90']} leftdev {lh_dev:.2f}")
for k in store:
    store[k] = np.stack(store[k])
TRg = store["TRg"]; consA = store["consA"]; hole = store["hole"]

# ---- per-clip oracle colour fits (pooled over frames, fixed subsample)
rng = np.random.default_rng(0)


def pool(src, dst, m, n=20000):
    xs, ys = [], []
    for j in range(len(src)):
        idx = np.flatnonzero(m[j].ravel())
        if idx.size == 0:
            continue
        idx = rng.choice(idx, min(n, idx.size), replace=False)
        xs.append(src[j].reshape(-1, 3)[idx].astype(np.float64) / 255.)
        ys.append(dst[j].reshape(-1, 3)[idx].astype(np.float64) / 255.)
    return np.concatenate(xs), np.concatenate(ys)


MA = S.fit_affine(*pool(store["A"], TRg, consA))
Acc = np.where(consA[..., None], np.stack([S.apply_affine(a, MA) for a in store["A"]]), TRg)
MC = S.fit_affine(*pool(store["C"], TRg, ~hole))
Ccc_naive = np.where(hole[..., None], store["C"], np.stack([S.apply_affine(a, MC) for a in store["C"]]))
# ADDENDUM 1: the camera colour map is fitted on flow-registered left->GT pairs (MA) and applied to C
Ccc = np.where(hole[..., None], store["C"], np.stack([S.apply_affine(a, MA) for a in store["C"]]))
log("colour fits done")

# ---- LPIPS (scalar, score_clip_ll batching) and spatial maps
lp = S.Lp("alex")
cand = {"O_A": store["A"], "O_Acc": Acc, "O_B": store["B"], "O_C": store["C"], "O_Ccc": Ccc,
        "O_Ccc_naive": Ccc_naive,
        "K1": store["K1"], "K2": store["K2"], "BR_raw": store["BR"], "copyleft_win": Lhalf}
cand.update({r: rows[r] for r in rows})
res = dict(clip=CLIP, family=S.family(CLIP), frames=frames, regjson=jpath, per_frame=per, rows={})
for k, img in cand.items():
    v = lp(img, TRg)
    sm = lp.spatial(img, TRg)
    nonhole_cons = consA & ~hole
    e = dict(lpips=float(np.mean(v)), frames=v,
             psnr=float(np.mean([S.psnr_u8(img[j], TRg[j]) for j in range(len(frames))])),
             sp_mean=float(sm.mean()),
             sp_hole=float(sm[hole].mean()) if hole.any() else None,
             sp_nonhole=float(sm[~hole].mean()),
             sp_trueoccl=float(sm[~consA].mean()) if (~consA).any() else None,
             sp_cons_nonhole=float(sm[nonhole_cons].mean()),
             mass_hole=float(sm[hole].sum() / sm.sum()), mass_trueoccl=float(sm[~consA].sum() / sm.sum()),
             sharp=S.sharp_score(img))
    res["rows"][k] = e
    log(f"{k:12s} LPIPS {e['lpips']:.4f}  PSNR {e['psnr']:.2f}  sp nonhole {e['sp_nonhole']:.4f} hole "
        f"{e['sp_hole'] if e['sp_hole'] is None else round(e['sp_hole'], 4)}  cons {e['sp_cons_nonhole']:.4f}")
res["gtSharp"] = S.sharp_score(TRg)
# ---- M4 U3: consistent pixels whose colour-corrected left-eye luma differs from GT by > 24/255
YAcc, YT = S.luma(Acc), S.luma(TRg)
u3 = [float((np.abs(YAcc[j] - YT[j])[consA[j]] > 24 / 255).mean()) for j in range(len(frames))]
res["U3_frames"] = u3
res["U3"] = float(np.mean(u3))
# ---- exploratory: input fidelity, LPIPS(row, model input BR) over non-hole pixels (spatial) and whole frame
fid = {}
for k in ["origin", "AYS8", "deliverable", "s25"]:
    smf = lp.spatial(rows[k], store["BR"])
    fid[k] = dict(lpips_vs_BR=float(np.mean(lp(rows[k], store["BR"]))), sp_nonhole=float(smf[~hole].mean()))
res["fidelity_to_input"] = fid
log("U3", res["U3"], "fidelity", {k: round(v["sp_nonhole"], 4) for k, v in fid.items()})
res["colour_fit"] = dict(MA=MA.tolist(), MC=MC.tolist())

# ---- M1 camera differences (spectra shift-invariant: windows unregistered)
def specs(arr):
    return np.array([S.band_power(S.luma(a)) for a in arr])


TLwin = np.stack([S.box_crop(D["TLtrain"][j], 0, 0) for j in range(len(frames))])
TRwin = np.stack([S.box_crop(D["TR"][j], 0, 0) for j in range(len(frames))])
P = dict(T_L=specs(TLwin), T_R=specs(TRwin), V_L=specs(v2["left"]), V_R=specs(v2["right"]),
         IN_L=specs(Lhalf), origin=specs(rows["origin"]), deliverable=specs(rows["deliverable"]),
         s25=specs(rows["s25"]), TRg=specs(TRg))
m1 = dict(bandpower={k: v.mean(0).tolist() for k, v in P.items()})
m1["rho_T"] = np.sqrt(P["T_R"].mean(0) / P["T_L"].mean(0)).tolist()
m1["rho_V"] = np.sqrt(P["V_R"].mean(0) / P["V_L"].mean(0)).tolist()
m1["rho_origin_vs_input"] = np.sqrt(P["origin"].mean(0) / P["IN_L"].mean(0)).tolist()
m1["rho_deliv_vs_input"] = np.sqrt(P["deliverable"].mean(0) / P["IN_L"].mean(0)).tolist()
m1["rho_s25_vs_input"] = np.sqrt(P["s25"].mean(0) / P["IN_L"].mean(0)).tolist()
m1["rho_GT_vs_input"] = np.sqrt(P["TRg"].mean(0) / P["IN_L"].mean(0)).tolist()
m1["sharp"] = dict(T_L=S.sharp_score(TLwin), T_R=S.sharp_score(TRwin), V_L=S.sharp_score(v2["left"]),
                   V_R=S.sharp_score(v2["right"]), IN_L=S.sharp_score(Lhalf))
m1["noise"] = {k: float(np.mean([S.immerkaer_sigma(S.luma(a)) for a in arr]))
               for k, arr in (("T_L", TLwin), ("T_R", TRwin), ("V_L", v2["left"]), ("V_R", v2["right"]))}
# exposure / colour on the registered overlap (left eye warped into the GT frame, consistent pixels)
YA = S.luma(store["A"]); YG = S.luma(TRg)
m1["exposure"] = dict(dY_8bit=float((YG[consA] - YA[consA]).mean() * 255),
                      contrast_ratio=float(YG[consA].std() / YA[consA].std()))
labA = S.lab_mean(store["A"], consA); labG = S.lab_mean(TRg, consA)
m1["lab_offset_GT_minus_left"] = (labG - labA).tolist()
m1["psnr_A_cons"] = S.psnr_u8(store["A"], TRg, consA)
m1["psnr_Acc_cons"] = S.psnr_u8(Acc, TRg, consA)


# relative blur (secondary): sigma grid on luma, consistent pixels, after a gain/offset fit
def sig_fit(Lw, R, m, grid=np.arange(0, 2.51, 0.25)):
    eR, eL = [], []
    for s in grid:
        a, b = [], []
        for j in range(len(Lw)):
            yl, yr = S.luma(Lw[j]), S.luma(R[j])
            mm = m[j].copy(); mm[:8] = mm[-8:] = False; mm[:, :8] = mm[:, -8:] = False
            gl, gr = S.gauss(yl, s), S.gauss(yr, s)
            for x, y, acc in ((gl, yr, a), (yl, gr, b)):
                X = np.stack([x[mm], np.ones(mm.sum())], 1)
                coef, *_ = np.linalg.lstsq(X, y[mm], rcond=None)
                acc.append(float(((X @ coef - y[mm]) ** 2).mean()))
        eR.append(np.mean(a)); eL.append(np.mean(b))
    return float(grid[int(np.argmin(eR))]), float(grid[int(np.argmin(eL))])


sR, sL = sig_fit(store["A"], TRg, consA)
cR, cL = sig_fit(store["Yrt"], TRg, consA & ~hole)
m1["sigma"] = dict(sigma_R=sR, sigma_L=sL, sigma_rel=sR - sL, ctrl_sigma_R=cR, ctrl_sigma_L=cL,
                   ctrl_sigma_rel=cR - cL, sigma_rel_corrected=(sR - sL) - (cR - cL) / np.sqrt(2))
res["m1"] = m1
log(f"M1 rho_T {np.round(m1['rho_T'], 3).tolist()} rho_V {np.round(m1['rho_V'], 3).tolist()} "
    f"origin/input {np.round(m1['rho_origin_vs_input'], 3).tolist()} sigma {m1['sigma']}")
res["seconds"] = time.time() - T0
json.dump(res, open(oj, "w"))
np.savez_compressed(os.path.join(OUT, f"{CLIP}_masks.npz"), consA=consA, hole=hole)
# one inspection panel (half res): TRg | O_A | O_B | O_C | origin
j = len(frames) // 2
half = lambda a: cv2.resize(a, (TW // 2, TH // 2), interpolation=cv2.INTER_AREA)
pan = np.concatenate([half(x) for x in (TRg[j], store["A"][j], store["B"][j], store["C"][j], rows["origin"][j])], 1)
for q, s in enumerate(["GT REG_FRAME", "O_A left@true geom", "O_B GT@model geom", "O_C input BR+oracle holes", "origin"]):
    cv2.putText(pan, s, (q * TW // 2 + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
cv2.imwrite(os.path.join(OUT, f"{CLIP}_oracles_f{frames[j]:03d}.png"), cv2.cvtColor(pan, cv2.COLOR_RGB2BGR))
log("wrote", oj)
print("CLIP_DONE", CLIP, flush=True)
