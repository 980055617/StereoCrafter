#!/usr/bin/env python
"""M1 broad train sample + the per-family post-processing maps P11 (colour) / P12 (filter choice) of PREREG M6.
Valid TRAIN clips only (role train in fulldata_v1.json, id < 0310, not test/dev, G0 per clip), 24 per family
(every k-th of the sorted valid list), frames 10/40/70.  CPU only.
usage: python job_trainsample_v1.py <out_json>"""
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
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
T0 = time.time()
TH, TW = S.TH, S.TW
FR = [10, 40, 70]
NPF = int(os.environ.get("SK_NPF", "24"))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


split = json.load(open("scripts/distill/splits/fulldata_v1.json"))
excl = set(S.TEST) | set(S.DEV)
cands = sorted(c for c, v in split["clips"].items() if v["role"] == "train" and int(c) < 310 and c not in excl
               and os.path.exists(f"video_data/train/{c}_train.mp4"))
fams = {"AVP": [c for c in cands if int(c) <= 159], "iPhone": [c for c in cands if 160 <= int(c) <= 309]}
log({k: len(v) for k, v in fams.items()})

# GT-independent filter grid of PREREG M6 (P0-P10), applied to uint8 RGB
GRID = [("P0", "id", 0, 0)] + [(f"P{1 + i}", "unsharp", s, a) for i, (s, a) in
                               enumerate([(1, .3), (1, .6), (1, 1.), (2, .3), (2, .6), (2, 1.)])] + \
       [("P7", "blur", .5, 0), ("P8", "blur", 1., 0), ("P9", "grain", 2 / 255, 0), ("P10", "grain", 4 / 255, 0)]


def apply_P(img_u8, kind, s, a, seed=0):
    x = img_u8.astype(np.float32) / 255.
    if kind == "id":
        y = x
    elif kind == "unsharp":
        y = x + a * (x - S.gauss(x, s))
    elif kind == "blur":
        y = S.gauss(x, s)
    elif kind == "grain":
        y = x + np.random.default_rng(seed).normal(0, s, x.shape).astype(np.float32)
    return S.q8(y)


dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def dis_flow(a_u8, b_u8):
    ga = cv2.cvtColor(a_u8, cv2.COLOR_RGB2GRAY); gb = cv2.cvtColor(b_u8, cv2.COLOR_RGB2GRAY)
    return dis.calc(ga, gb, None)


lp = S.Lp("alex")
res = dict(frames=FR, per_family={}, clips={})
for fam, lst in fams.items():
    k = max(1, len(lst) // NPF)
    pick = lst[::k][:NPF]
    log(fam, "picked", pick)
    pairs_src, pairs_dst = [], []
    lp_rows = {g[0]: [] for g in GRID}
    lp_rows_f = {g[0]: [] for g in GRID}
    for c in pick:
        real = S.check_valid_train_bundle(c)
        vr = VideoReader(f"video_data/train/{c}_train.mp4", ctx=cpu(0))
        n = len(vr)
        fr = [f for f in FR if f < n]
        f0 = vr[0].asnumpy(); H, W = f0.shape[0] // 2, f0.shape[1] // 2
        t0, l0 = (H // 128 * 128 - TH) // 2, (W // 128 * 128 - TW) // 2
        b = vr.get_batch(fr).asnumpy()
        TLw = b[:, t0:t0 + TH, l0:l0 + TW].copy()
        mad = float(np.mean([np.abs((TLw[j].astype(np.float32) - b[j, t0:t0 + TH, W + l0:W + l0 + TW]) -
                                    (TLw[j].astype(np.float32) - b[j, t0:t0 + TH, W + l0:W + l0 + TW]).mean((0, 1))).mean()
                             for j in range(len(fr))]))
        if mad <= 5.0:
            log(c, "G0 FAIL mad", mad); res["clips"][c] = dict(valid=False, mad=mad); continue
        # TR box around the window for the global registration (TR crop moved to match TL)
        TRbox = b[:, t0 - S.BOX_T:t0 + TH + S.BOX_B, W + l0 - S.BOX_L:W + l0 + TW + S.BOX_R].copy()
        TRw = TRbox[:, S.BOX_T:S.BOX_T + TH, S.BOX_L:S.BOX_L + TW]
        del b
        v2 = {}
        for eye in ("left", "right"):
            vv = VideoReader(f"video_data/{eye}_eye_v2/{c}.mp4", ctx=cpu(0))
            bb = vv.get_batch(fr).asnumpy()
            v2[eye] = bb[:, t0:t0 + TH, l0:l0 + TW].copy(); del bb
        e = dict(valid=True, mad=mad, family=fam, frames=fr, real=real, shifts=[], psnr_global=[])
        PT_L = np.array([S.band_power(S.luma(a)) for a in TLw]); PT_R = np.array([S.band_power(S.luma(a)) for a in TRw])
        PV_L = np.array([S.band_power(S.luma(a)) for a in v2["left"]]); PV_R = np.array([S.band_power(S.luma(a)) for a in v2["right"]])
        e["rho_T"] = np.sqrt(PT_R.mean(0) / PT_L.mean(0)).tolist()
        e["rho_V"] = np.sqrt(PV_R.mean(0) / PV_L.mean(0)).tolist()
        e["sharp"] = dict(T_L=S.sharp_score(TLw), T_R=S.sharp_score(TRw), V_L=S.sharp_score(v2["left"]),
                          V_R=S.sharp_score(v2["right"]))
        e["noise"] = dict(T_L=float(np.mean([S.immerkaer_sigma(S.luma(a)) for a in TLw])),
                          T_R=float(np.mean([S.immerkaer_sigma(S.luma(a)) for a in TRw])),
                          V_L=float(np.mean([S.immerkaer_sigma(S.luma(a)) for a in v2["left"]])),
                          V_R=float(np.mean([S.immerkaer_sigma(S.luma(a)) for a in v2["right"]])))
        dY, cr, lab_off = [], [], []
        GTg = []
        for j in range(len(fr)):
            dy, dx, ps = S.best_shift_psnr(TRbox[j], TLw[j], (-4, 4), (-80, 16))
            e["shifts"].append((dy, dx)); e["psnr_global"].append(ps)
            G = np.ascontiguousarray(S.box_crop(TRbox[j], dy, dx)); GTg.append(G)
            # flow-registered pairs: left warped into the (globally shifted) right frame, consistent pixels
            f_rl = dis_flow(G, TLw[j]); f_lr = dis_flow(TLw[j], G)
            cons = S.consistency(f_rl, f_lr)
            cons[:16] = cons[-16:] = False; cons[:, :16] = cons[:, -16:] = False
            Aw = S.q8(S.warp(TLw[j], f_rl))
            yA, yG = S.luma(Aw), S.luma(G)
            dY.append(float((yG[cons] - yA[cons]).mean() * 255)); cr.append(float(yG[cons].std() / yA[cons].std()))
            lab_off.append((S.lab_mean(G, cons) - S.lab_mean(Aw, cons)).tolist())
            cidx = np.flatnonzero(cons.ravel())
            idx = np.random.default_rng(int(c) * 100 + j).choice(cidx, min(5000, cidx.size), replace=False)
            pairs_src.append(Aw.reshape(-1, 3)[idx].astype(np.float64) / 255.)
            pairs_dst.append(G.reshape(-1, 3)[idx].astype(np.float64) / 255.)
            # P12 selection data: filters on the left eye vs GT registered to it (global = what GT training sees;
            # flow = camera look only, inconsistent pixels taken from GT)
            Af = np.where(cons[..., None], Aw, G)
            for (nm, kind, s, a) in GRID:
                lp_rows[nm].append(lp(apply_P(TLw[j], kind, s, a, seed=j)[None], G[None])[0])
                lp_rows_f[nm].append(lp(apply_P(Af, kind, s, a, seed=j)[None], G[None])[0])
        e["copyleft_lpips_global"] = [float(x) for x in lp(TLw, np.stack(GTg))]
        e["exposure_dY_8bit"] = float(np.mean(dY)); e["contrast_ratio"] = float(np.mean(cr))
        e["lab_offset_R_minus_L"] = np.mean(lab_off, 0).tolist()
        res["clips"][c] = e
        log(c, fam, f"rho_T b3 {e['rho_T'][2]:.3f} rho_V b3 {e['rho_V'][2]:.3f} dY {e['exposure_dY_8bit']:+.2f} "
            f"lab {np.round(e['lab_offset_R_minus_L'], 2).tolist()} copyleft {np.mean(e['copyleft_lpips_global']):.4f}")
    M = S.fit_affine(np.concatenate(pairs_src), np.concatenate(pairs_dst))
    sel_g = {nm: float(np.mean(v)) for nm, v in lp_rows.items()}
    sel_f = {nm: float(np.mean(v)) for nm, v in lp_rows_f.items()}
    good = [c for c in pick if res["clips"].get(c, {}).get("valid")]
    res["per_family"][fam] = dict(
        picked=pick, valid=good, P11_affine=M.tolist(),
        P12_lpips_by_filter_global=sel_g, P12_choice_global=min(sel_g, key=sel_g.get),
        P12_lpips_by_filter_flow=sel_f, P12_choice_flow=min(sel_f, key=sel_f.get),
        median_rho_T=np.median([res["clips"][c]["rho_T"] for c in good], 0).tolist(),
        median_rho_V=np.median([res["clips"][c]["rho_V"] for c in good], 0).tolist(),
        median_dY=float(np.median([res["clips"][c]["exposure_dY_8bit"] for c in good])),
        median_lab=np.median([res["clips"][c]["lab_offset_R_minus_L"] for c in good], 0).tolist())
    log(fam, "P12 global", res["per_family"][fam]["P12_choice_global"], "flow", res["per_family"][fam]["P12_choice_flow"],
        "median rho_T", np.round(res["per_family"][fam]["median_rho_T"], 3).tolist(),
        "rho_V", np.round(res["per_family"][fam]["median_rho_V"], 3).tolist())
res["grid"] = GRID
res["seconds"] = time.time() - T0
json.dump(res, open(OUT, "w"))
log("wrote", OUT)
