#!/usr/bin/env python
"""Summarise phase A (prep_crops_v3.py) and build the MAIN / CONTRAST training specs with the PRE-REGISTERED exclusions
(PREREG.txt): clip excluded if its registration hit the search boundary; window excluded if kept < 0.25.
CONTRAST = the first 8 clips of train_order (train_clips_v1.json order) that keep >= 1 window.
Also writes the residual-disparity statistics (local best shift vs the clip-global shift) over all windows.
usage: python build_specs_v1.py <crops_root> <latent_dir> <out_dir_records>
"""
import json, os, sys
import numpy as np
CROPS, LAT, REC = sys.argv[1], sys.argv[2], sys.argv[3]
KEPT_MIN = 0.25
order = json.load(open(os.path.join(REC, "train_clips_v1.json")))["clips"]
rows, keep_w, excl_clip, excl_win = [], [], [], []
hist_dx = np.zeros(33, np.int64); n_info = n_cons = n_flat = 0
for c in order:
    j = json.load(open(os.path.join(CROPS, c, "clip.json")))
    kw = []
    for ws in j["windows_stats"]:
        ok = (not j["boundary"]) and ws["kept"] >= KEPT_MIN
        (kw if ok else excl_win).append((c, ws["start"]) if ok else (c, ws["start"], round(ws["kept"], 3), j["boundary"]))
        ci = np.load(os.path.join(CROPS, c, f"w{ws['start']:03d}", "cellinfo.npy"))
        ls = np.load(os.path.join(CROPS, c, f"w{ws['start']:03d}", "lshift.npy"))
        info = (ci & 1) > 0; flat = (ci & 2) > 0
        sel = info & ~flat
        dx = ls[:, 1][sel].astype(np.int64)
        hist_dx += np.bincount(np.clip(dx // 2 + 16, 0, 32), minlength=33)
        n_info += int(info.sum()); n_flat += int((info & flat).sum()); n_cons += int(((ci & 4) > 0)[sel].sum())
    if j["boundary"]: excl_clip.append(c)
    keep_w += kw
    rows.append(dict(clip=c, ddy=j["ddy"], ddx=j["ddx"], boundary=j["boundary"], mean_psnr=j["mean_psnr"], mean_psnr_zero=j["mean_psnr_zero"],
                     kept=[round(ws["kept"], 4) for ws in j["windows_stats"]], n_keep=len(kw),
                     hole=[round(ws["hole_frac"], 4) for ws in j["windows_stats"]],
                     rgb_off=[[round(x, 2) for x in ws["rgb_offset_255"]] for ws in j["windows_stats"]]))
kept_all = np.array([k for r in rows for k in r["kept"]])
contrast_clips = [r["clip"] for r in rows if r["n_keep"] > 0][:8]
contrast_w = [w for w in keep_w if w[0] in contrast_clips]
summ = dict(n_clips=len(rows), n_windows=int(kept_all.size), kept_quantiles={q: float(np.quantile(kept_all, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
            kept_mean=float(kept_all.mean()), excluded_boundary_clips=excl_clip, n_windows_kept=len(keep_w),
            n_windows_excluded=len(excl_win), n_clips_with_windows=len(set(w[0] for w in keep_w)),
            clips_without_windows=[r["clip"] for r in rows if r["n_keep"] == 0],
            kept_mean_of_used=float(np.mean([k for r in rows for k, ws in zip(r["kept"], r["kept"]) if k >= KEPT_MIN and not r["boundary"]])),
            contrast_clips=contrast_clips, n_contrast_windows=len(contrast_w),
            residual_dx_hist_px={int((i - 16) * 2): int(v) for i, v in enumerate(hist_dx) if v},
            frac_informative_nonflat_within_2px=float(hist_dx[15:18].sum() / max(hist_dx.sum(), 1)),
            frac_within_4px=float(hist_dx[14:19].sum() / max(hist_dx.sum(), 1)),
            frac_within_8px=float(hist_dx[12:21].sum() / max(hist_dx.sum(), 1)),
            ddx_by_fmt=None)
json.dump(dict(summary=summ, clips=rows, excluded_windows=excl_win), open(os.path.join(REC, "phaseA_summary_v1.json"), "w"), indent=1)
def spec(name, wins, steps, ck_every):
    return dict(name=name, latent_dir=LAT, windows=[[c, int(s)] for c, s in wins], steps=steps, ck_every=ck_every, seed=1234, lr=1e-5,
                ck_dir="/mnt/ssd_data/deep_20261004/scale_gt/ck", rec_dir=os.path.join(REC, "train"))
json.dump(spec("main_v1", keep_w, 3000, 250), open(os.path.join(REC, "spec_main_v1.json"), "w"), indent=1)
json.dump(spec("contrast8_v1", contrast_w, 1000, 250), open(os.path.join(REC, "spec_contrast8_v1.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
