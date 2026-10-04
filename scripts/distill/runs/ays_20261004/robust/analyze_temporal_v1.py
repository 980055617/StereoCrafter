#!/usr/bin/env python
"""ays_20261004 / robust -- flicker analysis (CPU only).  Definitions: PREREG.txt section (2), readings F-literal,
F-slope (primary for the reframing), F-direct.

inputs : outputs/ays_20261004/robust/temporal_v1/<clip>.json (+ .log)  -- score_temporal_ll.py, every frame
         outputs/finalcheck_20261004/validate/v1_temporal/temporal_576_12clip.json  (V1: gate T1 + s25 context)
         outputs/finalcheck_20261004/validate/v1_temporal/diag_D1D2.json          (D2 unsharp-mask origin R_k)
         scripts/distill/runs/ays_20261004/robust/ROWS_pass1.json                (score_clip_ll ROW sharpness, STEP=4)
         optional: the registered stats JSON of analyze_reg_v1.py (C1 LPIPS deltas for F-direct)
usage: python analyze_temporal_v1.py <out_table.txt> <out_stats.json> <out_png> [<reg_stats.json>]
"""
import json
import math
import os
import sys

import numpy as np

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT_T, OUT_J, OUT_PNG = sys.argv[1:4]
REG = sys.argv[4] if len(sys.argv) > 4 else ""
for p in (OUT_T, OUT_J, OUT_PNG):
    assert not os.path.exists(p), f"refusing to overwrite {p}"
R = "scripts/distill/runs/ays_20261004/robust"
O = "outputs/ays_20261004/robust"
VD = "outputs/finalcheck_20261004/validate/v1_temporal"
ROWS = json.load(open(f"{R}/ROWS_pass1.json"))
CLIPS = ROWS["clips"]
V1 = json.load(open(f"{VD}/temporal_576_12clip.json"))
DIAG = json.load(open(f"{VD}/diag_D1D2.json"))
T = {c: json.load(open(f"{O}/temporal_v1/{c}.json"))[c] for c in CLIPS}   # each file is {clip: {GT, <clip>_<label>...}}
ORIGIN, AYS8, DELIV, T5N, T5P, OT5P, S25 = ("origin_ll", "AYS8_origin_g101", "mstudent2_step800_deliv_ll",
                                            "deliv_g100_T5nat", "deliv_g100_T5pad", "origin_g100_T5pad", "s25_ll")
MINE = [ORIGIN, AYS8, DELIV, T5N, T5P, OT5P]
NAME = {ORIGIN: "Karras origin 8x2 @1.01", AYS8: "AYS8 origin 8x2 @1.01", DELIV: "deliverable 8x2 @1.01",
        T5N: "deliverable T5@1.00 unpad (ships)", T5P: "deliverable T5@1.00 pad8", OT5P: "origin T5@1.00 pad8",
        S25: "origin s25 (V1 file)", "Rk": "unsharp-mask origin R_k (V1 D2)", "GT": "GT right eye"}
B, SEED = 100_000, 20261004
B_GEN_PREREG, B_USM_PREREG = 1.5475, 0.7556
L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


def val(c, lab, m):
    if lab == "GT":
        return T[c]["GT"][m]
    if lab == S25:
        return V1[c][f"{c}_{lab}"][m]
    if lab == "Rk":
        return DIAG[c]["D2"][m]
    return T[c][f"{c}_{lab}"][m]


def ratio_metric(c, lab, m):
    if m == "ratio":
        return val(c, lab, "seam") / val(c, lab, "nonseam")
    return val(c, lab, m)


def sharp(c, lab):
    return ROWS["cells"][c][lab]["sharp"] if lab in ROWS["cells"][c] else None


P("=" * 150)
P("ays_20261004 / robust -- FLICKER of the AYS8 schedule vs the deliverable (12 test clips, 576x1024, every frame)")
P(f"scorer: scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py (unchanged); scores: {O}/temporal_v1/<clip>.json")
P("prereg: scripts/distill/runs/ays_20261004/robust/PREREG.txt section (2)")
P("=" * 150)

# =========================================================================================== gates
gates = {}
t0 = []
for c in CLIPS:
    log = open(f"{O}/temporal_v1/{c}.log").read()
    raft = "[temporal] RAFT loaded" in log
    finite = all(math.isfinite(T[c][f"{c}_{lab}"]["warp"]) for lab in MINE) and math.isfinite(T[c]["GT"]["warp"])
    have = all(f"{c}_{lab}" in T[c] for lab in MINE)
    t0.append(dict(clip=c, raft=raft, finite=finite, all_rows=have))
gates["T0"] = dict(pass_=all(x["raft"] and x["finite"] and x["all_rows"] for x in t0), per_clip=t0)
P(f"\nT0 RAFT loaded on every clip, every warp finite, all {len(MINE)} rows present: {'PASS' if gates['T0']['pass_'] else 'FAIL'}")
for x in t0:
    if not (x["raft"] and x["finite"] and x["all_rows"]):
        P(f"    FAIL {x}")
t1_bad, t1_max = [], 0.0
for c in CLIPS:
    for lab in ("GT", ORIGIN, DELIV):
        mine = T[c]["GT"] if lab == "GT" else T[c][f"{c}_{lab}"]
        ref = V1[c]["GT"] if lab == "GT" else V1[c][f"{c}_{lab}"]
        for m in ("warp", "tLP", "seam", "nonseam"):
            dv = abs(mine[m] - ref[m])
            t1_max = max(t1_max, dv)
            if dv > 2e-5:
                t1_bad.append((c, lab, m, mine[m], ref[m]))
        for k in ("t0", "l0", "n"):
            if mine[k] != ref[k]:
                t1_bad.append((c, lab, k, mine[k], ref[k]))
gates["T1"] = dict(pass_=not t1_bad, failing=t1_bad, max_abs_dev=t1_max)
P(f"T1 GT / origin / deliverable warp, tLP, seam, nonseam equal V1 within 2e-5 and t0/l0/n equal: "
  f"{'PASS' if not t1_bad else 'FAIL'}  (max |dev| {t1_max:.1e})  -> V1's s25 and D2 values share these flows")
for b in t1_bad:
    P(f"    FAIL {b}")

# =========================================================================================== tables
SHOW = ["GT", ORIGIN, AYS8, DELIV, T5N, T5P, OT5P, S25, "Rk"]
TAB = {}
for m, title in (("warp", "warp error (RAFT flow on the GT window, fwd-bwd mask; lower = steadier)"),
                 ("tLP", "tLP = mean LPIPS(frame t, t+1)"), ("ratio", "seam ratio (seam / nonseam)")):
    P(f"\n--- {title} ---")
    P(f"  {'row':36s}" + "".join(f"{c:>8s}" for c in CLIPS) + f"{'MEAN':>9s}")
    for lab in SHOW:
        if lab == "Rk" and m == "ratio":
            vals = [DIAG[c]["D2"]["seam"] / DIAG[c]["D2"]["nonseam"] for c in CLIPS]
        else:
            vals = [ratio_metric(c, lab, m) for c in CLIPS]
        TAB[f"{m}|{lab}"] = vals
        P(f"  {NAME[lab]:36s}" + "".join(f"{x:8.4f}" for x in vals) + f"{np.mean(vals):9.5f}")
    P(f"  per clip ratio vs Karras origin:")
    for lab in SHOW[2:]:
        r = [a / b for a, b in zip(TAB[f"{m}|{lab}"], TAB[f"{m}|{ORIGIN}"])]
        mm = np.mean(TAB[f"{m}|{lab}"]) / np.mean(TAB[f"{m}|{ORIGIN}"]) - 1
        worse = sum(a > b for a, b in zip(TAB[f"{m}|{lab}"], TAB[f"{m}|{ORIGIN}"]))
        P(f"  {'  ' + NAME[lab]:36s}" + "".join(f"{x:8.3f}" for x in r) + f"{100 * mm:+8.1f}%  higher on {worse}/12")

# =========================================================================================== F-literal
inc = {lab: float(np.mean(TAB[f"warp|{lab}"]) / np.mean(TAB[f"warp|{ORIGIN}"]) - 1) for lab in SHOW[2:]}
worse = {lab: int(sum(a > b for a, b in zip(TAB[f"warp|{lab}"], TAB[f"warp|{ORIGIN}"]))) for lab in SHOW[2:]}
same = inc[AYS8] >= 0.168
P("\n--- F-literal (pre-registered): does AYS8 show the same warp increase as the deliverable? ---")
P(f"  warp increase vs Karras origin (ratio of 12-clip means - 1):  deliverable {100 * inc[DELIV]:+.1f}% ({worse[DELIV]}/12)   "
  f"AYS8 {100 * inc[AYS8]:+.1f}% ({worse[AYS8]}/12)   s25 {100 * inc[S25]:+.1f}% ({worse[S25]}/12)   "
  f"T5nat (ships) {100 * inc[T5N]:+.1f}% ({worse[T5N]}/12)")
P(f"  rule: SAME iff AYS8 >= +16.8 % (0.75 x +22.4 %)  ->  {'YES' if same else 'NO'}")

# =========================================================================================== F-slope
CL = np.array(CLIPS)


def xy(lab, m="warp"):
    if lab == "Rk":
        x = np.log(np.array([DIAG[c]["D2"]["sharp_achieved"] / DIAG[c]["D2"]["sharp_origin"] for c in CLIPS]))
        y = np.log(np.array([DIAG[c]["D2"][m] / V1[c][f"{c}_{ORIGIN}"][m] for c in CLIPS]))
        return x, y
    x = np.log(np.array([sharp(c, lab) / sharp(c, ORIGIN) for c in CLIPS]))
    y = np.log(np.array([val(c, lab, m) / val(c, ORIGIN, m) for c in CLIPS]))
    return x, y


def slope0(x, y):
    return float((x * y).sum() / (x * x).sum())


def boot_slope(x, y, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    X, Y = x[idx], y[idx]
    bs = (X * Y).sum(1) / (X * X).sum(1)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


SL = {}
P("\n--- F-slope (PRIMARY for the reframing): flicker cost per unit sharpness ---")
P("  x = ln(sharp_row / sharp_origin) (score_clip_ll ROW sharpness, STEP=4; R_k: its own D2 STEP=1 fields), "
  "y = ln(metric_row / metric_origin); b = sum(xy)/sum(x^2) (through the origin); 95 % CI = percentile bootstrap over "
  "clips, B=100k, seed 20261004")
for m in ("warp", "tLP"):
    xg = np.concatenate([xy(DELIV, m)[0], xy(S25, m)[0]])
    yg = np.concatenate([xy(DELIV, m)[1], xy(S25, m)[1]])
    b_gen = slope0(xg, yg)
    b_usm = slope0(*xy("Rk", m))
    SL[m] = dict(b_gen=b_gen, b_usm=b_usm, rows={})
    P(f"\n  [{m}]  references recomputed: b_gen (deliverable+s25 pooled, n=24) {b_gen:.4f}   b_usm (unsharp mask R_k) {b_usm:.4f}"
      + (f"   (PREREG {B_GEN_PREREG} / {B_USM_PREREG}: {'match' if (round(b_gen, 4), round(b_usm, 4)) == (B_GEN_PREREG, B_USM_PREREG) else 'MISMATCH'})"
         if m == "warp" else "   (PREREG 1.1501 / 0.5050)"))
    P(f"  {'row':36s} {'geo sharp':>10s} {'geo ' + m:>10s} {'b':>8s} {'95% CI':>20s} {'b w/o 0301':>11s} "
      f"{'OLS slope':>10s} {'intercept':>10s} {'corr':>6s}")
    for lab in (AYS8, DELIV, S25, T5N, "Rk"):
        x, y = xy(lab, m)
        b = slope0(x, y)
        ci = boot_slope(x, y)
        k = CL != "0301"
        b_no = slope0(x[k], y[k])
        A = np.vstack([x, np.ones_like(x)]).T
        ols = np.linalg.lstsq(A, y, rcond=None)[0]
        corr = float(np.corrcoef(x, y)[0, 1])
        SL[m]["rows"][lab] = dict(b=b, ci=ci, b_wo_0301=b_no, ols_slope=float(ols[0]), ols_icpt=float(ols[1]), corr=corr,
                                  geo_sharp=float(np.exp(x.mean())), geo_metric=float(np.exp(y.mean())),
                                  x=x.tolist(), y=y.tolist())
        P(f"  {NAME[lab]:36s} {np.exp(x.mean()):10.4f} {np.exp(y.mean()):10.4f} {b:8.4f} {f'[{ci[0]:.3f},{ci[1]:.3f}]':>20s} "
          f"{b_no:11.4f} {ols[0]:10.4f} {ols[1]:+10.4f} {corr:6.3f}")
    ci = SL[m]["rows"][AYS8]["ci"]
    in_gen, in_usm = ci[0] <= b_gen <= ci[1], ci[0] <= b_usm <= ci[1]
    if in_gen and in_usm:
        rd = "INCONCLUSIVE (the CI contains both references)"
    elif in_gen:
        rd = "TRACKS SHARPNESS LIKE THE DELIVERABLE (same flicker cost per unit sharpness)"
    elif ci[1] < b_gen:
        rd = "CHEAPER than the generative line"
    else:
        rd = "COSTLIER than the generative line"
    SL[m]["reading_AYS8"] = rd
    P(f"  READING{' (pre-registered)' if m == 'warp' else ' (descriptive)'} for AYS8 [{m}]: b {SL[m]['rows'][AYS8]['b']:.4f} CI "
      f"[{ci[0]:.3f},{ci[1]:.3f}] vs b_gen {b_gen:.4f} / b_usm {b_usm:.4f} -> {rd}")
    xa, ya = xy(AYS8, m)
    pred = np.exp(b_gen * xa)
    mo = np.array([val(c, ORIGIN, m) for c in CLIPS])
    obs = np.exp(ya)
    SL[m]["pred_AYS8"] = dict(per_clip_pred=pred.tolist(), per_clip_obs=obs.tolist(),
                              pred_ratio_of_means=float((mo * pred).mean() / mo.mean()),
                              obs_ratio_of_means=float((mo * obs).mean() / mo.mean()),
                              pred_geo=float(np.exp((b_gen * xa).mean())), obs_geo=float(np.exp(ya.mean())))
    P(f"  AYS8 {m} ratio per clip, predicted from b_gen | observed:")
    P("    " + "  ".join(f"{c}:{p:.3f}|{o:.3f}" for c, p, o in zip(CLIPS, pred, obs)))
    P(f"    ratio of 12-clip means: predicted {SL[m]['pred_AYS8']['pred_ratio_of_means']:.4f}  observed "
      f"{SL[m]['pred_AYS8']['obs_ratio_of_means']:.4f};  geo-mean predicted {SL[m]['pred_AYS8']['pred_geo']:.4f}  observed "
      f"{SL[m]['pred_AYS8']['obs_geo']:.4f}")

# =========================================================================================== F-direct
P("\n--- F-direct: deliverable 8x2 vs AYS8 origin 8x2 (same 16 evals/window, same noise) ---")
wd = np.array(TAB[f"warp|{DELIV}"]); wa = np.array(TAB[f"warp|{AYS8}"])
td = np.array(TAB[f"tLP|{DELIV}"]); ta = np.array(TAB[f"tLP|{AYS8}"])
sd = np.array([sharp(c, DELIV) for c in CLIPS]); sa = np.array([sharp(c, AYS8) for c in CLIPS])
P("  per clip warp ratio deliverable/AYS8: " + "  ".join(f"{c}:{r:.3f}" for c, r in zip(CLIPS, wd / wa)))
P("  per clip sharpness ratio deliverable/AYS8: " + "  ".join(f"{c}:{r:.3f}" for c, r in zip(CLIPS, sd / sa)))
b_gen_w = SL["warp"]["b_gen"]
fd = dict(warp_ratio_of_means=float(wd.mean() / wa.mean()), warp_higher=int((wd > wa).sum()),
          tLP_ratio_of_means=float(td.mean() / ta.mean()), tLP_higher=int((td > ta).sum()),
          sharp_geo=float(np.exp(np.log(sd / sa).mean())),
          warp_pred_from_b_gen=float(np.exp(b_gen_w * np.log(sd / sa)).mean()),
          warp_geo=float(np.exp(np.log(wd / wa).mean())))
P(f"  warp: ratio of means {fd['warp_ratio_of_means']:.4f} ({100 * (fd['warp_ratio_of_means'] - 1):+.1f}%), deliverable higher "
  f"on {fd['warp_higher']}/12; geo {fd['warp_geo']:.4f}; predicted from b_gen and the sharpness ratio (geo "
  f"{fd['sharp_geo']:.4f}): {fd['warp_pred_from_b_gen']:.4f} (mean of per-clip predictions)")
P(f"  tLP : ratio of means {fd['tLP_ratio_of_means']:.4f} ({100 * (fd['tLP_ratio_of_means'] - 1):+.1f}%), deliverable higher on "
  f"{fd['tLP_higher']}/12")
lp = {c: (ROWS["cells"][c][DELIV]["lpips"] - ROWS["cells"][c][AYS8]["lpips"]) for c in CLIPS}
fd["lpips_unreg_mean"] = float(np.mean(list(lp.values())))
P(f"  LPIPS (C1, UNREG ROW values) deliverable - AYS8: mean {fd['lpips_unreg_mean']:+.5f}")
if REG and os.path.exists(REG):
    rs = json.load(open(REG))
    k1 = [k for k in rs["contrasts"] if k.startswith("C1:")][0]
    for v in ("REG_FRAME", "REG_CLIP"):
        s = rs["contrasts"][k1][v]
        fd[f"lpips_{v}"] = dict(mean=s["mean"], ci_pct=s["ci_pct"], improved=s["improved"])
        P(f"  LPIPS (C1, {v}) deliverable - AYS8: mean {s['mean']:+.5f}, pct95 [{s['ci_pct'][0]:+.4f},{s['ci_pct'][1]:+.4f}], "
          f"negative {s['improved']}/12   [{REG}]")
P(f"  => the deliverable's extra LPIPS over AYS8 ({fd['lpips_unreg_mean']:+.4f} UNREG) comes with "
  f"{100 * (fd['warp_ratio_of_means'] - 1):+.1f}% warp and {100 * (fd['tLP_ratio_of_means'] - 1):+.1f}% tLP relative to AYS8 origin")

# =========================================================================================== context
P("\n--- CONTEXT (descriptive) ---")
wt5p, wot5p = np.array(TAB[f"warp|{T5P}"]), np.array(TAB[f"warp|{OT5P}"])
P(f"  matched sampler T5@1.00 pad8: deliverable / origin warp ratio of means {wt5p.mean() / wot5p.mean():.4f} "
  f"(deliverable higher on {(wt5p > wot5p).sum()}/12); origin T5pad vs Karras origin 8x2 {np.mean(wot5p) / np.mean(TAB[f'warp|{ORIGIN}']):.4f}")
P(f"  shipped config (deliverable T5@1.00 unpadded) vs Karras origin 8x2: warp {100 * inc[T5N]:+.1f}% ({worse[T5N]}/12)  "
  f"[4-clip judge J1: +21.99 %]")
P(f"  GT right eye mean warp {np.mean(TAB['warp|GT']):.5f}; tLP {np.mean(TAB['tLP|GT']):.4f}")

# =========================================================================================== figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, m in zip(axs, ("warp", "tLP")):
        for lab, col, mk in ((DELIV, "#d62728", "o"), (S25, "#9467bd", "s"), (AYS8, "#1f77b4", "D"), (T5N, "#ff7f0e", "^"),
                             ("Rk", "#7f7f7f", "x")):
            x, y = xy(lab, m)
            ax.scatter(x, y, c=col, marker=mk, label=NAME[lab], s=34)
        xx = np.linspace(0, 0.25, 10)
        ax.plot(xx, SL[m]["b_gen"] * xx, color="#d62728", lw=1, ls="--", label=f"b_gen {SL[m]['b_gen']:.2f} (deliv+s25)")
        ax.plot(xx, SL[m]["b_usm"] * xx, color="#7f7f7f", lw=1, ls=":", label=f"b_usm {SL[m]['b_usm']:.2f} (unsharp mask)")
        ax.plot(xx, SL[m]["rows"][AYS8]["b"] * xx, color="#1f77b4", lw=1, label=f"AYS8 b {SL[m]['rows'][AYS8]['b']:.2f}")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("ln(sharpness / Karras origin)"); ax.set_ylabel(f"ln({m} / Karras origin)")
        ax.set_title(f"{m} vs sharpness, 12 clips per row")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    P(f"\nfigure: {OUT_PNG}")
except Exception as e:  # the figure is optional; quote the error
    P(f"\nfigure NOT written: {e!r}")

json.dump(dict(gates=gates, tables=TAB, increase_vs_origin=inc, worse_vs_origin=worse, f_literal_same=same,
               slopes=SL, f_direct=fd, sources=dict(temporal=f"{O}/temporal_v1", v1=VD, rows=f"{R}/ROWS_pass1.json",
                                                    reg=REG)),
          open(OUT_J, "w"), indent=1)
open(OUT_T, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT_T} {OUT_J}")
