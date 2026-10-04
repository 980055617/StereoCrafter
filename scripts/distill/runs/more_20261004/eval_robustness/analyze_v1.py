#!/usr/bin/env python
"""Analysis of the registered-GT re-score (eval_robustness lane).  CPU only.  Definitions: PREREG.txt.

usage: python analyze_v1.py <score_dir> <published_rows.json> <out_table.txt> <out_stats.json> <out_perframe.csv>
"""
import csv
import itertools
import json
import math
import os
import sys

import numpy as np
from scipy import stats

SCORE, PUB, OUT_T, OUT_J, OUT_CSV = sys.argv[1:6]
pub = json.load(open(PUB))
CLIPS = pub["clips"]
ORIGIN = "origin_ll"
DELIV = "mstudent2_step800_deliv_ll"
T5 = "deliv_g100_T5nat"
S25 = "s25_ll"
T5P = "deliv_g100_T5pad"
MAMBA = "mamba_ll"
ROWS = [ORIGIN, DELIV, T5, S25, T5P, MAMBA]
NAME = {ORIGIN: "origin (deployed 8 steps @1.01)", DELIV: "DELIVERABLE (deployed 8 steps @1.01)",
        T5: "deliverable T5 @1.00 (unpadded)", S25: "origin 25 steps (teacher)",
        T5P: "deliverable T5 @1.00 (padded)", MAMBA: "shipped 5-slot Mamba"}
VARS = ["UNREG", "REG_CLIP", "REG_FRAME", "REG_FRAME_RAW", "BLK_FRAME", "BLK_LOCAL"]
PRIMARY = "REG_FRAME"
CONTRASTS = [(DELIV, ORIGIN), (T5, ORIGIN), (S25, ORIGIN), (T5P, ORIGIN), (MAMBA, ORIGIN),
             (DELIV, S25), (T5, DELIV), (T5, T5P)]
B_MAIN, B_HIER, SEED = 100_000, 20_000, 20261004

D = {c: json.load(open(os.path.join(SCORE, f"{c}.json"))) for c in CLIPS}
# PREREG_ADDENDUM_boundary.txt: a clip re-scored with the widened grid replaces its score_v1 file in the primary
# analysis; the original-range file is kept for the sensitivity printout.
WIDE = sys.argv[6] if len(sys.argv) > 6 else ""
D_ORIG = {}
for c in CLIPS:
    pw = os.path.join(WIDE, f"{c}.json") if WIDE else ""
    if pw and os.path.exists(pw):
        D_ORIG[c] = D[c]
        D[c] = json.load(open(pw))
ILL = [c for c in CLIPS if D[c]["reg"]["boundary_raw_frames"] or D[c]["reg"]["boundary_clip"]]
L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


def clipmean(c, row, var):
    return float(D[c]["configs"][row]["lpips_clip"][var])


def frames(c, row, var):
    return np.asarray(D[c]["configs"][row]["lpips"][var], np.float64)


# ============================================================================================ gates
gates = {}
P("=" * 140)
P("REGISTERED-GT ROBUSTNESS OF THE THESIS LPIPS NUMBERS -- 12 test clips, lossless FFV1 renders, LPIPS-alex, SCORE_STEP=4")
P(f"scores: {SCORE}   scorer: scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py   prereg: PREREG.txt")
P("=" * 140)
P("\n--- GATES ---")
g0_bad = []
for c in CLIPS:
    for r in ROWS:
        cf, pb = D[c]["configs"][r], pub["cells"][c][r]
        if abs(cf["lpips_clip"]["UNREG"] - pb["lpips"]) > 1e-4 or (cf["dy"], cf["dx"]) != (pb["dy"], pb["dx"]) \
                or cf["n"] != pb["n"]:
            g0_bad.append((c, r, cf["lpips_clip"]["UNREG"], pb["lpips"], cf["dy"], cf["dx"], pb["dy"], pb["dx"]))
maxdev = max(abs(D[c]["configs"][r]["lpips_clip"]["UNREG"] - pub["cells"][c][r]["lpips"]) for c in CLIPS for r in ROWS)
means_ok = all(f"{np.mean([clipmean(c, r, 'UNREG') for c in CLIPS]):.4f}" == f"{pub['means'][r]:.4f}" for r in ROWS)
gates["G0"] = dict(pass_=not g0_bad and means_ok, cells_failing=g0_bad, max_abs_dev=maxdev, means_4dp_equal=means_ok)
P(f"G0 UNREG reproduces 72 published cells within 1e-4 and offsets/n: {'PASS' if not g0_bad else 'FAIL'}  "
  f"(max |dev| {maxdev:.2e}); 12-clip means equal at 4 dp: {'PASS' if means_ok else 'FAIL'}")
for b in g0_bad:
    P(f"    FAIL {b}")

GEO = {"0147": (8, -1, -41), "0141": (72, 0, -59), "0042": (32, -1, -36), "0128": (116, -1, -34),
       "0301": (40, 1, -15), "0204": (64, -1, -17)}
g1a = []
for c, (f, gy, gx) in GEO.items():
    a = D[c]["anchor_trainBR"][str(f)]
    sy, sx = D[c]["reg"]["raw_ddy"][f], D[c]["reg"]["raw_ddx"][f]
    ok = abs(a["ddy"] - gy) <= 1 and abs(a["ddx"] - gx) <= 2
    g1a.append(dict(clip=c, frame=f, published=(gy, gx), trainBR=(a["ddy"], a["ddx"]), splatBR=(sy, sx), ok=ok))
regb = json.load(open("scripts/distill/runs/clean_controls/regB/registration/registration_perframe.json"))["0301"]
rbraw = {p["frame"]: (p["ddy"], p["ddx"]) for p in regb["per_frame_raw"]}
an = D["0301"]["anchor_trainBR"]
same = sum(int((an[k]["ddy"], an[k]["ddx"]) == rbraw[int(k)]) for k in an)
g1b = dict(identical=same, total=len(an), ok=same >= 0.9 * len(an))
REGB_CLIP = {"0301": (0, -12), "0204": (-1, -17)}
g1c = [dict(clip=c, regB=v, mine=(D[c]["reg"]["clip_ddy"], D[c]["reg"]["clip_ddx"]),
            ok=abs(D[c]["reg"]["clip_ddy"] - v[0]) <= 1 and abs(D[c]["reg"]["clip_ddx"] - v[1]) <= 1)
       for c, v in REGB_CLIP.items()]
gates["G1"] = dict(pass_=all(x["ok"] for x in g1a) and g1b["ok"] and all(x["ok"] for x in g1c), a=g1a, b=g1b, c=g1c)
P(f"G1 registration anchors: {'PASS' if gates['G1']['pass_'] else 'FAIL'}")
for x in g1a:
    P(f"    (a) {x['clip']} f{x['frame']:<3d} review {x['published']} -> train-BR {x['trainBR']}  splat-BR {x['splatBR']}  "
      f"{'ok' if x['ok'] else 'FAIL'}")
P(f"    (b) regB 0301 per-frame raw identical on {same}/{len(an)} scored frames  {'ok' if g1b['ok'] else 'FAIL'}")
for x in g1c:
    P(f"    (c) {x['clip']} regB clip optimum {x['regB']} -> mine {x['mine']}  {'ok' if x['ok'] else 'FAIL'}")
g2 = []
for c in CLIPS:
    r = D[c]["reg"]
    md5s = {D[c]["configs"][k]["md5_left"] for k in ROWS}
    bnd_ = bool(r["boundary_raw_frames"]) or r["boundary_clip"]
    # a boundary hit is handled by PREREG_ADDENDUM_boundary.txt: widened once (file from WIDE) -> if it persists the
    # clip's REG_FRAME is flagged ILL-POSED; an un-widened boundary hit is a gate failure
    ok = (r["smooth_mean_psnr"] >= r["zero_mean_psnr"] and r["clip_mean_psnr"] >= r["zero_mean_psnr"]
          and len(md5s) == 1 and (not bnd_ or c in D_ORIG))
    g2.append(dict(clip=c, ok=ok, widened=c in D_ORIG, boundary_raw=len(r["boundary_raw_frames"]),
                   boundary_clip=r["boundary_clip"], left_md5_unique=len(md5s), ill_posed=c in ILL))
gates["G2"] = dict(pass_=all(x["ok"] for x in g2), per_clip=g2, widened=sorted(D_ORIG), ill_posed=ILL)
P(f"G2 registered PSNR >= unregistered, bit-identical left halves, boundary hits widened once: "
  f"{'PASS' if gates['G2']['pass_'] else 'FAIL'}" + "".join(f"\n    FAIL {x}" for x in g2 if not x["ok"]))
for c in sorted(D_ORIG):
    ro, rw = D_ORIG[c]["reg"], D[c]["reg"]
    P(f"    widened (addendum): {c} grid ddy {D[c]['search']['ddy']} ddx {D[c]['search']['ddx']}: boundary frames "
      f"{len(ro['boundary_raw_frames'])} -> {len(rw['boundary_raw_frames'])}, clip optimum ({ro['clip_ddy']},{ro['clip_ddx']}) "
      f"-> ({rw['clip_ddy']},{rw['clip_ddx']}), REG_FRAME ddx range [{min(rw['smooth_ddx'])},{max(rw['smooth_ddx'])}] "
      f"ddy range [{min(rw['smooth_ddy'])},{max(rw['smooth_ddy'])}]" + ("  -> ILL-POSED (multi-plane)" if c in ILL else ""))

# ============================================================================================ registration summary
P("\n--- REGISTRATION (target = model input BR from the splatting video, holes excluded; identical for every row) ---")
P(f"  {'clip':5s} {'quadrant':>10s} {'clip shift':>11s} {'REG_FRAME ddx range':>20s} {'ddy range':>10s} "
  f"{'PSNR(TR,BR) dB: unreg':>22s} {'clip':>6s} {'frame':>6s} {'raw':>6s}  {'trainBR==splatBR shift':>22s} "
  f"{'trainBR vs splatBR':>18s}")
regsum = {}
for c in CLIPS:
    r = D[c]["reg"]
    fr = D[c]["frames"]
    sdx = [r["smooth_ddx"][f] for f in fr]
    sdy = [r["smooth_ddy"][f] for f in fr]
    an = D[c]["anchor_trainBR"]
    agree = sum(int((an[str(f)]["ddy"], an[str(f)]["ddx"]) == (r["raw_ddy"][f], r["raw_ddx"][f])) for f in fr)
    tbsb = np.mean([an[str(f)]["trainBR_vs_splatBR_psnr"] for f in fr])
    regsum[c] = dict(clip_shift=(r["clip_ddy"], r["clip_ddx"]), frame_ddx=(min(sdx), max(sdx)),
                     frame_ddy=(min(sdy), max(sdy)), psnr_unreg=r["zero_mean_psnr"], psnr_clip=r["clip_mean_psnr"],
                     psnr_frame=r["smooth_mean_psnr"], psnr_raw=r["raw_mean_psnr"], trainBR_agree=agree,
                     n_scored=len(fr), trainBR_vs_splatBR_psnr=tbsb)
    q = D[c]["quadrant"]
    P(f"  {c:5s} {q[0]:>4d}x{q[1]:<5d} ({r['clip_ddy']:+3d},{r['clip_ddx']:+4d}) {f'[{min(sdx)},{max(sdx)}]':>20s} "
      f"{f'[{min(sdy)},{max(sdy)}]':>10s} {r['zero_mean_psnr']:22.2f} {r['clip_mean_psnr']:6.2f} "
      f"{r['smooth_mean_psnr']:6.2f} {r['raw_mean_psnr']:6.2f}  {f'{agree}/{len(fr)}':>22s} {tbsb:15.2f} dB")

# ============================================================================================ absolute numbers
P("\n--- 12-CLIP MEAN LPIPS PER ROW AND GT VARIANT (lower is better).  BLK_* are block-metric values, only comparable "
  "with each other ---")
P(f"  {'row':38s}" + "".join(f"{v:>14s}" for v in VARS))
absm = {}
for r in ROWS:
    absm[r] = {v: float(np.mean([clipmean(c, r, v) for c in CLIPS])) for v in VARS}
    P(f"  {NAME[r]:38s}" + "".join(f"{absm[r][v]:14.4f}" for v in VARS))
P(f"  {'change vs UNREG (origin row)':38s}" + "".join(f"{absm[ORIGIN][v] - absm[ORIGIN]['UNREG']:+14.4f}" for v in VARS))
P(f"  {'relative change (origin row)':38s}" + "".join(
    f"{100 * (absm[ORIGIN][v] / absm[ORIGIN]['UNREG'] - 1):+13.1f}%" for v in VARS))
FMT = {"4400x4400 tiles": [c for c in CLIPS if D[c]["quadrant"][0] == 2200],
       "2160x3840 tiles": [c for c in CLIPS if D[c]["quadrant"][0] == 1080]}
P("\n  per tile format (6 clips each): mean LPIPS and delta vs origin")
fmt_tab = {}
for fk, cl in FMT.items():
    P(f"  [{fk}: {' '.join(cl)}]   columns per variant: mean LPIPS, delta vs origin, clips improved")
    P(f"  {'row':38s}" + "".join(f"{v:<22s}" for v in VARS))
    for r in ROWS:
        vals = {v: float(np.mean([clipmean(c, r, v) for c in cl])) for v in VARS}
        dv = {v: float(np.mean([clipmean(c, r, v) - clipmean(c, ORIGIN, v) for c in cl])) for v in VARS}
        nimp = {v: int(sum(clipmean(c, r, v) < clipmean(c, ORIGIN, v) for c in cl)) for v in VARS}
        fmt_tab[f"{fk} | {r}"] = dict(mean=vals, delta=dv, improved=nimp)
        P(f"  {NAME[r]:38s}" + "".join(f"{vals[v]:9.4f}" + (f" {dv[v]:+.4f} {nimp[v]}/6" if r != ORIGIN else " " * 13)
                                       for v in VARS))
P("\n  right-eye PSNR (dB, from the mean MSE over frames, averaged over clips) -- the distortion side:")
P(f"  {'row':38s}" + "".join(f"{v:>14s}" for v in VARS[:4]))
rps = {}
for r in ROWS:
    rps[r] = {v: float(np.mean([D[c]["configs"][r]["rPSNR"][v] for c in CLIPS])) for v in VARS[:4]}
    P(f"  {NAME[r]:38s}" + "".join(f"{rps[r][v]:14.3f}" for v in VARS[:4]))

# ============================================================================================ per clip
P("\n--- PER CLIP: LPIPS and delta vs origin, UNREG | REG_CLIP | REG_FRAME (primary) ---")
for v in ["UNREG", "REG_CLIP", "REG_FRAME"]:
    P(f"  [{v}]  {'':30s}" + "".join(f"{c:>8s}" for c in CLIPS) + f"{'mean':>9s} {'improved':>9s}")
    for r in ROWS:
        vals = [clipmean(c, r, v) for c in CLIPS]
        P(f"  {NAME[r]:38s}" + "".join(f"{x:8.4f}" for x in vals) + f"{np.mean(vals):9.4f}")
    for r in ROWS[1:]:
        d = [clipmean(c, r, v) - clipmean(c, ORIGIN, v) for c in CLIPS]
        P(f"  {'  d ' + NAME[r]:38s}" + "".join(f"{x:+8.4f}" for x in d) + f"{np.mean(d):+9.4f} "
          f"{sum(x < 0 for x in d):>5d}/12")
    P()


# ============================================================================================ statistics
def bca(d, boot, theta, alpha=0.05):
    n = len(d)
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
    prop = min(max(prop, 1.0 / len(boot)), 1 - 1.0 / len(boot))
    z0 = stats.norm.ppf(prop)
    jk = np.array([np.mean(np.delete(d, i)) for i in range(n)])
    num = np.sum((jk.mean() - jk) ** 3)
    den = 6.0 * (np.sum((jk.mean() - jk) ** 2) ** 1.5)
    a = num / den if den > 0 else 0.0
    out = []
    for z in (stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)):
        q = stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.quantile(boot, q)))
    return out


def contrast_stats(rowA, rowB, var, clips=None):
    clips = CLIPS if clips is None else clips
    d = np.array([clipmean(c, rowA, var) - clipmean(c, rowB, var) for c in clips])
    n = len(d)
    SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, n, size=(B_MAIN, n))].mean(1)
    pct = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    tq = stats.t.ppf(0.975, n - 1)
    se = d.std(ddof=1) / math.sqrt(n)
    k = int(np.sum(d < 0))
    p_sign = float(stats.binomtest(k, n, 0.5).pvalue)
    p_wil = float(stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue)
    perm = (SIGNS * np.abs(d)).mean(1)
    p_perm = float(np.mean(np.abs(perm) >= abs(d.mean()) - 1e-15))
    # two-level bootstrap: clips, then frames within clip (paired frames)
    rng2 = np.random.default_rng(SEED + 1)
    fd = [frames(c, rowA, var) - frames(c, rowB, var) for c in clips]
    hb = np.empty(B_HIER)
    for b in range(B_HIER):
        cs = rng2.integers(0, n, size=n)
        hb[b] = np.mean([fd[i][rng2.integers(0, len(fd[i]), size=len(fd[i]))].mean() for i in cs])
    worst = int(np.argmax(d))
    return dict(mean=float(d.mean()), per_clip=d.tolist(), improved=k, n=n, ci_pct=pct, ci_bca=bca(d, boot, d.mean()),
                ci_t=[float(d.mean() - tq * se), float(d.mean() + tq * se)], sd=float(d.std(ddof=1)),
                p_sign=p_sign, p_wilcoxon=p_wil, p_signflip=p_perm,
                ci_hier=[float(np.quantile(hb, 0.025)), float(np.quantile(hb, 0.975))],
                worst=(clips[worst], float(d[worst])), best=(clips[int(np.argmin(d))], float(d.min())))


ST = {}
P("--- PAIRED CONTRASTS, clip = unit (n = 12).  CIs: percentile bootstrap B=100k (DESCRIPTIVE at n=12), BCa, Student-t,")
P("    two-level (clips then frames) bootstrap B=20k.  p: exact two-sided sign test / Wilcoxon signed-rank / sign-flip ---")
for (a, b) in CONTRASTS:
    key = f"{a} - {b}"
    ST[key] = {}
    P(f"\n  {NAME[a]}  minus  {NAME[b]}")
    P(f"    {'variant':14s} {'mean d':>8s} {'impr':>6s} {'pct95 CI':>20s} {'BCa95 CI':>20s} {'t95 CI':>20s} "
      f"{'hier95 CI':>20s} {'p_sign':>8s} {'p_wilc':>8s} {'p_flip':>8s} {'worst clip':>16s}")
    for v in VARS:
        s = contrast_stats(a, b, v)
        ST[key][v] = s
        f2 = lambda x: f"[{x[0]:+.4f},{x[1]:+.4f}]"
        P(f"    {v:14s} {s['mean']:+8.4f} {s['improved']:>3d}/12 {f2(s['ci_pct']):>20s} {f2(s['ci_bca']):>20s} "
          f"{f2(s['ci_t']):>20s} {f2(s['ci_hier']):>20s} {s['p_sign']:8.5f} {s['p_wilcoxon']:8.5f} "
          f"{s['p_signflip']:8.5f} {s['worst'][0]:>6s} {s['worst'][1]:+.4f}")

# ============================================================================================ verdicts
P("\n--- PRE-REGISTERED VERDICTS ---")
VER = {}


def survives(key, var):
    s = ST[key][var]
    return s["mean"] < 0 and s["ci_pct"][1] < 0 and s["improved"] >= 10


for tag, (a, b) in (("V1", (DELIV, ORIGIN)), ("V2", (T5, ORIGIN)), ("V3", (S25, ORIGIN))):
    key = f"{a} - {b}"
    sv_f, sv_c = survives(key, "REG_FRAME"), survives(key, "REG_CLIP")
    ratio = ST[key]["REG_FRAME"]["mean"] / ST[key]["UNREG"]["mean"]
    twelve = {v: ST[key][v]["improved"] == 12 for v in VARS}
    VER[tag] = dict(contrast=key, survives_REG_FRAME=sv_f, robust_REG_CLIP=sv_c, ratio_frame_over_unreg=ratio,
                    magnitude_preserved=0.5 <= ratio <= 2.0, twelve_of_twelve=twelve)
    P(f"  {tag} {NAME[a]} vs {NAME[b]}: REG_FRAME {'SURVIVES' if sv_f else 'DOES NOT SURVIVE'}; "
      f"REG_CLIP {'holds -> ROBUST' if sv_c and sv_f else 'does not hold'}; "
      f"magnitude ratio REG_FRAME/UNREG {ratio:.2f} ({'preserved' if 0.5 <= ratio <= 2.0 else 'NOT preserved'}); "
      f"12/12 under: " + ", ".join(f"{v}={'yes' if twelve[v] else 'no (' + str(ST[key][v]['improved']) + ')'}"
                                   for v in VARS))
SENS = {}
if ILL:
    keep = [c for c in CLIPS if c not in ILL]
    P(f"  SENSITIVITY (addendum): without the ILL-POSED clip(s) {ILL}, n={len(keep)}:")
    for tag, (a, b) in (("V1", (DELIV, ORIGIN)), ("V2", (T5, ORIGIN)), ("V3", (S25, ORIGIN))):
        for v in ("REG_FRAME", "REG_CLIP", "UNREG"):
            s_ = contrast_stats(a, b, v, keep)
            need = math.ceil(len(keep) * 10 / 12)
            SENS[f"{tag} {v}"] = s_
            P(f"    {tag} {v:10s} mean {s_['mean']:+.4f}  improved {s_['improved']}/{len(keep)}  pct95 "
              f"[{s_['ci_pct'][0]:+.4f},{s_['ci_pct'][1]:+.4f}]  p_sign {s_['p_sign']:.5f}  "
              f"{'meets (a)+(b) and p_sign<0.05' if s_['mean'] < 0 and s_['ci_pct'][1] < 0 and s_['p_sign'] < 0.05 else 'does not meet'}")
if D_ORIG:
    P("  ORIGINAL-RANGE (score_v1) values of the widened clip(s), delta vs origin:")
    for c in sorted(D_ORIG):
        for v in ("REG_FRAME", "REG_CLIP", "BLK_LOCAL"):
            P(f"    {c} {v:10s} " + "  ".join(
                f"{NAME[r][:22]}: orig-range {D_ORIG[c]['configs'][r]['lpips_clip'][v] - D_ORIG[c]['configs'][ORIGIN]['lpips_clip'][v]:+.4f}"
                f" / widened {clipmean(c, r, v) - clipmean(c, ORIGIN, v):+.4f}" for r in (DELIV, T5, S25)))
order = absm[S25][PRIMARY] < absm[DELIV][PRIMARY] < absm[ORIGIN][PRIMARY]
VER["V4"] = dict(order_preserved=order, means={r: absm[r][PRIMARY] for r in (S25, DELIV, ORIGIN)})
P(f"  V4 ordering s25 < deliverable < origin under REG_FRAME: {'PRESERVED' if order else 'NOT PRESERVED'} "
  f"({absm[S25][PRIMARY]:.4f} < {absm[DELIV][PRIMARY]:.4f} < {absm[ORIGIN][PRIMARY]:.4f})")

# ============================================================================================ per-frame analysis
P("\n--- PER-FRAME ANALYSIS (deliverable - origin per frame; descriptive, no frame-pooled tests) ---")
PF = {}
for (a, b) in ((DELIV, ORIGIN), (T5, ORIGIN), (S25, ORIGIN)):
    for v in ("UNREG", "REG_FRAME"):
        key = f"{a} - {b} | {v}"
        rows = []
        for c in CLIPS:
            d = frames(c, a, v) - frames(c, b, v)
            n = len(d)
            k20 = math.ceil(0.2 * n)
            srt = np.sort(d)
            share = float(srt[:k20].sum() / d.sum()) if d.sum() < 0 else float("nan")
            hole = np.asarray(D[c]["hole_frac"])
            regp = np.asarray([D[c]["reg"]["smooth_psnr"][f] for f in D[c]["frames"]])
            fidx = np.asarray(D[c]["frames"])
            wk = np.asarray(D[c]["window_k"])
            rho_h = float(stats.spearmanr(d, hole).statistic) if np.std(hole) > 0 else float("nan")
            rho_r = float(stats.spearmanr(d, regp).statistic)
            rho_t = float(stats.spearmanr(d, fidx).statistic)
            half_ = n // 2
            rows.append(dict(clip=c, n=n, frac_improved=float(np.mean(d < 0)), mean=float(d.mean()),
                             median=float(np.median(d)), sd=float(d.std(ddof=1)), min=float(d.min()),
                             max=float(d.max()), top20_share=share, rho_hole=rho_h, rho_regpsnr=rho_r,
                             rho_time=rho_t, first_half=float(d[:half_].mean()), second_half=float(d[half_:].mean()),
                             win0=float(d[wk == 0].mean()), win_later=float(d[wk > 0].mean())))
        PF[key] = rows
        med_frac = float(np.median([r["frac_improved"] for r in rows]))
        shares = [r["top20_share"] for r in rows if not math.isnan(r["top20_share"])]
        med_share = float(np.median(shares)) if shares else float("nan")
        if med_frac >= 0.75 and med_share <= 0.40:
            verdict = "UNIFORM"
        elif med_share > 0.60 or med_frac < 0.60:
            verdict = "CONCENTRATED"
        else:
            verdict = "INTERMEDIATE"
        PF[key + " | summary"] = dict(median_frac_improved=med_frac, median_top20_share=med_share, verdict=verdict,
                                      n_frames_total=int(sum(r["n"] for r in rows)),
                                      frac_improved_pooled=float(np.mean(np.concatenate(
                                          [frames(c, a, v) - frames(c, b, v) for c in CLIPS]) < 0)))
        if a in (DELIV,) or v == "REG_FRAME":
            P(f"\n  {NAME[a]} - {NAME[b]}  [{v}]  -> {verdict}  (median clip: {med_frac:.2f} of frames improved, "
              f"top-20% share {med_share:.2f}; pooled {PF[key + ' | summary']['frac_improved_pooled']:.2f} of "
              f"{PF[key + ' | summary']['n_frames_total']} frames improved)")
            if a == DELIV:
                P(f"    {'clip':5s} {'n':>3s} {'impr':>5s} {'mean':>8s} {'median':>8s} {'sd':>7s} {'min':>8s} {'max':>8s} "
                  f"{'top20':>6s} {'rho_hole':>8s} {'rho_regP':>8s} {'rho_t':>6s} {'1st half':>8s} {'2nd half':>8s} "
                  f"{'win0':>8s} {'win>=1':>8s}")
                for r in rows:
                    P(f"    {r['clip']:5s} {r['n']:3d} {r['frac_improved']:5.2f} {r['mean']:+8.4f} {r['median']:+8.4f} "
                      f"{r['sd']:7.4f} {r['min']:+8.4f} {r['max']:+8.4f} {r['top20_share']:6.2f} {r['rho_hole']:+8.2f} "
                      f"{r['rho_regpsnr']:+8.2f} {r['rho_time']:+6.2f} {r['first_half']:+8.4f} {r['second_half']:+8.4f} "
                      f"{r['win0']:+8.4f} {r['win_later']:+8.4f}")
                P(f"    median rho(delta, hole frac) {np.nanmedian([r['rho_hole'] for r in rows]):+.2f} "
                  f"({sum(r['rho_hole'] < 0 for r in rows)}/12 negative = bigger gain on frames with more holes); "
                  f"median rho(delta, frame index) {np.median([r['rho_time'] for r in rows]):+.2f}; "
                  f"window 0 vs later (mean over clips) {np.mean([r['win0'] for r in rows]):+.4f} vs "
                  f"{np.mean([r['win_later'] for r in rows]):+.4f}")
                # position in window, clip-centred residuals
                res_pos = {}
                for c in CLIPS:
                    d = frames(c, a, v) - frames(c, b, v)
                    for pos, wk_, x in zip(D[c]["window_pos"], D[c]["window_k"], d - d.mean()):
                        bin_ = "w0 pos0-2" if wk_ == 0 and pos <= 2 else ("pos3-5" if pos <= 5 else
                                                                         ("pos6-9" if pos <= 9 else "pos10-13"))
                        res_pos.setdefault(bin_, []).append(x)
                P("    clip-centred delta by position in the 14-frame sampler window: " + "  ".join(
                    f"{k_} {np.mean(res_pos[k_]):+.4f} (n={len(res_pos[k_])})"
                    for k_ in ("w0 pos0-2", "pos3-5", "pos6-9", "pos10-13") if k_ in res_pos))
                PF[key + " | position"] = {k_: [float(np.mean(v_)), len(v_)] for k_, v_ in res_pos.items()}

# between- vs within-clip variance of the per-frame deltas (why the clip is the unit)
d_by = [frames(c, DELIV, PRIMARY) - frames(c, ORIGIN, PRIMARY) for c in CLIPS]
within_sd = float(np.sqrt(np.mean([x.var(ddof=1) for x in d_by])))
between_sd = float(np.std([x.mean() for x in d_by], ddof=1))
ac1 = float(np.mean([np.corrcoef(x[:-1], x[1:])[0, 1] for x in d_by]))
P(f"\n  deliverable - origin [REG_FRAME]: within-clip SD of per-frame deltas {within_sd:.4f}, between-clip SD of clip "
  f"means {between_sd:.4f}, mean lag-1 autocorrelation of the per-frame delta series (frames 4 apart) {ac1:+.2f}")
PF["variance"] = dict(within_sd=within_sd, between_sd=between_sd, lag1_autocorr=ac1)

# ============================================================================================ write
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    hdr = ["clip", "frame", "window_k", "window_pos", "hole_frac", "reg_ddy", "reg_ddx", "reg_psnr", "unreg_psnr"]
    for r in ROWS:
        for v in VARS:
            hdr.append(f"{r}|{v}")
    w.writerow(hdr)
    for c in CLIPS:
        dd = D[c]
        for j, f in enumerate(dd["frames"]):
            row = [c, f, dd["window_k"][j], dd["window_pos"][j], f"{dd['hole_frac'][j]:.6f}",
                   dd["reg"]["smooth_ddy"][f], dd["reg"]["smooth_ddx"][f], f"{dd['reg']['smooth_psnr'][f]:.4f}",
                   f"{dd['reg']['zero_psnr'][f]:.4f}"]
            for r in ROWS:
                for v in VARS:
                    row.append(f"{dd['configs'][r]['lpips'][v][j]:.6f}")
            w.writerow(row)
json.dump(dict(gates=gates, registration=regsum, abs_means=absm, rpsnr=rps, per_format=fmt_tab, contrasts=ST,
               verdicts=VER, sensitivity_without_ill=SENS, ill_posed=ILL, widened=sorted(D_ORIG), perframe=PF),
          open(OUT_J, "w"), indent=1, default=str)
open(OUT_T, "w").write("\n".join(L) + "\n")
print("ANALYSIS_DONE", OUT_T, OUT_J, OUT_CSV)
