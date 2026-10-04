#!/usr/bin/env python
"""ays_20261004 / robust -- analysis of the registered-GT re-score (CPU only).  Definitions: PREREG.txt (same dir).

contrast_stats() and bca() are copied VERBATIM from scripts/distill/runs/more_20261004/eval_robustness/analyze_v1.py
(same B, same seeds, same tests).  Gates R0 / R1 / R2 and the V-rule as pre-registered.

usage: python analyze_reg_v1.py <out_table.txt> <out_stats.json> [<ays5_dir> <ays5_wide_dir> <ays5_rows.json>]
  pass 1 is read from outputs/ays_20261004/robust/score_reg_v1 (+ score_reg_v1_wide for 0125, PRIMARY as in
  eval_robustness); the optional pass-2 (AYS5) dirs add contrast C6 and gate R1'.
"""
import itertools
import json
import math
import os
import sys

import numpy as np
from scipy import stats

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT_T, OUT_J = sys.argv[1], sys.argv[2]
for p in (OUT_T, OUT_J):
    assert not os.path.exists(p), f"refusing to overwrite {p}"
A5DIR = sys.argv[3] if len(sys.argv) > 3 else ""
A5WIDE = sys.argv[4] if len(sys.argv) > 4 else ""
A5ROWS = sys.argv[5] if len(sys.argv) > 5 else ""

R = "scripts/distill/runs/ays_20261004/robust"
O = "outputs/ays_20261004/robust"
ER = "outputs/more_20261004/eval_robustness"
ROWS = json.load(open(f"{R}/ROWS_pass1.json"))
CLIPS = ROWS["clips"]
ORIGIN, AYS8, DELIV, T5P, OT5P, T5N, S25 = ("origin_ll", "AYS8_origin_g101", "mstudent2_step800_deliv_ll",
                                            "deliv_g100_T5pad", "origin_g100_T5pad", "deliv_g100_T5nat", "s25_ll")
AYS5 = "AYS5pad8_origin_g100"
NAME = {ORIGIN: "Karras origin 8x2 @1.01 (deployed)", AYS8: "AYS8 origin 8x2 @1.01", DELIV: "deliverable 8x2 @1.01",
        T5P: "deliverable T5@1.00 pad8", OT5P: "origin T5@1.00 pad8", T5N: "deliverable T5@1.00 unpadded (ships)",
        S25: "origin s25 (teacher)", AYS5: "AYS5 origin @1.00 pad8"}
VARS = ["UNREG", "REG_CLIP", "REG_FRAME", "REG_FRAME_RAW", "BLK_FRAME", "BLK_LOCAL"]
SHARED_ER = [ORIGIN, DELIV, T5N, T5P, S25]
B_MAIN, B_HIER, SEED = 100_000, 20_000, 20261004
ILL_CLIP = "0125"

D = {c: json.load(open(f"{O}/score_reg_v1/{c}.json")) for c in CLIPS}
D_ORIG = {ILL_CLIP: D[ILL_CLIP]}
D[ILL_CLIP] = json.load(open(f"{O}/score_reg_v1_wide/{ILL_CLIP}.json"))
L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


def clipmean(c, row, var, DD=None):
    return float((DD or D)[c]["configs"][row]["lpips_clip"][var])


def frames(c, row, var, DD=None):
    return np.asarray((DD or D)[c]["configs"][row]["lpips"][var], np.float64)


# ------------------------------------------------------------------------------------------- verbatim (analyze_v1.py)
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


def contrast_stats(rowA, rowB, var, clips=None, DD=None):
    clips = CLIPS if clips is None else clips
    d = np.array([clipmean(c, rowA, var, DD) - clipmean(c, rowB, var, DD) for c in clips])
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
    fd = [frames(c, rowA, var, DD) - frames(c, rowB, var, DD) for c in clips]
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
# ------------------------------------------------------------------------------------------- end verbatim


P("=" * 150)
P("ays_20261004 / robust -- REGISTERED-GT re-score of the AYS contrasts (12 test clips, lossless FFV1, LPIPS-alex, SCORE_STEP=4)")
P(f"scores: {O}/score_reg_v1/<clip>.json (+ score_reg_v1_wide/0125.json PRIMARY for 0125)   scorer: "
  "scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py (+v1w), unchanged")
P("prereg: scripts/distill/runs/ays_20261004/robust/PREREG.txt")
P("=" * 150)

# =========================================================================================== gates
gates = {}
P("\n--- GATES ---")
r0_bad, r0_max = [], 0.0
for c in CLIPS:
    for r in ROWS["labels"]:
        cf, pb = D[c]["configs"][r], ROWS["cells"][c][r]
        dv = abs(cf["lpips_clip"]["UNREG"] - pb["lpips"])
        r0_max = max(r0_max, dv)
        if dv > 1e-4 or (cf["dy"], cf["dx"]) != (pb["dy"], pb["dx"]) or cf["n"] != pb["n"] or cf["path"] != pb["path"]:
            r0_bad.append((c, r, cf["lpips_clip"]["UNREG"], pb["lpips"], cf["dy"], cf["dx"], pb["dy"], pb["dx"]))
gates["R0"] = dict(pass_=not r0_bad, failing=r0_bad, max_abs_dev=r0_max)
P(f"R0 UNREG reproduces all {len(CLIPS) * len(ROWS['labels'])} published ROW cells within 1e-4 (+ offsets, n, path): "
  f"{'PASS' if not r0_bad else 'FAIL'}  (max |dev| {r0_max:.2e})")
for b in r0_bad:
    P(f"    FAIL {b}")


def r1_compare(mine, theirs, rows):
    bad, maxv, maxf = [], 0.0, 0.0
    for k in ("clip_ddy", "clip_ddx", "smooth_ddy", "smooth_ddx", "raw_ddy", "raw_ddx"):
        if mine["reg"][k] != theirs["reg"][k]:
            bad.append(("reg", k))
    if mine["frames"] != theirs["frames"]:
        bad.append(("frames",))
    for r in rows:
        for v in VARS:
            dv = abs(mine["configs"][r]["lpips_clip"][v] - theirs["configs"][r]["lpips_clip"][v])
            maxv = max(maxv, dv)
            if dv > 1e-6:
                bad.append((r, v, dv))
            fa, fb = np.asarray(mine["configs"][r]["lpips"][v]), np.asarray(theirs["configs"][r]["lpips"][v])
            maxf = max(maxf, float(np.max(np.abs(fa - fb))) if fa.shape == fb.shape else float("inf"))
    return bad, maxv, maxf


r1 = []
for c in CLIPS:
    mine_o = json.load(open(f"{O}/score_reg_v1/{c}.json"))
    theirs_o = json.load(open(f"{ER}/score_v1/{c}.json"))
    bad, mv, mf = r1_compare(mine_o, theirs_o, SHARED_ER)
    r1.append(dict(clip=c, file="score_v1", bad=bad, max_clip_dev=mv, max_frame_dev=mf))
bad, mv, mf = r1_compare(D[ILL_CLIP], json.load(open(f"{ER}/score_v1_wide/{ILL_CLIP}.json")), SHARED_ER)
r1.append(dict(clip=ILL_CLIP, file="score_v1_wide", bad=bad, max_clip_dev=mv, max_frame_dev=mf))
r1_pass = all(not x["bad"] for x in r1)
gates["R1"] = dict(pass_=r1_pass, per_file=r1)
P(f"R1 the {len(SHARED_ER)} rows shared with eval_robustness ({', '.join(SHARED_ER)}) equal its score_v1 (+ score_v1_wide 0125) "
  f"in all 6 variants within 1e-6, registration arrays identical: {'PASS' if r1_pass else 'FAIL'}  "
  f"(max |clip dev| {max(x['max_clip_dev'] for x in r1):.1e}, max |frame dev| {max(x['max_frame_dev'] for x in r1):.1e})")
for x in r1:
    if x["bad"]:
        P(f"    FAIL {x['clip']} {x['file']}: {x['bad'][:6]}")
r2 = {c: len({D[c]["configs"][r]["md5_left"] for r in ROWS["labels"]}) for c in CLIPS}
r2o = len({D_ORIG[ILL_CLIP]["configs"][r]["md5_left"] for r in ROWS["labels"]})
gates["R2"] = dict(pass_=all(v == 1 for v in r2.values()) and r2o == 1, distinct_left_md5=r2)
P(f"R2 left halves bit-identical across all {len(ROWS['labels'])} rows of each clip: "
  f"{'PASS' if gates['R2']['pass_'] else 'FAIL'}  {r2}")
rw = D[ILL_CLIP]["reg"]
P(f"    0125 (inherited addendum): widened grid ddy {D[ILL_CLIP]['search']['ddy']} ddx {D[ILL_CLIP]['search']['ddx']}; "
  f"boundary raw frames {len(D_ORIG[ILL_CLIP]['reg']['boundary_raw_frames'])} -> {len(rw['boundary_raw_frames'])}; "
  f"clip optimum ({rw['clip_ddy']},{rw['clip_ddx']}); REG_FRAME of 0125 = ILL-POSED (multi-plane), n=11 sensitivity below")

# =========================================================================================== optional pass 2 (AYS5)
D5 = None
if A5DIR:
    rows5 = json.load(open(A5ROWS))
    D5 = {c: json.load(open(f"{A5DIR}/{c}.json")) for c in CLIPS}
    D5_ORIG = {ILL_CLIP: D5[ILL_CLIP]}
    D5[ILL_CLIP] = json.load(open(f"{A5WIDE}/{ILL_CLIP}.json"))
    shared5 = [r for r in rows5["labels"] if r in ROWS["labels"]]
    r1p = []
    for c in CLIPS:
        b_, mv_, mf_ = r1_compare(json.load(open(f"{A5DIR}/{c}.json")), json.load(open(f"{O}/score_reg_v1/{c}.json")), shared5)
        r1p.append(dict(clip=c, file="pass2 vs pass1", bad=b_, max_clip_dev=mv_, max_frame_dev=mf_))
    b_, mv_, mf_ = r1_compare(D5[ILL_CLIP], D[ILL_CLIP], shared5)
    r1p.append(dict(clip=ILL_CLIP, file="pass2 wide vs pass1 wide", bad=b_, max_clip_dev=mv_, max_frame_dev=mf_))
    gates["R1p"] = dict(pass_=all(not x["bad"] for x in r1p), per_file=r1p, shared=shared5)
    P(f"R1' pass-2 shared rows ({', '.join(shared5)}) equal pass 1 within 1e-6, registration identical: "
      f"{'PASS' if gates['R1p']['pass_'] else 'FAIL'}  (max |clip dev| {max(x['max_clip_dev'] for x in r1p):.1e})")
    for x in r1p:
        if x["bad"]:
            P(f"    FAIL {x['clip']}: {x['bad'][:6]}")
    r0p_bad, r0p_unchecked, r0p_max = [], [], 0.0
    for c in CLIPS:
        cf, pb = D5[c]["configs"][AYS5], rows5["cells"][c][AYS5]
        if pb.get("lpips") is None or (isinstance(pb.get("lpips"), float) and math.isnan(pb["lpips"])):
            r0p_unchecked.append(c)
            continue
        dv = abs(cf["lpips_clip"]["UNREG"] - pb["lpips"])
        r0p_max = max(r0p_max, dv)
        if dv > 1e-4:
            r0p_bad.append((c, cf["lpips_clip"]["UNREG"], pb["lpips"]))
    r2p = {c: len({D5[c]["configs"][r]["md5_left"] for r in rows5["labels"]}) for c in CLIPS}
    gates["R0p"] = dict(pass_=not r0p_bad, failing=r0p_bad, unchecked=r0p_unchecked, max_abs_dev=r0p_max)
    gates["R2p"] = dict(pass_=all(v == 1 for v in r2p.values()), distinct_left_md5=r2p)
    P(f"R0' AYS5 UNREG vs its ROW lines within 1e-4: {'PASS' if not r0p_bad else 'FAIL'} (max |dev| {r0p_max:.2e}; "
      f"unchecked {r0p_unchecked or 'none'})   R2' left halves identical: {'PASS' if gates['R2p']['pass_'] else 'FAIL'}")
    # merged view: pass-1 rows + AYS5 from pass 2 (registration identical by R1')
    for c in CLIPS:
        D[c]["configs"][AYS5] = D5[c]["configs"][AYS5]
    D_ORIG[ILL_CLIP]["configs"][AYS5] = D5_ORIG[ILL_CLIP]["configs"][AYS5]

# =========================================================================================== registration summary
P("\n--- REGISTRATION (model-independent; identical for every row by construction and by R1) ---")
P(f"  {'clip':5s} {'clip shift':>11s} {'REG_FRAME ddx range':>20s} {'PSNR(TR,BR) unreg':>18s} {'clip':>7s} {'frame':>7s}")
for c in CLIPS:
    r = D[c]["reg"]
    sdx = [r["smooth_ddx"][f] for f in D[c]["frames"]]
    P(f"  {c:5s} ({r['clip_ddy']:+3d},{r['clip_ddx']:+4d}) {f'[{min(sdx)},{max(sdx)}]':>20s} {r['zero_mean_psnr']:18.2f} "
      f"{r['clip_mean_psnr']:7.2f} {r['smooth_mean_psnr']:7.2f}")

# =========================================================================================== absolute means
ROWS_SHOWN = [ORIGIN, AYS8, DELIV, T5P, OT5P, T5N, S25] + ([AYS5] if D5 else [])
P("\n--- 12-CLIP MEAN LPIPS PER ROW AND GT VARIANT (BLK_* only comparable with each other) ---")
P(f"  {'row':40s}" + "".join(f"{v:>14s}" for v in VARS))
absm = {}
for r in ROWS_SHOWN:
    absm[r] = {v: float(np.mean([clipmean(c, r, v) for c in CLIPS])) for v in VARS}
    P(f"  {NAME[r]:40s}" + "".join(f"{absm[r][v]:14.4f}" for v in VARS))

# =========================================================================================== contrasts
CON = [("C1", DELIV, AYS8, True), ("C2", T5P, OT5P, True), ("C3", AYS8, ORIGIN, False), ("C4", DELIV, ORIGIN, False),
       ("C5", T5P, AYS8, False), ("C7", S25, AYS8, False), ("C8", T5N, AYS8, False)]
if D5:
    CON += [("C6", T5P, AYS5, True), ("C6b", AYS5, OT5P, False)]
ST = {}
P("\n--- PER CLIP DELTAS (A - B), UNREG | REG_CLIP | REG_FRAME (primary) ---")
for tag, a, b, prim in CON:
    P(f"  {tag} {NAME[a]}  minus  {NAME[b]}{'   [PRIMARY]' if prim else '   [context]'}")
    P(f"    {'variant':10s}" + "".join(f"{c:>8s}" for c in CLIPS) + f"{'mean':>9s} {'neg':>6s}")
    for v in ("UNREG", "REG_CLIP", "REG_FRAME"):
        d = [clipmean(c, a, v) - clipmean(c, b, v) for c in CLIPS]
        P(f"    {v:10s}" + "".join(f"{x:+8.4f}" for x in d) + f"{np.mean(d):+9.4f} {sum(x < 0 for x in d):>3d}/12")

P("\n--- PAIRED CONTRASTS, clip = unit (n = 12).  CIs: percentile bootstrap B=100k (DESCRIPTIVE at n=12), BCa, Student-t,")
P("    two-level (clips then frames) bootstrap B=20k.  p: exact two-sided sign test / Wilcoxon signed-rank / sign-flip ---")
f2 = lambda x: f"[{x[0]:+.4f},{x[1]:+.4f}]"
for tag, a, b, prim in CON:
    key = f"{tag}: {a} - {b}"
    ST[key] = {}
    P(f"\n  {tag} {NAME[a]}  minus  {NAME[b]}")
    P(f"    {'variant':14s} {'mean d':>8s} {'neg':>6s} {'pct95 CI':>20s} {'BCa95 CI':>20s} {'t95 CI':>20s} "
      f"{'hier95 CI':>20s} {'p_sign':>8s} {'p_wilc':>8s} {'p_flip':>8s} {'worst clip':>16s}")
    for v in VARS:
        s = contrast_stats(a, b, v)
        ST[key][v] = s
        P(f"    {v:14s} {s['mean']:+8.4f} {s['improved']:>3d}/12 {f2(s['ci_pct']):>20s} {f2(s['ci_bca']):>20s} "
          f"{f2(s['ci_t']):>20s} {f2(s['ci_hier']):>20s} {s['p_sign']:8.5f} {s['p_wilcoxon']:8.5f} "
          f"{s['p_signflip']:8.5f} {s['worst'][0]:>6s} {s['worst'][1]:+.4f}")


# =========================================================================================== verdicts
def vrule(key, var):
    s = ST[key][var]
    return s["mean"] < 0 and s["ci_pct"][1] < 0 and s["improved"] >= 10


P("\n--- PRE-REGISTERED VERDICTS (V-rule: under REG_FRAME mean<0, pct95 upper<0, >=10/12; ROBUST iff also REG_CLIP) ---")
VER = {}
for tag, a, b, prim in CON:
    if tag not in ("C1", "C2", "C3", "C6"):
        continue
    key = f"{tag}: {a} - {b}"
    sf, sc, su = vrule(key, "REG_FRAME"), vrule(key, "REG_CLIP"), vrule(key, "UNREG")
    ratio = ST[key]["REG_FRAME"]["mean"] / ST[key]["UNREG"]["mean"] if ST[key]["UNREG"]["mean"] != 0 else float("nan")
    VER[tag] = dict(contrast=key, vrule_UNREG=su, survives_REG_FRAME=sf, robust_REG_CLIP=sf and sc,
                    ratio_frame_over_unreg=ratio, magnitude_preserved=0.5 <= ratio <= 2.0,
                    counts={v: ST[key][v]["improved"] for v in VARS},
                    direction_preserved=ST[key]["REG_FRAME"]["mean"] < 0 and ST[key]["REG_CLIP"]["mean"] < 0)
    P(f"  {tag} {NAME[a]} vs {NAME[b]}: V-rule UNREG {'meets' if su else 'does not meet'}; REG_FRAME "
      f"{'SURVIVES' if sf else 'DOES NOT SURVIVE'}; REG_CLIP {'holds -> ROBUST' if sf and sc else ('holds' if sc else 'does not hold')}; "
      f"magnitude REG_FRAME/UNREG {ratio:.2f} ({'preserved' if 0.5 <= ratio <= 2.0 else 'NOT preserved'}); "
      f"DIRECTION {'PRESERVED' if VER[tag]['direction_preserved'] else 'NOT preserved'} (REG_FRAME and REG_CLIP means < 0); "
      f"negative clips " + ", ".join(f"{v} {ST[key][v]['improved']}/12" for v in VARS))
key3 = f"C3: {AYS8} - {ORIGIN}"
for v in ("UNREG", "REG_FRAME", "REG_CLIP"):
    s = ST[key3][v]
    fg = s["mean"] <= -0.002 and max(s["per_clip"]) <= 0.005
    VER[f"C3_freegain_{v}"] = fg
    P(f"  C3 judge free-gain rule under {v:9s}: mean {s['mean']:+.5f}, worst clip {s['worst'][0]} {s['worst'][1]:+.4f} -> "
      f"{'PASS (free gain)' if fg else 'FAIL'}")

SENS = {}
keep = [c for c in CLIPS if c != ILL_CLIP]
P(f"\n  SENSITIVITY: without the ILL-POSED clip {ILL_CLIP} (n={len(keep)}); the V-rule count threshold scales to "
  f"ceil(11*10/12) = {math.ceil(len(keep) * 10 / 12)}")
for tag, a, b, prim in CON:
    if tag not in ("C1", "C2", "C3", "C6"):
        continue
    for v in ("REG_FRAME", "REG_CLIP", "UNREG"):
        s_ = contrast_stats(a, b, v, keep)
        SENS[f"{tag} {v}"] = s_
        P(f"    {tag} {v:10s} mean {s_['mean']:+.4f}  negative {s_['improved']}/{len(keep)}  pct95 {f2(s_['ci_pct'])}  "
          f"t95 {f2(s_['ci_t'])}  p_sign {s_['p_sign']:.5f}")
P(f"\n  ORIGINAL-RANGE (score_reg_v1, un-widened grid) 0125 deltas vs widened (primary):")
for tag, a, b, prim in CON:
    for v in ("REG_FRAME", "REG_CLIP"):
        o_ = clipmean(ILL_CLIP, a, v, D_ORIG) - clipmean(ILL_CLIP, b, v, D_ORIG)
        w_ = clipmean(ILL_CLIP, a, v) - clipmean(ILL_CLIP, b, v)
        P(f"    {tag} {v:10s} original-range {o_:+.4f}  widened {w_:+.4f}")

# =========================================================================================== right-eye PSNR (distortion side)
P("\n--- right-eye PSNR (dB, from the mean MSE over frames, averaged over clips) ---")
P(f"  {'row':40s}" + "".join(f"{v:>14s}" for v in VARS[:4]))
rps = {}
for r in ROWS_SHOWN:
    rps[r] = {v: float(np.mean([D[c]["configs"][r]["rPSNR"][v] for c in CLIPS])) for v in VARS[:4]}
    P(f"  {NAME[r]:40s}" + "".join(f"{rps[r][v]:14.3f}" for v in VARS[:4]))

json.dump(dict(gates=gates, absolute=absm, contrasts=ST, verdicts=VER, sensitivity_n11=SENS, rPSNR=rps,
               sources=dict(pass1=f"{O}/score_reg_v1", pass1_wide=f"{O}/score_reg_v1_wide", pass2=A5DIR,
                            pass2_wide=A5WIDE, er=ER)),
          open(OUT_J, "w"), indent=1, default=str)
open(OUT_T, "w").write("\n".join(L) + "\n")
P(f"wrote {OUT_T} and {OUT_J}")
