#!/usr/bin/env python
"""Builds TABLES_v2.txt (skeptic lane; v2 = v1 + review fixes: O_C reported as a LEVEL next to the non-oracle BRt,
R6 magnitude/count phrasing, R7 with clip counts, the bar vs scale_gt's WORKS-AT-SCALE clauses, G2 edge note) from the per-clip JSONs.  Rules R1-R7 exactly as PREREG.txt (+ addenda).
usage: python analyze_v1.py <out_txt>"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
REC = "scripts/distill/runs/deep_20261004/skeptic"
OUTD = "outputs/deep_20261004/skeptic"
TEST = ["0042", "0052", "0125", "0128", "0141", "0147", "0170", "0204", "0225", "0251", "0259", "0301"]
AVP, IPH = TEST[:6], TEST[6:]
NEARMONO = ["0204", "0225", "0251", "0259", "0301"]
out_path = sys.argv[1]
assert not os.path.exists(out_path), f"refusing to overwrite {out_path}"
L = []


def P(*a):
    L.append(" ".join(str(x) for x in a))


def load_dir(d):
    r = {}
    for p in sorted(glob.glob(f"{OUTD}/{d}/*.json")):
        c = os.path.basename(p)[:-5]
        if re.fullmatch(r"\d{4}", c):
            r[c] = json.load(open(p))
    return r


def tci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    from scipy import stats
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return (x.mean() - h, x.mean() + h)


def sign_p(neg, n):
    from scipy import stats
    return float(stats.binomtest(neg, n, 0.5).pvalue) if n else float("nan")


S38 = load_dir("s38_v1")
M3 = load_dir("m3_v2")
M6 = load_dir("m6_v2")
TS = json.load(open(f"{OUTD}/trainsample_v1.json")) if os.path.exists(f"{OUTD}/trainsample_v1.json") else None
M5S = json.load(open(f"{OUTD}/m5supp_v1.json")) if os.path.exists(f"{OUTD}/m5supp_v1.json") else None
G0 = json.load(open(f"{REC}/G0_test12.json"))

P("=" * 118)
P("SKEPTIC LANE (deep_20261004) -- TABLES_v2 (supersedes TABLES_v1 wording; numbers identical).  CPU only.  Pre-registration: PREREG.txt + ADDENDUM_1..5 (3-5 exploratory).")
P("FRAME SETS: S38 = the 38 published scored frames (origin REG_FRAME 0.2432); S10 = every 4th of them (oracle set; origin 0.2455);")
P("S6 = 6 frames for the slow held-out metrics.  Deltas are only ever taken within one frame set.")
P(f"inputs: {OUTD}/s38_v1 ({len(S38)} clips), m3_v2 ({len(M3)}), m6_v2 ({len(M6)}), trainsample_v1.json "
  f"({'yes' if TS else 'MISSING'}), m5supp_v1.json ({'yes' if M5S else 'MISSING'})")
P("=" * 118)

# ------------------------------------------------------------------ gates
P("\n--- GATES ---")
P(f"G0 validity (12 test clips): {sum(v['valid'] for v in G0.values())}/12 valid; min MAD(TL,TR) "
  f"{min(v['mad_TL_TR'] for v in G0.values()):.2f} [{REC}/G0_test12.json]")
ks = []
for f in (f"{REC}/check_temporal_align_v1_smoke.log", f"{REC}/G1_temporal_align_rest.log"):
    for line in open(f):
        m = re.search(r"(splat TL vs left_v2|train TR vs right_v2).*best k=([+-]\d)", line)
        if m:
            ks.append(int(m.group(2)))
P(f"G1 temporal alignment: {len(ks)} gate comparisons (splat TL vs left_v2, train TR vs right_v2), best k = 0 on "
  f"{sum(k == 0 for k in ks)}/{len(ks)}")
devs = [(c, r, abs(e[k])) for c, d in S38.items() for r, e in d["rows"].items() for k in ("dev_UNREG", "dev_REG_FRAME") if k in e]
mx = max(devs, key=lambda t: t[2]) if devs else None
P(f"G2 CPU LPIPS reproduces eval_robustness / ays registered cells: {len(devs)} cells, max |dev| "
  f"{mx[2]:.2e} ({mx[0]} {mx[1]}); rule 2e-4 -> {'PASS (at the edge of the rule; CPU fp32 vs GPU numerics)' if mx and mx[2] <= 2e-4 else 'FAIL'}")
if TS:
    for fam, pf in TS["per_family"].items():
        bad = [c for c in pf["picked"] if not TS["clips"][c].get("valid")]
        P(f"G0 train sample {fam}: {len(pf['valid'])}/{len(pf['picked'])} valid; excluded by MAD<=5: {bad}")

# ------------------------------------------------------------------ the bar
P("\n--- THE ROUND'S BAR vs EXISTING ROWS (REG_FRAME, 12 clips, S38; my CPU re-score of the published cells) ---")
P("bar = 'registered LPIPS >= 0.005 better than origin on >= 9/12 clips'")
for r in ["AYS8", "deliverable", "s25"]:
    shr = np.mean([S38[c]["rows"][r]["sharp"] for c in TEST]) / np.mean([S38[c]["rows"]["origin"]["sharp"] for c in TEST])
    dps = np.mean([S38[c]["rows"][r]["psnr_REG_FRAME"] - S38[c]["rows"]["origin"]["psnr_REG_FRAME"] for c in TEST])
    P(f"  {r:12s} scale_gt WORKS-AT-SCALE clauses: 12-clip sharp ratio vs origin x{shr:.3f} (needs >= 0.90), "
      f"registered PSNR change {dps:+.3f} dB (needs >= -0.5)")
    d = [S38[c]["rows"][r]["lpips_REG_FRAME"] - S38[c]["rows"]["origin"]["lpips_REG_FRAME"] for c in TEST]
    P(f"  {r:12s} mean d {np.mean(d):+.4f}  improved {sum(x < 0 for x in d)}/12  t95 [{tci(d)[0]:+.4f},{tci(d)[1]:+.4f}]"
      f"  -> {'PASSES the bar' if np.mean(d) <= -0.005 and sum(x < 0 for x in d) >= 9 else 'does not pass'}"
      f"{'  (TRAINING-FREE)' if r in ('AYS8', 's25') else ''}")

# ------------------------------------------------------------------ Q1 camera differences
P("\n--- Q1 CAMERA DIFFERENCES: left eye vs REAL right eye on the deployed 576x1024 window ---")
P("rho_b = sqrt(P_right/P_left) of Hann-windowed luma power, bands b1 [1/32,1/16) b2 [1/16,1/8) b3 [1/8,1/4) b4 [1/4,1/2]")
P("cyc/px.  T = train tile (what the metric sees), V = crf-12 v2 sources (the cameras).  <1 = right eye has LESS detail.")
P(f"{'clip':5s} {'fam':6s} {'rho_T b1..b4':>27s} {'rho_V b1..b4':>27s} {'sh R/L T':>8s} {'sh R/L V':>8s} {'dY8':>6s} "
  f"{'dL*':>5s} {'da*':>5s} {'db*':>5s} {'nzL':>5s} {'nzR':>5s} {'sig_rel':>7s}")
for c in TEST:
    if c not in M3:
        P(f"{c} PENDING"); continue
    m = M3[c]["m1"]
    P(f"{c:5s} {M3[c]['family']:6s} {' '.join(f'{x:6.3f}' for x in m['rho_T'])} {' '.join(f'{x:6.3f}' for x in m['rho_V'])} "
      f"{m['sharp']['T_R'] / m['sharp']['T_L']:8.3f} {m['sharp']['V_R'] / m['sharp']['V_L']:8.3f} "
      f"{m['exposure']['dY_8bit']:+6.2f} {m['lab_offset_GT_minus_left'][0]:+5.2f} {m['lab_offset_GT_minus_left'][1]:+5.2f} "
      f"{m['lab_offset_GT_minus_left'][2]:+5.2f} {m['noise']['V_L']:5.2f} {m['noise']['V_R']:5.2f} "
      f"{m['sigma']['sigma_rel_corrected']:+7.3f}")
fam_rho = {}
for fam, cl in (("AVP", AVP), ("iPhone", IPH)):
    tT = [M3[c]["m1"]["rho_T"] for c in cl if c in M3]
    tV = [M3[c]["m1"]["rho_V"] for c in cl if c in M3]
    sT = [v["rho_T"] for v in TS["clips"].values() if v.get("valid") and v["family"] == fam] if TS else []
    sV = [v["rho_V"] for v in TS["clips"].values() if v.get("valid") and v["family"] == fam] if TS else []
    pT, pV = np.array(tT + sT), np.array(tV + sV)
    fam_rho[fam] = (pT, pV)
    if len(pT):
        P(f"  {fam:6s} pooled test+train n={len(pT)}: median rho_T {np.round(np.median(pT, 0), 3).tolist()}  "
          f"rho_V {np.round(np.median(pV, 0), 3).tolist()}  b3 IQR T [{np.percentile(pT[:, 2], 25):.3f},"
          f"{np.percentile(pT[:, 2], 75):.3f}]  share b3<0.87: {np.mean(pT[:, 2] < 0.87):.2f}  share b3>1.15: "
          f"{np.mean(pT[:, 2] > 1.15):.2f}")
        mT, mV = np.median(pT[:, 2]), np.median(pV[:, 2])
        if mT < 0.87 and mV < 0.87:
            v = "GT SOFTER than input"
        elif mT > 1.15 and mV > 1.15:
            v = "GT SHARPER than input"
        elif (mT < 0.87) != (mV < 0.87) or (mT > 1.15) != (mV > 1.15):
            v = "asymmetry only in one data version (codec, not camera)"
        else:
            v = "no material asymmetry"
        P(f"  R1 {fam}: median b3 T {mT:.3f} V {mV:.3f} -> {v}")
    sg = [M3[c]["m1"]["sigma"]["sigma_rel_corrected"] for c in cl if c in M3]
    if sg:
        P(f"  R2 {fam}: median bias-corrected sigma_rel {np.median(sg):+.3f} px over {len(sg)} test clips -> "
          f"{'MATERIAL' if abs(np.median(sg)) >= 0.5 else 'not material'}  (raw {[M3[c]['m1']['sigma']['sigma_rel'] for c in cl if c in M3]}, "
          f"ctrl {[M3[c]['m1']['sigma']['ctrl_sigma_rel'] for c in cl if c in M3]})")
if TS:
    for fam in ("AVP", "iPhone"):
        cl = [v for v in TS["clips"].values() if v.get("valid") and v["family"] == fam]
        P(f"  train sample {fam} n={len(cl)}: exposure dY median {np.median([v['exposure_dY_8bit'] for v in cl]):+.2f} "
          f"(IQR {np.percentile([v['exposure_dY_8bit'] for v in cl], 25):+.2f},{np.percentile([v['exposure_dY_8bit'] for v in cl], 75):+.2f}) "
          f"| Lab offset median {np.round(np.median([v['lab_offset_R_minus_L'] for v in cl], 0), 2).tolist()} | noise V L/R median "
          f"{np.median([v['noise']['V_L'] for v in cl]):.2f}/{np.median([v['noise']['V_R'] for v in cl]):.2f} | copy-left LPIPS "
          f"(global reg.) median {np.median([np.mean(v['copyleft_lpips_global']) for v in cl]):.4f}")

P("\nDETAIL KEPT FROM THE INPUT (amplitude ratio output/input-left, same window, S10; b1..b4) -- the blur the user sees:")
for k in ("rho_origin_vs_input", "rho_deliv_vs_input", "rho_s25_vs_input", "rho_GT_vs_input"):
    a = np.array([M3[c]["m1"][k] for c in TEST if c in M3])
    if len(a):
        P(f"  {k:22s} median {np.round(np.median(a, 0), 3).tolist()}  min b3 {a[:, 2].min():.3f} max b3 {a[:, 2].max():.3f}")
P("Mean-gradient sharpness (score_clip_ll 'sharp', S38): left input | real right GT | origin | s25")
rat = []
for c in TEST:
    d = S38[c]
    rat.append((d["rows"]["origin"]["sharp"] / d["leftSharp"], d["gtSharp_UNREG"] / d["leftSharp"], d["rows"]["s25"]["sharp"] / d["leftSharp"]))
    P(f"  {c} {d['leftSharp']:.4f} | {d['gtSharp_UNREG']:.4f} | {d['rows']['origin']['sharp']:.4f} | {d['rows']['s25']['sharp']:.4f}"
      f"   origin/left {rat[-1][0]:.2f}  GT/left {rat[-1][1]:.2f}  s25/left {rat[-1][2]:.2f}")
rat = np.array(rat)
P(f"  medians: origin/left {np.median(rat[:, 0]):.2f}  GT/left {np.median(rat[:, 1]):.2f}  s25/left {np.median(rat[:, 2]):.2f}")

# ------------------------------------------------------------------ copy-left
P("\n--- M2 COPY-LEFT (a NO-STEREO output: the input left eye shipped as the right eye), S38 ---")
P(f"{'clip':5s} {'origin U':>9s} {'origin R':>9s} {'CL U':>7s} {'CL R':>7s} {'CL OPT':>7s} {'OPTpsnr':>7s} {'s25 R':>7s} {'deliv R':>7s}")
cols = {k: [] for k in ("oU", "oR", "cU", "cR", "cO", "sR", "dR", "aR")}
for c in TEST:
    d = S38[c]; cl = d["copyleft"]; r = d["rows"]
    vals = dict(oU=r["origin"]["lpips_UNREG"], oR=r["origin"]["lpips_REG_FRAME"], cU=cl["lpips_UNREG"], cR=cl["lpips_REG_FRAME"],
                cO=cl["lpips_OPT"], sR=r["s25"]["lpips_REG_FRAME"], dR=r["deliverable"]["lpips_REG_FRAME"], aR=r["AYS8"]["lpips_REG_FRAME"])
    for k, v in vals.items():
        cols[k].append(v)
    P(f"{c:5s} {vals['oU']:9.4f} {vals['oR']:9.4f} {vals['cU']:7.4f} {vals['cR']:7.4f} {vals['cO']:7.4f} {cl['psnr_OPT']:7.2f} "
      f"{vals['sR']:7.4f} {vals['dR']:7.4f}{'  near-mono' if c in NEARMONO else ''}")
cols = {k: np.array(v) for k, v in cols.items()}
P(f"mean  {cols['oU'].mean():9.4f} {cols['oR'].mean():9.4f} {cols['cU'].mean():7.4f} {cols['cR'].mean():7.4f} {cols['cO'].mean():7.4f}"
  f"         {cols['sR'].mean():7.4f} {cols['dR'].mean():7.4f}")
P(f"  copy-left beats origin: UNREG {int((cols['cU'] < cols['oU']).sum())}/12 (mean d {np.mean(cols['cU'] - cols['oU']):+.4f}); "
  f"REG_FRAME {int((cols['cR'] < cols['oR']).sum())}/12 (mean d {np.mean(cols['cR'] - cols['oR']):+.4f}); "
  f"OPT-vs-origin REG {int((cols['cO'] < cols['oR']).sum())}/12 (mean d {np.mean(cols['cO'] - cols['oR']):+.4f})")
P(f"  copy-left OPT beats s25 REG on {int((cols['cO'] < cols['sR']).sum())}/12; copy-left UNREG beats s25 UNREG on "
  f"{sum(S38[c]['copyleft']['lpips_UNREG'] < S38[c]['rows']['s25']['lpips_UNREG'] for c in TEST)}/12")
nm = [c for c in NEARMONO]
wins = sum(S38[c]["copyleft"]["lpips_OPT"] < S38[c]["rows"]["origin"]["lpips_REG_FRAME"] for c in nm)
P(f"  R3 near-mono clips {nm}: CL-OPT < origin REG_FRAME on {wins}/5 -> "
  f"{'TRUE: copying the left eye beats origin on near-mono clips' if wins >= 4 else 'not met'}")

# ------------------------------------------------------------------ oracles
P("\n--- M3 ORACLES, S10 frames, target = REG_FRAME GT.  These are LEVELS, not a ceiling (see the BRt line below) ---")
RW = ["origin", "AYS8", "deliverable", "s25", "BR_raw", "O_C", "O_Ccc", "O_Ccc_naive", "O_A", "O_Acc", "O_B", "copyleft_win", "K1", "K2"]
P(f"{'clip':5s} " + " ".join(f"{r[:11]:>11s}" for r in RW))
tab = {r: [] for r in RW}
have = [c for c in TEST if c in M3]
for c in have:
    rr = M3[c]["rows"]
    for r in RW:
        tab[r].append(rr[r]["lpips"])
    P(f"{c:5s} " + " ".join(f"{rr[r]['lpips']:11.4f}" for r in RW))
if have:
    for nmn, sub in (("mean12", have), ("AVP", [c for c in have if c in AVP]), ("iPhone", [c for c in have if c in IPH]),
                     ("n11 -0125", [c for c in have if c != "0125"])):
        P(f"{nmn:9s}" + " ".join(f"{np.mean([M3[c]['rows'][r]['lpips'] for c in sub]):11.4f}" for r in RW))
    o = np.array(tab["origin"])
    for r, nmr in (("O_C", "H_C"), ("O_Ccc", "H_Ccc"), ("BR_raw", "H_BRraw")):
        h = o - np.array(tab[r])
        lvl = "no headroom beyond what s25/AYS take" if h.mean() <= 0.015 else ("modest" if h.mean() <= 0.05 else "substantial in principle")
        P(f"  R4 {nmr} = origin - {r}: mean {h.mean():+.4f} t95 [{tci(h)[0]:+.4f},{tci(h)[1]:+.4f}] positive on "
          f"{int((h > 0).sum())}/{len(h)}  -> {lvl}")
    geo = [c for c in have if M3[c]["rows"]["O_B"]["lpips"] >= 0.5 * M3[c]["rows"]["origin"]["lpips"]]
    BKx = json.load(open(f"{OUTD}/bookkeep_v1.json")) if os.path.exists(f"{OUTD}/bookkeep_v1.json") else None
    if BKx:
        for nmn, sub in (("all12", have), ("AVP", [c for c in have if c in AVP]), ("iPhone", [c for c in have if c in IPH])):
            P(f"  NON-ORACLE input-fidelity level (S10): BRt = model input + Telea hole fill, {nmn}: "
              f"{np.mean([BKx[c]['BRt'] for c in sub]):.4f}  vs O_C {np.mean([M3[c]['rows']['O_C']['lpips'] for c in sub]):.4f}  "
              f"vs origin {np.mean([M3[c]['rows']['origin']['lpips'] for c in sub]):.4f}  vs s25 {np.mean([M3[c]['rows']['s25']['lpips'] for c in sub]):.4f}  "
              f"vs O_B (floor at this geometry) {np.mean([M3[c]['rows']['O_B']['lpips'] for c in sub]):.4f}")
        P("  -> O_C is beaten by the non-oracle BRt (the oracle holes paste GT content into the input geometry and create seams);")
        P("     read O_C/BRt as the INPUT-FIDELITY LEVEL (~0.19), not an upper bound; a model that also removed the splat blur and")
        P("     cracks could go lower; O_B is the floor for any output that keeps the input geometry.")
    P(f"  R5 geometry-dominated clips (O_B >= 0.5 x origin): {len(geo)}/{len(have)} {geo}")
    P(f"  O_B / origin ratio per clip: " + " ".join(f"{c}:{M3[c]['rows']['O_B']['lpips'] / M3[c]['rows']['origin']['lpips']:.2f}" for c in have))
    P("  share of the origin->O_C gap that s25 / deliverable / AYS8 already close: " + " ".join(
        f"{r}:{np.mean(o - np.array(tab[r])) / max(np.mean(o - np.array(tab['O_C'])), 1e-9):.2f}" for r in ("s25", "deliverable", "AYS8")))

    P("\nSpatial LPIPS map means (S10): non-hole | model hole | true-occluded | consistent non-hole ; mass share in holes / true-occl.")
    for r in ["origin", "s25", "BR_raw", "O_C", "O_A", "O_B"]:
        a = lambda k: np.mean([M3[c]["rows"][r][k] for c in have if M3[c]["rows"][r][k] is not None])
        P(f"  {r:10s} {a('sp_nonhole'):.4f} | {a('sp_hole'):.4f} | {a('sp_trueoccl'):.4f} | {a('sp_cons_nonhole'):.4f} ; "
          f"{a('mass_hole'):.3f} / {a('mass_trueoccl'):.3f}")
    P("\nResidual geometry after the per-frame global registration: |model disparity - true disparity| on textured, consistent,")
    P("non-hole pixels (RAFT, S10): median | p90 | share >2px | >4px | >8px     (+ true disparity median, p5-p95 vs TRg)")
    for c in have:
        pf = M3[c]["per_frame"]
        g = lambda k: np.mean([x[k] for x in pf if x[k] is not None])
        P(f"  {c} {g('resid_abs_med'):5.2f} | {g('resid_abs_p90'):5.2f} | {g('resid_gt2'):.2f} | {g('resid_gt4'):.2f} | {g('resid_gt8'):.2f}"
          f"     true occl {g('true_occl'):.3f}  hole {g('hole'):.3f}  U3 {M3[c].get('U3', float('nan')):.3f}")
    P("Controls: K1 = GT bicubic-shifted by (0.5,0.5) px; K2 = GT round-tripped through the two RAFT warps.  "
      f"mean K1 {np.mean(tab['K1']):.4f}, K2 {np.mean(tab['K2']):.4f}")
    P("\nInput fidelity (EXPLORATORY): LPIPS(row, model input BR), spatial mean over non-hole pixels")
    for r in ["origin", "AYS8", "deliverable", "s25"]:
        P(f"  {r:12s} " + " ".join(f"{c}:{M3[c]['fidelity_to_input'][r]['sp_nonhole']:.3f}" for c in have if 'fidelity_to_input' in M3[c])
          + f"   mean {np.mean([M3[c]['fidelity_to_input'][r]['sp_nonhole'] for c in have if 'fidelity_to_input' in M3[c]]):.4f}")

# ------------------------------------------------------------------ unobservable share
P("\n--- M4 UNOBSERVABLE SHARE ---")
u1 = {c: float(np.mean(S38[c]["hole_frac"])) for c in TEST}
P("U1 model-hole fraction (S38): " + " ".join(f"{c}:{u1[c]:.3f}" for c in TEST) + f"  mean {np.mean(list(u1.values())):.3f}")
if have:
    u2 = {c: float(np.mean([x["true_occl"] for x in M3[c]["per_frame"]])) for c in have}
    P("U2 true-occlusion + out-of-view + flow-failure fraction of GT (S10): " + " ".join(f"{c}:{u2[c]:.3f}" for c in have)
      + f"  mean {np.mean(list(u2.values())):.3f}")
    P("U3 consistent pixels with |dY| > 24/255 after colour fit: " + " ".join(f"{c}:{M3[c].get('U3', float('nan')):.3f}" for c in have))
    P("U4 origin's spatial-LPIPS mass in model holes / in true-occluded pixels: " + " ".join(
        f"{c}:{M3[c]['rows']['origin']['mass_hole']:.3f}/{M3[c]['rows']['origin']['mass_trueoccl']:.3f}" for c in have))

# ------------------------------------------------------------------ seed spread
P("\n--- Q3 SEED SPREAD: LPIPS between two equally good outputs (deliverable T5@1.00 natural vs RNG-padded, windows>=1) ---")
sd = []
for c in TEST:
    s = S38[c]["seed"]
    sd.append(s["lpips_T5nat_T5pad"])
    P(f"  {c} LPIPS(out,out') {s['lpips_T5nat_T5pad']:.4f}  PSNR {s['psnr']:5.2f}  |dLPIPS to GT| {s['dGT_REG']:.4f}  "
      f"mass in holes {s['mass_in_holes']:.3f} (hole area {s['hole_area']:.3f})  vs LPIPS to GT {S38[c]['rows']['T5nat']['lpips_REG_FRAME']:.4f}")
P(f"  mean LPIPS(out,out') {np.mean(sd):.4f}; ratio to the outputs' own REG LPIPS "
  f"{np.mean(sd) / np.mean([S38[c]['rows']['T5nat']['lpips_REG_FRAME'] for c in TEST]):.2f}")
if M5S:
    P(f"  supplement (LOSSY mp4v, origin whole-clip seeds, 0042): pairs {json.dumps(M5S['pair'])}; codec control "
      f"LPIPS(mp4v(origin_ll), origin_ll) {M5S['codec_ctrl_lpips']:.4f} (PSNR {M5S['codec_ctrl_psnr']:.2f}); to GT {json.dumps(M5S['toGT_REG'])}; "
      f"left-half check PSNR {M5S['left_half_check_psnr_seed1_vs_ll']:.2f}")

# ------------------------------------------------------------------ hackability
P("\n--- Q4 LPIPS HACKABILITY: GT-independent post-processing of ORIGIN (M6), 12-clip means of the change vs P0 ---")
if M6:
    hv = [c for c in TEST if c in M6]
    names = list(M6[hv[0]]["rows"].keys())
    P(f"clips: {len(hv)}  P12 choice AVP {TS['per_family']['AVP']['P12_choice_global'] if TS else '?'} / iPhone "
      f"{TS['per_family']['iPhone']['P12_choice_global'] if TS else '?'} (flow-registered choice: "
      f"{TS['per_family']['AVP']['P12_choice_flow'] if TS else '?'} / {TS['per_family']['iPhone']['P12_choice_flow'] if TS else '?'})")
    P(f"{'row':12s} {'dAlexREG':>9s} {'impr':>5s} {'dAlexUN':>8s} {'dPSNR':>7s} {'dVGG':>8s} {'dDISTS':>8s} {'dBRISQ':>7s} "
      f"{'flatHF/GT':>9s} {'stripe/GT':>9s} {'sharp/GT':>8s}")
    base = {c: M6[c]["rows"]["P0"] for c in hv}
    summary = {}
    for nmr in names:
        g = lambda k: np.array([M6[c]["rows"][nmr][k] - base[c][k] for c in hv])
        dA = g("alex_REG_FRAME")
        fr = np.mean([M6[c]["rows"][nmr]["decompose"]["flatHF"] / M6[c]["gt_decompose"]["flatHF"] for c in hv])
        st = np.mean([M6[c]["rows"][nmr]["decompose"]["stripeE"] / M6[c]["gt_decompose"]["stripeE"] for c in hv])
        sh = np.mean([M6[c]["rows"][nmr]["sharp"] / M6[c]["gtSharp_REG"] for c in hv])
        summary[nmr] = dict(dA=dA, dV=g("vgg_REG"), dD=g("dists_REG"), dP=g("psnr_REG"), dB=g("brisque"))
        P(f"{nmr:12s} {dA.mean():+9.4f} {int((dA < 0).sum()):3d}/{len(hv)} {g('alex_UNREG').mean():+8.4f} {g('psnr_REG').mean():+7.3f} "
          f"{g('vgg_REG').mean():+8.4f} {g('dists_REG').mean():+8.4f} {g('brisque').mean():+7.2f} {fr:9.3f} {st:9.3f} {sh:8.3f}")
    # cross-fit
    grid = [f"P{i}" for i in range(1, 11)]
    cf = {}
    for fam, sel_cl, ev_cl in (("AVP->iPhone", [c for c in hv if c in AVP], [c for c in hv if c in IPH]),
                               ("iPhone->AVP", [c for c in hv if c in IPH], [c for c in hv if c in AVP])):
        if not sel_cl or not ev_cl:
            continue
        best = min(grid, key=lambda n: np.mean([M6[c]["rows"][n]["alex_REG_FRAME"] - base[c]["alex_REG_FRAME"] for c in sel_cl]))
        d = [M6[c]["rows"][best]["alex_REG_FRAME"] - base[c]["alex_REG_FRAME"] for c in ev_cl]
        cf[fam] = (best, d)
        P(f"  cross-fit {fam}: chose {best} on the selection family; held-out family mean d {np.mean(d):+.4f}, improved {sum(x < 0 for x in d)}/{len(d)}")
    allcf = sum((v[1] for v in cf.values()), [])
    gm = []
    if allcf:
        P(f"  cross-fitted P over all 12 clips: mean d {np.mean(allcf):+.4f}, improved {sum(x < 0 for x in allcf)}/{len(allcf)}")
        gm.append(("cross-fit", np.mean(allcf), sum(x < 0 for x in allcf)))
    for nmr in ("P11", "P12", "P12f", "P13"):
        dA = summary[nmr]["dA"]
        gm.append((nmr, dA.mean(), int((dA < 0).sum())))
    game = [g for g in gm if g[1] <= -0.005 and g[2] >= 9]
    mag = [g for g in gm if g[1] <= -0.005]
    P(f"  R6 bar GAMEABLE by a GT-independent / train-fitted post-filter: {'YES ' + str(game) if game else 'NO by the pre-registered rule'}"
      f"{' -- but MAGNITUDE reached by ' + str([(g[0], round(g[1], 4), str(g[2]) + '/12') for g in mag]) + ' (count fails: < 9/12)' if mag and not game else ''}"
      f"  (all candidates {[(g[0], round(g[1], 4), g[2]) for g in gm]})")
    best_any = min(grid, key=lambda n: summary[n]["dA"].mean())
    P(f"  descriptive: best single grid filter on all 12 (test-selected, NOT a headline): {best_any} mean d "
      f"{summary[best_any]['dA'].mean():+.4f}, improved {int((summary[best_any]['dA'] < 0).sum())}/12")
    for nmr in names:
        dA, dV, dD = summary[nmr]["dA"].mean(), summary[nmr]["dV"].mean(), summary[nmr]["dD"].mean()
        if dA <= -0.005:
            tag = []
            tag.append("VGG catches" if dV >= 0 else "VGG agrees")
            tag.append("DISTS catches" if dD >= 0 else "DISTS agrees")
            P(f"  R7 {nmr:12s} alex {dA:+.4f}: {', '.join(tag)} (dVGG {dV:+.4f} [{int((summary[nmr]['dV'] < 0).sum())}/12 improved], "
              f"dDISTS {dD:+.4f} [{int((summary[nmr]['dD'] < 0).sum())}/12], dPSNR {summary[nmr]['dP'].mean():+.3f})")
    for nmr in ("AYS8", "deliverable", "s25"):
        P(f"  R7 reference {nmr:12s} dVGG {summary[nmr]['dV'].mean():+.4f} ({int((summary[nmr]['dV'] < 0).sum())}/12)  dDISTS "
          f"{summary[nmr]['dD'].mean():+.4f} ({int((summary[nmr]['dD'] < 0).sum())}/12) -> "
          f"{'confirmed by both held-out metrics' if summary[nmr]['dV'].mean() < 0 and summary[nmr]['dD'].mean() < 0 else 'not confirmed by both'}")
    P(f"  R7 reading: LPIPS-VGG neither catches the sharpening filter P4 (dVGG {summary['P4']['dV'].mean():+.4f}) nor confirms the "
      f"deliverable (dVGG {summary['deliverable']['dV'].mean():+.4f}, {int((summary['deliverable']['dV'] < 0).sum())}/12); it catches grain only "
      f"(P9 {summary['P9']['dV'].mean():+.4f}, P10 {summary['P10']['dV'].mean():+.4f}).  DISTS separates by MAGNITUDE: deliverable "
      f"{summary['deliverable']['dD'].mean():+.4f} ({int((summary['deliverable']['dD'] < 0).sum())}/12) vs P4 {summary['P4']['dD'].mean():+.4f} "
      f"({int((summary['P4']['dD'] < 0).sum())}/12).  BRISQUE (NR) improves with every unsharp step even when LPIPS worsens "
      f"(P1/P2/P3: BRISQUE {summary['P1']['dB'].mean():+.2f}/{summary['P2']['dB'].mean():+.2f}/{summary['P3']['dB'].mean():+.2f}, "
      f"alex {summary['P1']['dA'].mean():+.4f}/{summary['P2']['dA'].mean():+.4f}/{summary['P3']['dA'].mean():+.4f}) -> an NR metric is a hack SIGNATURE, not a guard.")
    P("\nM7 stereo retention (DIS, S6): slope of d_output on d_input (1 = geometry kept, 0 = collapsed to the left eye)")
    for nmr in ("origin", "AYS8", "deliverable", "s25", "P4", "P8"):
        sl = [M6[c]["m7"][nmr]["slope"] for c in hv]
        ma = [M6[c]["m7"][nmr]["mean_abs_diff"] for c in hv]
        P(f"  {nmr:12s} slope median {np.median(sl):.3f} (min {np.min(sl):.3f})  mean|d_O-d_BR| median {np.median(ma):.3f}px")
else:
    P("PENDING")

# ------------------------------------------------------------------ exploratory: composites (ADDENDUM 3)
COMP = load_dir("comp_v1")
P("\n--- EXPLORATORY (ADDENDUM 3): does LPIPS reward outputs the user rejected?  (S38 alex/PSNR; S6 vgg/DISTS/decompose) ---")
if COMP:
    hc = [c for c in TEST if c in COMP]
    P(f"clips {len(hc)}.  COMP_origin = model input BR + origin inside the holes (user-REJECTED mask-only compositing); "
      f"COMP_telea = BR + Telea fill (NO generative model)")
    P(f"{'row':12s} {'alexU':>7s} {'alexR':>7s} {'dAlexR':>8s} {'impr':>5s} {'PSNR_R':>7s} {'VGG':>7s} {'DISTS':>7s} {'flatHF/GT':>9s} {'stripeE/GT':>10s}")
    for nmr in ("origin", "s25", "COMP_origin", "COMP_telea"):
        g = lambda k: np.array([COMP[c]["rows"][nmr][k] for c in hc])
        d = g("alex_REG_FRAME") - np.array([COMP[c]["rows"]["origin"]["alex_REG_FRAME"] for c in hc])
        fr = np.mean([COMP[c]["rows"][nmr]["decompose"]["flatHF"] / COMP[c]["gt_decompose"]["flatHF"] for c in hc])
        st = np.mean([COMP[c]["rows"][nmr]["decompose"]["stripeE"] / COMP[c]["gt_decompose"]["stripeE"] for c in hc])
        P(f"{nmr:12s} {g('alex_UNREG').mean():7.4f} {g('alex_REG_FRAME').mean():7.4f} {d.mean():+8.4f} {int((d < 0).sum()):3d}/{len(hc)} "
          f"{g('psnr_REG').mean():7.2f} {g('vgg_REG').mean():7.4f} {g('dists_REG').mean():7.4f} {fr:9.3f} {st:10.3f}")
    P("  per clip alexR origin -> COMP_telea: " + " ".join(
        f"{c}:{COMP[c]['rows']['origin']['alex_REG_FRAME']:.3f}->{COMP[c]['rows']['COMP_telea']['alex_REG_FRAME']:.3f}" for c in hc))
else:
    P("PENDING")
BK = json.load(open(f"{OUTD}/bookkeep_v1.json")) if os.path.exists(f"{OUTD}/bookkeep_v1.json") else None
P("\n--- EXPLORATORY (ADDENDUM 5): registration bookkeeping control, S10 (same frames as the oracles) ---")
if BK and M3:
    hb = [c for c in TEST if c in BK and c in M3]
    P(f"{'clip':5s} {'origin':>7s} {'BRt':>7s} {'X@orig':>7s} {'X0ctrl':>7s} {'bookk':>7s} {'Og':>7s} {'|F|orig':>7s} {'|F|ctrl':>7s} "
      f"{'H_C':>7s} {'H_C-bk':>7s} {'gapBRt':>7s} {'gapBRt-bk':>9s}")
    hc_raw, hc_cor, gb_raw, gb_cor, bks = [], [], [], [], []
    for c in hb:
        b = BK[c]
        hcr = M3[c]["rows"]["origin"]["lpips"] - M3[c]["rows"]["O_C"]["lpips"]
        gbr = b["origin"] - b["BRt"]
        hc_raw.append(hcr); hc_cor.append(hcr - b["bookkeeping"]); gb_raw.append(gbr); gb_cor.append(gbr - b["bookkeeping"]); bks.append(b["bookkeeping"])
        P(f"{c:5s} {b['origin']:7.4f} {b['BRt']:7.4f} {b['X_BR_at_origin_geom']:7.4f} {b['X0_ctrl']:7.4f} {b['bookkeeping']:+7.4f} "
          f"{b['Og_origin_at_BR_geom']:7.4f} {b['flow_stats']['origin']:7.2f} {b['flow_stats']['ctrl']:7.2f} {hcr:+7.4f} {hcr - b['bookkeeping']:+7.4f} "
          f"{gbr:+7.4f} {gbr - b['bookkeeping']:+9.4f}")
    P(f"mean  bookkeeping {np.mean(bks):+.4f}; H_C raw {np.mean(hc_raw):+.4f} -> corrected {np.mean(hc_cor):+.4f} t95 "
      f"[{tci(hc_cor)[0]:+.4f},{tci(hc_cor)[1]:+.4f}] positive {sum(x > 0 for x in hc_cor)}/{len(hc_cor)}; origin-BRt (no-model "
      f"composite, S10) raw {np.mean(gb_raw):+.4f} -> corrected {np.mean(gb_cor):+.4f} positive {sum(x > 0 for x in gb_cor)}/{len(gb_cor)}")
    for fam, cl in (("AVP", AVP), ("iPhone", IPH)):
        ii = [k for k, c in enumerate(hb) if c in cl]
        P(f"  {fam}: H_C raw {np.mean([hc_raw[k] for k in ii]):+.4f} corrected {np.mean([hc_cor[k] for k in ii]):+.4f}; "
          f"s25 gain (S10) {np.mean([M3[hb[k]]['rows']['origin']['lpips'] - M3[hb[k]]['rows']['s25']['lpips'] for k in ii]):+.4f}")
else:
    P("PENDING")
AN = json.load(open(f"{OUTD}/aniso_v1.json")) if os.path.exists(f"{OUTD}/aniso_v1.json") else None
P("\n--- EXPLORATORY (ADDENDUM 4): vertical-stripe anisotropy in GT-flat, non-hole regions, relative to the GT (S6) ---")
if AN:
    rr = ["origin", "s25", "deliverable", "COMP_origin", "COMP_telea", "BR_raw"]
    P(f"{'clip':5s} {'GT aniso':>8s} " + " ".join(f"{r:>11s}" for r in rr))
    for c in TEST:
        if c in AN:
            P(f"{c:5s} {AN[c]['GT']['aniso']:8.3f} " + " ".join(f"{AN[c][r]['rel']:11.3f}" for r in rr))
    P(f"{'median':5s} {'':8s} " + " ".join(f"{np.median([AN[c][r]['rel'] for c in AN]):11.3f}" for r in rr))
    P("  COMP_telea minus origin (rel aniso): " + " ".join(f"{c}:{AN[c]['COMP_telea']['rel'] - AN[c]['origin']['rel']:+.3f}" for c in TEST if c in AN))
else:
    P("PENDING")

open(out_path, "w").write("\n".join(L) + "\n")
print("\n".join(L))
