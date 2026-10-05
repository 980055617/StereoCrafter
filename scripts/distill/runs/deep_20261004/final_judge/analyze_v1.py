#!/usr/bin/env python
"""final_judge analysis (CPU).  Rules: PREREG.txt (this dir).  Reads only:
  outputs/deep_20261004/final_judge/score_v1/<clip>.json      (judge_score_v1.py, this lane)
  outputs/deep_20261004/final_judge/nr_v1/<clip>.json         (judge_nr_v1.py, this lane; optional)
  published per-clip values of every lane (gate J0), paths in J0_SOURCES below.
usage: python analyze_v1.py <score_dir> <nr_dir|-> <out_prefix (new)>"""
import json
import os
import re
import sys

import numpy as np

os.chdir("/home/kawa/master_project/StereoCrafter")
SD, ND, OP = sys.argv[1], sys.argv[2], sys.argv[3]
for ext in (".txt", ".json"):
    assert not os.path.exists(OP + ext), OP + ext
TEST = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
S = {c: json.load(open(f"{SD}/{c}.json")) for c in TEST if os.path.exists(f"{SD}/{c}.json")}
CL = [c for c in TEST if c in S]
NR = {}
if ND != "-":
    NR = {c: json.load(open(f"{ND}/{c}.json")) for c in CL if os.path.exists(f"{ND}/{c}.json")}
O = "origin_ll"
ROWS = ["mstudent2_step800_deliv_ll", "s25_ll", "AYS8_origin_g101", "origin_g100_sd7", "origin_g100_sd1fill",
        "deliv_g100_sd7", "deliv_g100_sd1fill", "m2svid_fa_w16_ll", "INPUT_fill", "LEFT_AS_RIGHT", "selMain250_s8",
        "selMain250_s25", "HIRES_B", "HIRES_B_L"]
NAME = {"mstudent2_step800_deliv_ll": "deliverable (shipped)", "s25_ll": "origin 25 steps", "AYS8_origin_g101":
        "origin + AYS 8-step schedule", "origin_g100_sd7": "sdedit origin sigma7.28", "origin_g100_sd1fill":
        "sdedit origin sigma1.17+fill", "deliv_g100_sd7": "sdedit deliv sigma7.28", "deliv_g100_sd1fill":
        "sdedit deliv sigma1.17+fill", "m2svid_fa_w16_ll": "M2SVid (external)", "INPUT_fill": "NO MODEL: warp+row fill",
        "LEFT_AS_RIGHT": "NO STEREO: left eye as right", "selMain250_s8": "scale_gt GT-ft (8 steps)",
        "selMain250_s25": "scale_gt GT-ft (25 steps)", "HIRES_B": "1.75x work-res, INTER_AREA",
        "HIRES_B_L": "1.75x work-res, Lanczos"}
# F8 separation audit (section D of PREREG; facts from the lanes' own records, checked in SEPARATION_AUDIT_v1.txt)
SEP = {"mstudent2_step800_deliv_ll": "clean (10 train clips; step chosen on dev 0040/0091/0184/0245)",
       "s25_ll": "clean (no selection)", "AYS8_origin_g101": "clean (published schedule; only the run/extend decision "
       "used 0301)", "origin_g100_sd7": "NOT CLEAN (variant chosen on test 0301/0204/0052/0147)",
       "origin_g100_sd1fill": "NOT CLEAN (variant chosen on test 0301/0204/0052/0147)",
       "deliv_g100_sd7": "NOT CLEAN (variant chosen on test 0301/0204/0052/0147)",
       "deliv_g100_sd1fill": "NOT CLEAN (variant chosen on test 0301/0204/0052/0147)",
       "m2svid_fa_w16_ll": "clean wrt our split (external weights; their training data unknown)",
       "INPUT_fill": "n/a (no model)", "LEFT_AS_RIGHT": "n/a (no model)",
       "selMain250_s8": "clean (215 train clips; step chosen on dev)", "selMain250_s25": "clean",
       "HIRES_B": "test clips 0042/0170 only", "HIRES_B_L": "NOT CLEAN (Lanczos chosen after seeing 0042/0170)"}
VARS = ["UNREG", "REG_FRAME", "REG_CLIP"]
rng = np.random.default_rng(20261005)
BOOT = rng.integers(0, 12, size=(10000, 12))


def lp(c, lab, var):
    return S[c]["configs"][lab]["lpips_clip"][var]


def has(c, lab):
    return lab in S[c]["configs"]


def boot_ci(d):
    d = np.asarray(d)
    if len(d) != 12:
        return (float("nan"), float("nan"))
    m = d[BOOT].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


out = dict(clips=CL, rows={}, gates={}, flags={})
L = []
P = L.append
P("=" * 118)
P("FINAL JUDGE (deep_20261004/final_judge) -- re-score of every gain row from the lossless renders on disk")
P(f"score dir {SD}   nr dir {ND}   clips {len(CL)}/12   rules PREREG.txt")
P("=" * 118)

# ---------------------------------------------------------------------------------------------------- J0
P("\n--- GATE J0: reproduction of the published per-clip values (max |my - published|) ---")
J0 = {}


def j0(name, pairs, tol):
    pairs = [(a, b) for a, b in pairs if b is not None and not (isinstance(b, float) and np.isnan(b))]
    if not pairs:
        P(f"  {name:58s} no published values found")
        return
    dv = max(abs(a - b) for a, b in pairs)
    J0[name] = dict(n=len(pairs), maxdev=dv, tol=tol, ok=dv <= tol)
    P(f"  {name:58s} n={len(pairs):3d}  max|d| {dv:.2e}  tol {tol:.0e}  {'PASS' if dv <= tol else 'FAIL'}")


ER = {}
for c in CL:
    p = f"outputs/more_20261004/eval_robustness/score_v1{'_wide' if c == '0125' else ''}/{c}.json"
    ER[c] = json.load(open(p))
for lab in ["origin_ll", "mstudent2_step800_deliv_ll", "s25_ll"]:
    j0(f"eval_robustness {lab} UNREG/REG_FRAME/REG_CLIP",
       [(lp(c, lab, v), ER[c]["configs"][lab]["lpips_clip"][v]) for c in CL for v in VARS], 1e-4)
SDE = json.load(open("scripts/distill/runs/deep_20261004/sdedit/TABLE_S2_EXT12.json"))
for lab in ["origin_g100_sd7", "origin_g100_sd1fill", "deliv_g100_sd7", "deliv_g100_sd1fill"]:
    r = SDE["results"][lab]
    pairs = []
    for i, c in enumerate(r["clips"]):
        if c not in S:
            continue
        for v, k in (("REG_FRAME", "dREG_FRAME_vs_origin"), ("REG_CLIP", "dREG_CLIP_vs_origin"), ("UNREG", "dUNREG_vs_origin")):
            pairs.append((lp(c, lab, v) - lp(c, O, v), r[k][i]))
    j0(f"sdedit {lab} deltas vs origin (3 variants)", pairs, 2e-4)
txt = open("scripts/distill/runs/deep_20261004/sdedit/TABLE_S2_EXT12.txt").read()
pairs = []
for sec, v in (("REG_FRAME (PRIMARY", "REG_FRAME"), ("REG_CLIP (one GT", "REG_CLIP"), ("UNREGISTERED LPIPS", "UNREG")):
    blk = txt[txt.index(sec):]
    hdr = blk.split("\n")[1].split()[1:13]
    m = re.search(r"\n\s+INPUT_fill\s+([^\n]+)", blk)
    vals = [float(x) for x in m.group(1).split()[:12]]
    for c, x in zip(hdr, vals):
        if c in S:
            pairs.append((lp(c, "INPUT_fill", v), x))
j0("sdedit INPUT_fill (table printed at 4 dp)", pairs, 2e-4)
EM = json.load(open("scripts/distill/runs/deep_20261004/external_models/TABLE_12CLIP_r2.json"))
# scale_gt and external_models registered 0125 on the PREREG grid (their origin mean 0.2430), this lane and
# eval_robustness/sdedit/skeptic on the ADDENDUM wide grid (0.2432): 0125's REG_* are compared only for the UNREG column.
GRIDOK = lambda c, v: not (c == "0125" and v != "UNREG")
j0("external_models M2SVid (CPU-scored; 0125 REG on another grid, UNREG only)",
   [(lp(c, "m2svid_fa_w16_ll", v), EM[c]["m2"][v]) for c in CL for v in VARS if GRIDOK(c, v)], 2e-4)
pairs = []
for c in CL:
    g = json.load(open(f"outputs/deep_20261004/scale_gt/score_reg_test_main_v1_s250/{c}.json"))
    for lab in ("selMain250_s8", "selMain250_s25"):
        pairs += [(lp(c, lab, v), g["configs"][lab]["lpips_clip"][v]) for v in VARS if GRIDOK(c, v)]
j0("scale_gt selMain250 s8/s25 (0125 REG on another grid, UNREG only)", pairs, 1e-4)
pairs = []
for c in CL:
    k = json.load(open(f"outputs/deep_20261004/skeptic/s38_v1/{c}.json"))["rows"]["AYS8"]
    pairs += [(lp(c, "AYS8_origin_g101", "UNREG"), k["lpips_UNREG"]), (lp(c, "AYS8_origin_g101", "REG_FRAME"), k["lpips_REG_FRAME"])]
j0("skeptic AYS8 UNREG/REG_FRAME (CPU-scored, 5 dp)", pairs, 2e-4)
BD = json.load(open("scripts/distill/runs/deep_20261004/blur_diag/TABLE_BLUR_DIAG_r2.json"))["R4"]
pairs = []
for lab in ("HIRES_B", "HIRES_B_L"):
    for c in ("0042", "0170"):
        if c in S and has(c, lab):
            pairs += [(lp(c, lab, "REG_FRAME"), BD[lab]["LPIPS_REG"]["row"][c]), (lp(c, lab, "UNREG"), BD[lab]["LPIPS_UNREG"]["row"][c])]
j0("blur_diag HIRES_B / HIRES_B_L", pairs, 2e-4)
if len(CL) == 12:
    j0("skeptic LEFT_AS_RIGHT 12-clip means (UNREG 0.2814, REG 0.2638)",
       [(float(np.mean([lp(c, "LEFT_AS_RIGHT", "UNREG") for c in CL])), 0.2814),
        (float(np.mean([lp(c, "LEFT_AS_RIGHT", "REG_FRAME") for c in CL])), 0.2638)], 1e-4 + 5e-5)
out["gates"]["J0"] = J0
lw = all(S[c]["configs"][lab]["left_window_ok"] for c in CL for lab in S[c]["configs"])
P(f"  every row's own left-eye search landed on the deployed window: {lw}")

# ---------------------------------------------------------------------------------------------------- levels
P("\n--- LEVELS, 12-clip means (lower LPIPS = closer to the real right eye) ---")
P(f"  {'row':34s} {'UNREG':>7s} {'REG_FRAME':>9s} {'REG_CLIP':>8s} {'rPSNR_reg':>9s}")
for lab in [O] + ROWS:
    cc = [c for c in CL if has(c, lab)]
    if len(cc) < len(CL):
        continue
    P(f"  {NAME.get(lab, lab)[:34]:34s} {np.mean([lp(c, lab, 'UNREG') for c in cc]):7.4f} "
      f"{np.mean([lp(c, lab, 'REG_FRAME') for c in cc]):9.4f} {np.mean([lp(c, lab, 'REG_CLIP') for c in cc]):8.4f} "
      f"{np.mean([S[c]['configs'][lab]['rPSNR']['REG_FRAME'] for c in cc]):9.3f}")

# ---------------------------------------------------------------------------------------------------- contrasts + flags
P("\n--- CONTRASTS vs ORIGIN (row - origin; negative = better), bar = mean dREG_FRAME <= -0.005 AND >= 9/12 improved ---")


def ratio(c, lab, key, sub):
    a = S[c]["configs"][lab][sub][key]
    b = S[c]["configs"][O][sub][key]
    return a / b


for lab in ROWS:
    cc = [c for c in CL if has(c, lab)]
    if not cc:
        continue
    R = dict(name=NAME[lab], clips=cc, sep=SEP[lab])
    for v in VARS + ["BLK_LOCAL"]:
        d = [lp(c, lab, v) - lp(c, O, v) for c in cc]
        R[v] = dict(per_clip=d, mean=float(np.mean(d)), improved=int(sum(x < 0 for x in d)), n=len(d),
                    ci95=boot_ci(d), mean_n11=float(np.mean([x for c, x in zip(cc, d) if c != "0125"])) if len(cc) == 12 else None)
    R["dPSNR_reg"] = float(np.mean([S[c]["configs"][lab]["rPSNR"]["REG_FRAME"] - S[c]["configs"][O]["rPSNR"]["REG_FRAME"] for c in cc]))
    for key in ("edgeHF", "flatHF", "stripeE"):
        rr = [ratio(c, lab, key, "decomp_regframe") for c in cc]
        R[f"{key}_ratio"] = dict(per_clip=rr, mean=float(np.mean(rr)), n_gt_1p10=int(sum(x > 1.10 for x in rr)))
        rg = [S[c]["configs"][lab]["decomp_regframe"][key] / S[c]["gt_decomp_regframe"][str(S[c]["configs"][lab]["n"])]["gt"][key] for c in cc]
        R[f"{key}_vsGT"] = float(np.mean(rg))
    for key in ("selfEdgeHF", "sharp"):
        rr = [ratio(c, lab, key, "self") for c in cc]
        R[f"{key}_ratio"] = dict(per_clip=rr, mean=float(np.mean(rr)))
    if NR and all(c in NR for c in cc):
        R["dNIQE"] = float(np.mean([NR[c]["rows"][lab]["clip"]["niqe"] - NR[c]["rows"][O]["clip"]["niqe"] for c in cc]))
        R["dMUSIQ"] = float(np.mean([NR[c]["rows"][lab]["clip"]["musiq"] - NR[c]["rows"][O]["clip"]["musiq"] for c in cc]))
        R["NIQE_worse_clips"] = int(sum(NR[c]["rows"][lab]["clip"]["niqe"] > NR[c]["rows"][O]["clip"]["niqe"] for c in cc))
        R["MUSIQ_worse_clips"] = int(sum(NR[c]["rows"][lab]["clip"]["musiq"] < NR[c]["rows"][O]["clip"]["musiq"] for c in cc))
    fr, fu = R["REG_FRAME"]["mean"], R["UNREG"]["mean"]
    F = {}
    F["F1_UNREG>REG"] = (-fu) > (-fr) + 0.002
    F["F2_PSNR_texture"] = R["dPSNR_reg"] < -0.10 and fr < 0
    F["F3_blur"] = R["edgeHF_ratio"]["mean"] < 0.95 or R["selfEdgeHF_ratio"]["mean"] < 0.95
    F["F4_flat_texture"] = R["flatHF_ratio"]["mean"] > 1.10 and R["edgeHF_ratio"]["mean"] < 1.05
    F["F5_stripes"] = R["stripeE_ratio"]["mean"] > 1.10 or R["stripeE_ratio"]["n_gt_1p10"] >= 3
    F["F6_NR_worse"] = (R["dNIQE"] > 0.10 and R["dMUSIQ"] < -0.55) if "dNIQE" in R else None
    F["F8_separation"] = SEP[lab].startswith("NOT CLEAN")
    R["flags"] = F
    R["bar"] = (len(cc) == 12 and fr <= -0.005 and R["REG_FRAME"]["improved"] >= 9)
    if len(cc) == 12 and lab != "AYS8_origin_g101":
        ds = [lp(c, lab, "REG_FRAME") - lp(c, "AYS8_origin_g101", "REG_FRAME") for c in cc]
        du = [lp(c, lab, "UNREG") - lp(c, "AYS8_origin_g101", "UNREG") for c in cc]
        R["vsAYS8"] = dict(REG_FRAME=float(np.mean(ds)), improved=int(sum(x < 0 for x in ds)), ci95=boot_ci(ds),
                           UNREG=float(np.mean(du)), improved_unreg=int(sum(x < 0 for x in du)))
        R["strict_bar"] = R["vsAYS8"]["REG_FRAME"] <= -0.005 and R["vsAYS8"]["improved"] >= 9
    if len(cc) == 12 and lab != "mstudent2_step800_deliv_ll":
        ds = [lp(c, lab, "REG_FRAME") - lp(c, "mstudent2_step800_deliv_ll", "REG_FRAME") for c in cc]
        R["vsDeliv"] = dict(REG_FRAME=float(np.mean(ds)), improved=int(sum(x < 0 for x in ds)), ci95=boot_ci(ds))
    out["rows"][lab] = R
    P(f"\n[{NAME[lab]}]  ({lab}; {len(cc)} clips; separation: {SEP[lab]})")
    for v in VARS:
        x = R[v]
        P(f"   d{v:9s} mean {x['mean']:+.4f}  improved {x['improved']:2d}/{x['n']}  CI95 [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}]"
          + (f"  n11(no 0125) {x['mean_n11']:+.4f}" if x["mean_n11"] is not None else "")
          + "   per clip " + " ".join(f"{c}:{d:+.4f}" for c, d in zip(cc, x["per_clip"])))
    P(f"   registered PSNR change {R['dPSNR_reg']:+.3f} dB | ratios vs origin: edgeHF {R['edgeHF_ratio']['mean']:.3f}  "
      f"flatHF {R['flatHF_ratio']['mean']:.3f}  stripeE {R['stripeE_ratio']['mean']:.3f} (>1.10 on {R['stripeE_ratio']['n_gt_1p10']} clips)  "
      f"selfEdge {R['selfEdgeHF_ratio']['mean']:.3f}  sharp {R['sharp_ratio']['mean']:.3f}")
    P(f"   vs GT: edgeHF {R['edgeHF_vsGT']:.3f}  flatHF {R['flatHF_vsGT']:.3f}  stripeE {R['stripeE_vsGT']:.3f}"
      + (f" | NR: dNIQE {R['dNIQE']:+.3f} (worse on {R['NIQE_worse_clips']}/{len(cc)})  dMUSIQ {R['dMUSIQ']:+.2f} (worse on {R['MUSIQ_worse_clips']}/{len(cc)})" if "dNIQE" in R else " | NR: n/a"))
    if "vsAYS8" in R:
        a = R["vsAYS8"]
        P(f"   vs AYS8: dREG_FRAME {a['REG_FRAME']:+.4f} improved {a['improved']}/12 CI95 [{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}]; "
          f"dUNREG {a['UNREG']:+.4f} ({a['improved_unreg']}/12) -> strict bar {'MET' if R['strict_bar'] else 'not met'}")
    if "vsDeliv" in R:
        a = R["vsDeliv"]
        P(f"   vs deliverable: dREG_FRAME {a['REG_FRAME']:+.4f} improved {a['improved']}/12 CI95 [{a['ci95'][0]:+.4f},{a['ci95'][1]:+.4f}]")
    fl = [k for k, v in F.items() if v]
    P(f"   BAR {'MET' if R['bar'] else 'not met'}; flags raised: {fl if fl else 'none'}")

# ---------------------------------------------------------------------------------------------------- reference levels
if NR:
    P("\n--- NO-REFERENCE LEVELS (12-clip means; NIQE lower = better, MUSIQ higher = better) ---")
    for lab in ["GT_REGFRAME", O] + ROWS:
        cc = [c for c in CL if c in NR and lab in NR[c]["rows"]]
        if len(cc) == len(CL):
            P(f"  {NAME.get(lab, lab)[:34]:34s} NIQE {np.mean([NR[c]['rows'][lab]['clip']['niqe'] for c in cc]):6.3f}  "
              f"MUSIQ {np.mean([NR[c]['rows'][lab]['clip']['musiq'] for c in cc]):6.2f}")
P("\n--- origin's own decomposition vs GT as the MEAN OF PER-CLIP RATIOS (same statistic as the rows' 'vs GT' line) ---")
for key in ("edgeHF", "flatHF", "stripeE"):
    rg = [S[c]["configs"][O]["decomp_regframe"][key] / S[c]["gt_decomp_regframe"][str(S[c]["configs"][O]["n"])]["gt"][key] for c in CL]
    P(f"  origin {key:8s} vs GT {np.mean(rg):.3f}   per clip " + " ".join(f"{c}:{x:.2f}" for c, x in zip(CL, rg)))
P("\n--- scale_gt at 25 steps against the matching reference (origin 25 steps, s25_ll) ---")
for v in VARS:
    d = [lp(c, "selMain250_s25", v) - lp(c, "s25_ll", v) for c in CL]
    ci = boot_ci(d)
    P(f"  d{v:9s} (selMain250_s25 - s25_ll) mean {np.mean(d):+.4f}  improved {sum(x < 0 for x in d)}/12  CI95 [{ci[0]:+.4f},{ci[1]:+.4f}]")
    out["rows"].setdefault("selMain250_s25_vs_s25", {})[v] = dict(mean=float(np.mean(d)), improved=int(sum(x < 0 for x in d)), ci95=ci, per_clip=d)
P("\n--- GT vs origin decomposition levels (12-clip means of the absolute values) ---")
for key in ("edgeHF", "flatHF", "stripeE"):
    g = np.mean([S[c]["gt_decomp_regframe"][str(S[c]["configs"][O]["n"])]["gt"][key] for c in CL])
    o = np.mean([S[c]["configs"][O]["decomp_regframe"][key] for c in CL])
    P(f"  {key:8s} GT {g:.5f}  origin {o:.5f}  origin/GT {o / g:.3f}")
json.dump(out, open(OP + ".json", "w"), indent=1)
open(OP + ".txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
