#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- TEST analysis and pre-registered verdicts (PREREG.txt section 5).  CPU only.
usage: python analyze_test_v1.py <run> <sel> <us>
"""
import json
import os
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
L = "scripts/distill/runs/deep_20261004/decoder_ft"
RUN, SEL, US = sys.argv[1:4]
O = f"outputs/deep_20261004/decoder_ft/score_reg_test_{RUN}"
A = f"outputs/deep_20261004/decoder_ft/score_aux_test_{RUN}"
TP = f"outputs/deep_20261004/decoder_ft/temporal_test_{RUN}"
PF = f"outputs/deep_20261004/decoder_ft/pairflicker_test_{RUN}"
TEST = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
AVP, IPH = TEST[:6], TEST[6:]
R = {c: json.load(open(f"{O}/{c}.json"))["configs"] for c in TEST}
X = {c: json.load(open(f"{A}/{c}.json")) for c in TEST}
TJ = {c: json.load(open(f"{TP}/{c}.json"))[c] for c in TEST}
PJ = {c: json.load(open(f"{PF}/{c}.json")) for c in TEST}
RT = json.load(open(f"{L}/ROUNDTRIP_TEST_{RUN}.json"))
rng = np.random.default_rng(0)


def boot(d, B=10000):
    d = np.asarray(d)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def reg(c, lab, v="REG_FRAME"):
    return R[c][lab]["lpips_clip"][v]


def aux(c, lab, k):
    return X[c]["labels"][lab][k]


def dec(c, lab, k):
    return X[c]["labels"][lab]["decompose"][k]


def line(name, d, clips=TEST, w=8, higher_better=False):
    d = np.asarray(d)
    lo, hi = boot(d)
    sg = -1.0 if higher_better else 1.0          # "better" = lower, or higher for rPSNR / MUSIQ
    nb = int((sg * d < 0).sum())
    nres = int((sg * d < -0.004).sum())
    nwor = int((sg * d > 0.004).sum())
    return (f"{name:46s} mean {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}] better {nb:2d}/{len(d)} (|d|>0.004: {nres} better, "
            f"{nwor} worse)  AVP {d[:6].mean():+.4f} ({int((sg * d[:6] < 0).sum())}/6)  iPhone {d[6:].mean():+.4f} "
            f"({int((sg * d[6:] < 0).sum())}/6)")


P = print
P(f"decoder_ft TEST ({RUN}): selected decoder {SEL}, unsharp baseline {US}; 12 test clips, ONE evaluation")
P("rows: origin_cap__stock (== published origin_ll, gate C1/C2), deliv_cap__stock (== mstudent2_step800_deliv_ll), +D*, +US*")
P()
P("0. SANITY: stock rows equal the published rows (same md5 -> identical scores)")
for a, b in (("origin_cap__stock", "origin_ll"), ("deliv_cap__stock", "mstudent2_step800_deliv_ll")):
    mx = max(abs(reg(c, a, v) - reg(c, b, v)) for c in TEST for v in ("UNREG", "REG_FRAME", "REG_CLIP"))
    P(f"  {a} vs {b}: max |d| over 12 clips x 3 metrics = {mx:.2e}")
P()
P("1. MEANS (12 clips)")
labs = ["origin_cap__stock", f"origin_cap__{SEL}", f"origin_cap__{US}", "deliv_cap__stock", f"deliv_cap__{SEL}",
        f"deliv_cap__{US}", "AYS8_origin_g101", "s25_ll"]
P(f"  {'row':30s} {'REG_FRAME':>9s} {'REG_CLIP':>9s} {'BLK_LOC':>8s} {'UNREG':>8s} {'rPSNR':>7s} {'DISTS':>7s} {'VGG':>7s} "
  f"{'NIQE':>6s} {'MUSIQ':>6s} {'sharp':>7s} {'flatHF/GT':>9s} {'edgeHF/GT':>9s} {'stripeE/GT':>10s}")
for lab in labs:
    f = lambda k: np.mean([reg(c, lab, k) for c in TEST])
    P(f"  {lab:30s} {f('REG_FRAME'):9.4f} {f('REG_CLIP'):9.4f} {f('BLK_LOCAL'):8.4f} {f('UNREG'):8.4f} "
      f"{np.mean([R[c][lab]['rPSNR']['REG_FRAME'] for c in TEST]):7.3f} {np.mean([aux(c, lab, 'dists') for c in TEST]):7.4f} "
      f"{np.mean([aux(c, lab, 'lpips_vgg') for c in TEST]):7.4f} {np.mean([aux(c, lab, 'niqe') for c in TEST]):6.3f} "
      f"{np.mean([aux(c, lab, 'musiq') for c in TEST]):6.2f} {np.mean([aux(c, lab, 'sharp') for c in TEST]):7.4f} "
      f"{np.mean([dec(c, lab, 'flatHF') / X[c]['GT']['decompose']['flatHF'] for c in TEST]):9.3f} "
      f"{np.mean([dec(c, lab, 'edgeHF') / X[c]['GT']['decompose']['edgeHF'] for c in TEST]):9.3f} "
      f"{np.mean([dec(c, lab, 'stripeE') / X[c]['GT']['decompose']['stripeE'] for c in TEST]):10.3f}")
P(f"  {'GT (real right eye, REG_FRAME)':30s} {'':9s} {'':9s} {'':8s} {'':8s} {'':7s} {'':7s} {'':7s} "
  f"{np.mean([X[c]['GT']['niqe'] for c in TEST]):6.3f} {np.mean([X[c]['GT']['musiq'] for c in TEST]):6.2f} "
  f"{np.mean([X[c]['GT']['sharp'] for c in TEST]):7.4f}")
P()
verdict = {}
for model, lat, pub in (("origin", "origin_cap", "origin_ll"), ("deliverable", "deliv_cap", "mstudent2_step800_deliv_ll")):
    s, t, u = f"{lat}__stock", f"{lat}__{SEL}", f"{lat}__{US}"
    P(f"2. {'T1' if model == 'origin' else 'T2'}: {model} + D* ({SEL}) minus {model} stock decoder, same latents")
    dd = {}
    for v in ("REG_FRAME", "REG_CLIP", "BLK_LOCAL", "UNREG"):
        dd[v] = [reg(c, t, v) - reg(c, s, v) for c in TEST]
        P("  " + line(f"LPIPS-Alex {v}", dd[v]))
    dd["rPSNR"] = [R[c][t]["rPSNR"]["REG_FRAME"] - R[c][s]["rPSNR"]["REG_FRAME"] for c in TEST]
    dd["DISTS"] = [aux(c, t, "dists") - aux(c, s, "dists") for c in TEST]
    dd["VGG"] = [aux(c, t, "lpips_vgg") - aux(c, s, "lpips_vgg") for c in TEST]
    dd["NIQE"] = [aux(c, t, "niqe") - aux(c, s, "niqe") for c in TEST]
    dd["MUSIQ"] = [aux(c, t, "musiq") - aux(c, s, "musiq") for c in TEST]
    for k in ("rPSNR", "DISTS", "VGG", "NIQE", "MUSIQ"):
        P("  " + line(f"{k} change" + (" (dB, higher better)" if k == "rPSNR" else " (MUSIQ higher better)" if k == "MUSIQ"
                                       else " (training-loss family; diagnostic)" if k == "VGG" else ""), dd[k],
                      higher_better=k in ("rPSNR", "MUSIQ")))
    sr = [dec(c, t, "stripeE") / dec(c, s, "stripeE") for c in TEST]
    fr = [dec(c, t, "flatHF") / dec(c, s, "flatHF") for c in TEST]
    er = [dec(c, t, "edgeHF") / dec(c, s, "edgeHF") for c in TEST]
    shr = [aux(c, t, "sharp") / aux(c, s, "sharp") for c in TEST]
    s_flag = [c for c, r_ in zip(TEST, sr) if r_ > 1.10 and dec(c, t, "stripeE") > X[c]["GT"]["decompose"]["stripeE"]]
    f_flag = [c for c, r_ in zip(TEST, fr) if r_ > 1.10 and dec(c, t, "flatHF") > X[c]["GT"]["decompose"]["flatHF"]]
    P(f"  ratios D*/stock: stripeE {np.mean(sr):.3f} (max {max(sr):.3f}; flagged {s_flag or 'none'}), flatHF {np.mean(fr):.3f} "
      f"(max {max(fr):.3f}; flagged {f_flag or 'none'}), edgeHF {np.mean(er):.3f}, sharp {np.mean(shr):.3f}")
    tl = [TJ[c][f"{c}_{t}"]["tLP"] / TJ[c][f"{c}_{s}"]["tLP"] for c in TEST]
    wa = [TJ[c][f"{c}_{t}"]["warp"] / TJ[c][f"{c}_{s}"]["warp"] for c in TEST]
    tl_ok = np.mean(tl) <= 1.05 or np.mean([TJ[c][f"{c}_{t}"]["tLP"] for c in TEST]) <= np.mean([TJ[c]["GT"]["tLP"] for c in TEST])
    wa_ok = np.mean(wa) <= 1.05 or np.mean([TJ[c][f"{c}_{t}"]["warp"] for c in TEST]) <= np.mean([TJ[c]["GT"]["warp"] for c in TEST])
    sm = lambda lab: np.mean([TJ[c][f"{c}_{lab}"]["seam"] / TJ[c][f"{c}_{lab}"]["nonseam"] for c in TEST])
    seam_d = sm(t) - sm(s)
    pr = np.mean([PJ[c]["labels"][t]["ratio"] - PJ[c]["labels"][s]["ratio"] for c in TEST])
    P(f"  temporal: tLP ratio {np.mean(tl):.3f} (D* {np.mean([TJ[c][f'{c}_{t}']['tLP'] for c in TEST]):.4f}, stock "
      f"{np.mean([TJ[c][f'{c}_{s}']['tLP'] for c in TEST]):.4f}, GT {np.mean([TJ[c]['GT']['tLP'] for c in TEST]):.4f}); "
      f"warp ratio {np.mean(wa):.3f} (GT {np.mean([TJ[c]['GT']['warp'] for c in TEST]):.4f}); seam ratio {sm(s):.3f} -> {sm(t):.3f} "
      f"(d {seam_d:+.3f}); pair-flicker ratio {np.mean([PJ[c]['labels'][s]['ratio'] for c in TEST]):.4f} -> "
      f"{np.mean([PJ[c]['labels'][t]['ratio'] for c in TEST]):.4f} (d {pr:+.4f}; GT {np.mean([PJ[c]['GT']['ratio'] for c in TEST]):.4f})")
    du = {v: [reg(c, u, v) - reg(c, s, v) for c in TEST] for v in ("REG_FRAME", "UNREG")}
    du["DISTS"] = [aux(c, u, "dists") - aux(c, s, "dists") for c in TEST]
    du["rPSNR"] = [R[c][u]["rPSNR"]["REG_FRAME"] - R[c][s]["rPSNR"]["REG_FRAME"] for c in TEST]
    P("  unsharp baseline " + US + ": " + line("REG_FRAME", du["REG_FRAME"]))
    P(f"     DISTS {np.mean(du['DISTS']):+.4f}  UNREG {np.mean(du['UNREG']):+.4f}  rPSNR {np.mean(du['rPSNR']):+.3f} dB")
    m = {k: float(np.mean(v)) for k, v in dd.items()}
    g = dict(g1=m["rPSNR"] >= -0.10, g2=m["NIQE"] <= 0.05, g3=m["DISTS"] < 0,
             g4=(np.mean(sr) <= 1.10 and len(s_flag) < 3 and np.mean(fr) <= 1.10 and len(f_flag) < 3),
             g5=bool(tl_ok and wa_ok and seam_d <= 0.05 and pr <= 0.05),
             g6=(m["REG_FRAME"] < float(np.mean(du["REG_FRAME"])) and m["DISTS"] < float(np.mean(du["DISTS"]))),
             g7=abs(m["UNREG"]) <= 2 * abs(m["REG_FRAME"]))
    nb = int(sum(x < 0 for x in dd["REG_FRAME"]))
    allg = all(g.values())
    if m["REG_FRAME"] <= -0.005 and nb >= 9 and m["UNREG"] < 0 and allg:
        v_ = "DECODER GAIN"
    elif m["REG_FRAME"] < 0 and nb >= 9 and allg:
        v_ = "SMALL GAIN"
    else:
        v_ = "NO GAIN"
    verdict[model] = dict(verdict=v_, guards=g, mean=m, better=nb)
    P(f"  guards: " + "  ".join(f"{k} {'pass' if ok else 'FAIL'}" for k, ok in g.items()) + "   (g8 visual: VISUAL_TEST file)")
    P(f"  VERDICT {model}: {v_}  (REG_FRAME {m['REG_FRAME']:+.4f}, better {nb}/12, UNREG {m['UNREG']:+.4f})")
    P()
P("3. T3 ROUND BAR and T4 references (REG_FRAME; candidate minus reference)")
for cand in (f"origin_cap__{SEL}", f"deliv_cap__{SEL}", "deliv_cap__stock", f"deliv_cap__{US}"):
    for ref in ("origin_ll", "AYS8_origin_g101", "s25_ll"):
        P("  " + line(f"{cand} - {ref}", [reg(c, cand) - reg(c, ref) for c in TEST]))
    P("  " + line(f"{cand} - origin_ll (UNREG)", [reg(c, cand, 'UNREG') - reg(c, 'origin_ll', 'UNREG') for c in TEST]))
P()
P("4. DECODER-ONLY ROUND TRIP of the real right eye (diagnostic; blur_diag's VAE_GT input; registration-free)")
for d_ in RT["decoders"]:
    mm = RT["mean"][d_]
    P(f"  {d_:10s} LPIPS-Alex {mm['lpips']:.4f}  PSNR {mm['psnr']:.3f}  DISTS {mm['dists']:.4f}")
P()
P("5. PER-CLIP REG_FRAME (stock -> D*), origin | deliverable")
for c in TEST:
    P(f"  {c}  origin {reg(c, 'origin_cap__stock'):.4f} -> {reg(c, f'origin_cap__{SEL}'):.4f} "
      f"({reg(c, f'origin_cap__{SEL}') - reg(c, 'origin_cap__stock'):+.4f})   deliverable {reg(c, 'deliv_cap__stock'):.4f} -> "
      f"{reg(c, f'deliv_cap__{SEL}'):.4f} ({reg(c, f'deliv_cap__{SEL}') - reg(c, 'deliv_cap__stock'):+.4f})")
json.dump(verdict, open(f"{L}/VERDICT_TEST_{RUN}.json", "w"), indent=1)
P()
P("VERDICTS: " + "; ".join(f"{k}: {v['verdict']}" for k, v in verdict.items()))
