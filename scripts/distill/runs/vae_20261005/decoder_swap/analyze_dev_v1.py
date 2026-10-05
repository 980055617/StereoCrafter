#!/usr/bin/env python
"""vae_20261005 / decoder_swap -- DEV analysis and the pre-registered pass rule / selection (PREREG.txt section 3).  CPU only.

Reads (this lane): outputs/vae_20261005/decoder_swap/{score_reg_dev_v1,score_aux_dev_v1,temporal_dev_v1,pairflicker_dev_v1}/<clip>.json,
/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev/<clip>_<lab>/redecode.json, GATE_D1_dev.txt, GATE_D3_cd_determinism.txt.
Reference for gate D2: decoder_ft outputs/deep_20261004/decoder_ft/score_reg_dev_main_v1/<clip>.json (stock rows, read only).
Writes the table to stdout and SELECTION_DEV_v1.json next to this file.
usage: python analyze_dev_v1.py
"""
import json
import os

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
L = "scripts/distill/runs/vae_20261005/decoder_swap"
O = "outputs/vae_20261005/decoder_swap"
R = "/mnt/ssd_data/vae_20261005/decoder_swap/redec_dev"
DFT = "outputs/deep_20261004/decoder_ft/score_reg_dev_main_v1"
DEV = "0040 0082 0091 0184 0245 0268".split()
REGC = "0040 0082 0184 0245 0268".split()
MODELS = {"origin": "origin_cap", "deliverable": "deliv_cap"}
CANDS = ["ftmse", "ftema", "cd"]
ROWS = ["stock32"] + CANDS
COST_ORDER = {"ftmse": 0, "ftema": 0, "cd": 1}
OUTJ = f"{L}/SELECTION_DEV_v1.json"
assert not os.path.exists(OUTJ), f"refusing to overwrite {OUTJ}"

REG = {c: json.load(open(f"{O}/score_reg_dev_v1/{c}.json"))["configs"] for c in DEV}
AUX = {c: json.load(open(f"{O}/score_aux_dev_v1/{c}.json")) for c in DEV}
TMP = {c: json.load(open(f"{O}/temporal_dev_v1/{c}.json"))[c] for c in DEV}
PF = {c: json.load(open(f"{O}/pairflicker_dev_v1/{c}.json")) for c in DEV}


def lab(m, d):
    return f"{MODELS[m]}__{d}"


def reg(c, m, d, v="REG_FRAME"):
    return REG[c][lab(m, d)]["lpips_clip"][v]


def rpsnr(c, m, d):
    return REG[c][lab(m, d)]["rPSNR"]["REG_FRAME"]


def aux(c, m, d, k):
    return AUX[c]["labels"][lab(m, d)][k]


def dec(c, m, d, k):
    return AUX[c]["labels"][lab(m, d)]["decompose"][k]


def tmp(c, m, d, k):
    return TMP[c][f"{c}_{lab(m, d)}"][k]


def seamr(c, m, d):
    t = TMP[c][f"{c}_{lab(m, d)}"]
    return t["seam"] / t["nonseam"]


def timing(c, m, d):
    return json.load(open(f"{R}/{c}_{lab(m, d)}/redecode.json"))


# ------------------------------------------------------------------------------------------------ gates
gates = {}
d1 = open(f"{L}/GATE_D1_dev.txt").read()
gates["D1"] = "GATE D1 ALL PASS" in d1
dft = {c: json.load(open(f"{DFT}/{c}.json"))["configs"] for c in DEV}
d2 = []
for c in DEV:
    for m in MODELS:
        for v in ("REG_FRAME", "REG_CLIP", "BLK_LOCAL", "UNREG"):
            if c in REGC or v == "UNREG":
                d2.append(abs(reg(c, m, "stock", v) - dft[c][lab(m, "stock")]["lpips_clip"][v]))
gates["D2_max_abs_diff"] = max(d2)
gates["D2"] = max(d2) <= 1e-4
d3p = f"{L}/GATE_D3_cd_determinism.txt"
gates["D3"] = os.path.exists(d3p) and "GATE D3 PASS" in open(d3p).read()
for c in DEV:          # shared left half + identical left alignment for every row of a clip
    md = {REG[c][k]["md5_left"] for k in REG[c]}
    dd = {(REG[c][k]["dy"], REG[c][k]["dx"]) for k in REG[c]}
    assert len(md) == 1 and len(dd) == 1, (c, md, dd)
print(f"GATES: D1 stock md5 == capture: {gates['D1']}; D2 stock rows reproduce decoder_ft scorer values: max|d| "
      f"{gates['D2_max_abs_diff']:.2e} -> {gates['D2']}; D3 cd determinism: {gates['D3']}; shared left half/alignment: asserted")
for m in MODELS:
    print(f"  stock {m}: REG_FRAME 5-clip {np.mean([reg(c, m, 'stock') for c in REGC]):.4f}  UNREG 6-clip "
          f"{np.mean([reg(c, m, 'stock', 'UNREG') for c in DEV]):.4f}  (decoder_ft TABLE_DEV_main_v1: origin 0.2568/0.4616, "
          f"deliverable 0.2507/0.4561)")


# ------------------------------------------------------------------------------------------------ deltas
def summary(d, m, base="stock"):
    s = {}
    dr = [reg(c, m, d) - reg(c, m, base) for c in REGC]
    s["dREG_FRAME"], s["REG_better"] = float(np.mean(dr)), int(sum(x < 0 for x in dr))
    s["dREG_FRAME_per_clip"] = dict(zip(REGC, dr))
    for v in ("REG_CLIP", "BLK_LOCAL"):
        s[f"d{v}"] = float(np.mean([reg(c, m, d, v) - reg(c, m, base, v) for c in REGC]))
    du = [reg(c, m, d, "UNREG") - reg(c, m, base, "UNREG") for c in DEV]
    s["dUNREG"], s["UNREG_better"] = float(np.mean(du)), int(sum(x < 0 for x in du))
    s["drPSNR"] = float(np.mean([rpsnr(c, m, d) - rpsnr(c, m, base) for c in REGC]))
    s["dNIQE"] = float(np.mean([aux(c, m, d, "niqe") - aux(c, m, base, "niqe") for c in DEV]))
    s["NIQE_better"] = int(sum(aux(c, m, d, "niqe") < aux(c, m, base, "niqe") for c in DEV))
    s["dMUSIQ"] = float(np.mean([aux(c, m, d, "musiq") - aux(c, m, base, "musiq") for c in DEV]))
    s["dDISTS"] = float(np.mean([aux(c, m, d, "dists") - aux(c, m, base, "dists") for c in REGC]))
    s["dVGG"] = float(np.mean([aux(c, m, d, "lpips_vgg") - aux(c, m, base, "lpips_vgg") for c in REGC]))
    for k in ("edgeHF", "flatHF", "stripeE"):
        s[f"{k}_ratio"] = float(np.mean([dec(c, m, d, k) / dec(c, m, base, k) for c in REGC]))
        s[f"{k}_mean"] = float(np.mean([dec(c, m, d, k) for c in REGC]))
    s["dhaloFrac"] = float(np.mean([dec(c, m, d, "haloFrac") - dec(c, m, base, "haloFrac") for c in REGC]))
    s["tLP_ratio"] = float(np.mean([tmp(c, m, d, "tLP") / tmp(c, m, base, "tLP") for c in DEV]))
    s["warp_ratio"] = float(np.mean([tmp(c, m, d, "warp") / tmp(c, m, base, "warp") for c in DEV]))
    s["warp_mean"] = float(np.mean([tmp(c, m, d, "warp") for c in DEV]))
    s["dseam"] = float(np.mean([seamr(c, m, d) - seamr(c, m, base) for c in DEV]))
    s["pair_ratio"] = float(np.mean([PF[c]["labels"][lab(m, d)]["ratio"] for c in DEV]))
    return s


GTL = dict(warp=float(np.mean([TMP[c]["GT"]["warp"] for c in DEV])), tLP=float(np.mean([TMP[c]["GT"]["tLP"] for c in DEV])),
           stripeE=float(np.mean([AUX[c]["GT"]["decompose"]["stripeE"] for c in REGC])),
           flatHF=float(np.mean([AUX[c]["GT"]["decompose"]["flatHF"] for c in REGC])),
           edgeHF=float(np.mean([AUX[c]["GT"]["decompose"]["edgeHF"] for c in REGC])),
           niqe=float(np.mean([AUX[c]["GT"]["niqe"] for c in DEV])), musiq=float(np.mean([AUX[c]["GT"]["musiq"] for c in DEV])),
           pair=float(np.mean([PF[c]["GT"]["ratio"] for c in DEV])),
           seam=float(np.mean([TMP[c]["GT"]["seam"] / TMP[c]["GT"]["nonseam"] for c in DEV])))
S = {d: {m: summary(d, m) for m in MODELS} for d in ROWS}
S_vs32 = {d: {m: summary(d, m, base="stock32") for m in MODELS} for d in CANDS}


def passes(d):
    a, o = S[d]["deliverable"], S[d]["origin"]
    r = dict(P1=a["REG_better"] >= 4 and a["dREG_FRAME"] < 0, P2=o["dREG_FRAME"] <= 0, G1=a["dNIQE"] <= 0.05,
             G2=a["drPSNR"] >= -0.30, G3=a["dUNREG"] <= 0.001,
             G4=(a["warp_ratio"] <= 1.10) or (a["warp_mean"] <= GTL["warp"]), G5=a["dseam"] <= 0.05,
             G6=((a["stripeE_ratio"] <= 1.10) or (a["stripeE_mean"] <= GTL["stripeE"])) and
                ((a["flatHF_ratio"] <= 1.10) or (a["flatHF_mean"] <= GTL["flatHF"])))
    r["PASS"] = all(r.values())
    return r


P = {d: passes(d) for d in CANDS}
passing = [d for d in CANDS if P[d]["PASS"]]
sel = None
if passing and gates["D1"] and gates["D2"] and gates["D3"]:
    best = min(S[d]["deliverable"]["dREG_FRAME"] for d in passing)
    near = [d for d in passing if S[d]["deliverable"]["dREG_FRAME"] <= best + 0.0005]
    sel = sorted(near, key=lambda d: (COST_ORDER[d], S[d]["deliverable"]["dREG_FRAME"]))[0]

# ------------------------------------------------------------------------------------------------ print
H = ("row      model        dREG_FRAME  better  dREG_CLIP  dBLK_LOC  dUNREG  better  drPSNR   dNIQE  dMUSIQ  dDISTS   dVGG   "
     "edgeHF/st flatHF/st stripe/st dhalo%  tLP/st warp/st dseam  pair")
print("\nDEV deltas vs the STOCK decoder on the same latents (REG_*, rPSNR, DISTS, VGG, decompose: 5 registered clips; "
      "UNREG, NIQE, MUSIQ, temporal: 6 clips)")
print(H)
for d in ROWS:
    for m in MODELS:
        s = S[d][m]
        print(f"{d:8s} {m:11s} {s['dREG_FRAME']:+.4f}     {s['REG_better']}/5   {s['dREG_CLIP']:+.4f}   {s['dBLK_LOCAL']:+.4f}  "
              f"{s['dUNREG']:+.4f}  {s['UNREG_better']}/6   {s['drPSNR']:+.3f}  {s['dNIQE']:+.3f}  {s['dMUSIQ']:+.2f}  "
              f"{s['dDISTS']:+.4f} {s['dVGG']:+.4f}  {s['edgeHF_ratio']:.3f}     {s['flatHF_ratio']:.3f}     "
              f"{s['stripeE_ratio']:.3f}   {s['dhaloFrac']:+.3f}  {s['tLP_ratio']:.3f}  {s['warp_ratio']:.3f}  "
              f"{s['dseam']:+.3f} {s['pair_ratio']:.3f}")
print("\nDECODER EFFECT net of precision: candidate minus STOCK32 (same temporal decoder in fp32) on the same latents")
for d in CANDS:
    for m in MODELS:
        s = S_vs32[d][m]
        print(f"{d:8s} {m:11s} dREG_FRAME {s['dREG_FRAME']:+.4f} ({s['REG_better']}/5)  dUNREG {s['dUNREG']:+.4f}  drPSNR "
              f"{s['drPSNR']:+.3f}  dNIQE {s['dNIQE']:+.3f}  edgeHF {s['edgeHF_ratio']:.3f} flatHF {s['flatHF_ratio']:.3f} "
              f"stripeE {s['stripeE_ratio']:.3f}  warp {s['warp_ratio']:.3f}")
print("\nper-clip REG_FRAME change vs stock (deliverable | origin):")
for d in ROWS:
    print(f"  {d:8s} " + "  ".join(f"{c} {S[d]['deliverable']['dREG_FRAME_per_clip'][c]:+.4f}|{S[d]['origin']['dREG_FRAME_per_clip'][c]:+.4f}"
                                     for c in REGC))
print("\nabsolute levels (5-clip / 6-clip means as above):")
for m in MODELS:
    for d in ["stock"] + ROWS:
        print(f"  {m:11s} {d:8s} REG_FRAME {np.mean([reg(c, m, d) for c in REGC]):.4f} UNREG {np.mean([reg(c, m, d, 'UNREG') for c in DEV]):.4f} "
              f"rPSNR {np.mean([rpsnr(c, m, d) for c in REGC]):.3f} NIQE {np.mean([aux(c, m, d, 'niqe') for c in DEV]):.3f} "
              f"MUSIQ {np.mean([aux(c, m, d, 'musiq') for c in DEV]):.2f} DISTS {np.mean([aux(c, m, d, 'dists') for c in REGC]):.4f} "
              f"edgeHF {np.mean([dec(c, m, d, 'edgeHF') for c in REGC]):.5f} flatHF {np.mean([dec(c, m, d, 'flatHF') for c in REGC]):.5f} "
              f"stripeE {np.mean([dec(c, m, d, 'stripeE') for c in REGC]):.5f} tLP {np.mean([tmp(c, m, d, 'tLP') for c in DEV]):.4f} "
              f"warp {np.mean([tmp(c, m, d, 'warp') for c in DEV]):.5f}")
print(f"  GT (real right eye)  NIQE {GTL['niqe']:.3f} MUSIQ {GTL['musiq']:.2f} edgeHF {GTL['edgeHF']:.5f} flatHF {GTL['flatHF']:.5f} "
      f"stripeE {GTL['stripeE']:.5f} tLP {GTL['tLP']:.4f} warp {GTL['warp']:.5f} seam ratio {GTL['seam']:.3f} pair ratio {GTL['pair']:.3f}")
print("\ndecode cost (GPU-synchronised decode seconds; per-frame decoders decode only kept frames, stock decodes whole windows):")
TIM = {}
for d in ["stock"] + ROWS:
    t = [timing(c, m, d) for c in DEV for m in MODELS]
    TIM[d] = dict(sec_per_clip_as_run=float(np.mean([x["decode_seconds"] for x in t])),
                  sec_per_frame=float(np.mean([x["sec_per_decoded_frame"] for x in t])),
                  deployed_structure_sec_per_clip=float(np.mean([x["deployed_structure_seconds"] for x in t])))
    print(f"  {d:8s} {TIM[d]['sec_per_frame']:.3f} s/decoded frame; {TIM[d]['sec_per_clip_as_run']:.1f} s/clip as run; "
          f"{TIM[d]['deployed_structure_sec_per_clip']:.1f} s/clip if whole 14-frame windows are decoded")
print("\nPASS RULE (PREREG section 3): P1 deliv REG_FRAME >=4/5 & mean<0 | P2 origin mean<=0 | G1 dNIQE<=+0.05 | G2 drPSNR>=-0.30 | "
      "G3 dUNREG<=+0.001 | G4 warp ratio<=1.10 or <=GT | G5 dseam<=+0.05 | G6 stripeE & flatHF ratio<=1.10 or <=GT")
for d in CANDS:
    print(f"  {d:6s} " + "  ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in P[d].items()))
print(f"\nSELECTED for test: {sel if sel else 'NONE (no candidate passes; nothing goes to test)'}")
json.dump(dict(gates=gates, summary=S, summary_vs_stock32=S_vs32, GT=GTL, pass_rule=P, passing=passing, selected=sel,
               timing=TIM), open(OUTJ, "w"), indent=1)
print(f"wrote {OUTJ}")
