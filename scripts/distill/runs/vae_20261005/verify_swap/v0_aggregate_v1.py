#!/usr/bin/env python
"""vae_20261005 / verify_swap -- V0 (PREREG.txt): recompute decoder_swap's quoted dev / headroom means from its PER-CLIP JSONs
with code written here (not analyze_dev_v1.py / analyze_headroom_v1.py), and compare with SELECTION_DEV_v1.json "summary".
CPU only.  Reads only.  usage: python v0_aggregate_v1.py > AGG_CHECK_v1.txt
"""
import json
import math

import numpy as np

O = "/home/kawa/master_project/StereoCrafter/outputs/vae_20261005/decoder_swap"
SEL = "/home/kawa/master_project/StereoCrafter/scripts/distill/runs/vae_20261005/decoder_swap/SELECTION_DEV_v1.json"
DEV6 = ["0040", "0082", "0091", "0184", "0245", "0268"]
REG5 = ["0040", "0082", "0184", "0245", "0268"]          # 0091 outside the registration grid (decoder_swap PREREG section 3)
MOD = {"origin": "origin_cap", "deliverable": "deliv_cap"}
CANDS = ["stock32", "ftmse", "ftema", "cd"]

reg = {c: json.load(open(f"{O}/score_reg_dev_v1/{c}.json"))["configs"] for c in DEV6}
aux = {c: json.load(open(f"{O}/score_aux_dev_v1/{c}.json"))["labels"] for c in DEV6}
tmp = {c: json.load(open(f"{O}/temporal_dev_v1/{c}.json"))[c] for c in DEV6}
sel = json.load(open(SEL))["summary"]

worst = 0.0
print("V0 aggregation check: my recomputation from decoder_swap per-clip JSONs vs SELECTION_DEV_v1.json summary")
print(f"{'cand':8s} {'model':12s} {'quantity':16s} {'mine':>12s} {'selection':>12s} {'|d|':>9s}")
for d in CANDS:
    for mname, m in MOD.items():
        s = sel[d][mname]
        L, B = f"{m}__{d}", f"{m}__stock"
        dreg = [reg[c][L]["lpips_clip"]["REG_FRAME"] - reg[c][B]["lpips_clip"]["REG_FRAME"] for c in REG5]
        mine = {
            "dREG_FRAME": float(np.mean(dreg)),
            "REG_better": int(sum(x < 0 for x in dreg)),
            "dUNREG": float(np.mean([reg[c][L]["lpips_clip"]["UNREG"] - reg[c][B]["lpips_clip"]["UNREG"] for c in DEV6])),
            "drPSNR": float(np.mean([reg[c][L]["rPSNR"]["REG_FRAME"] - reg[c][B]["rPSNR"]["REG_FRAME"] for c in REG5])),
            "dNIQE": float(np.mean([aux[c][L]["niqe"] - aux[c][B]["niqe"] for c in DEV6])),
            "edgeHF_ratio": float(np.mean([aux[c][L]["decompose"]["edgeHF"] / aux[c][B]["decompose"]["edgeHF"] for c in REG5])),
            "flatHF_ratio": float(np.mean([aux[c][L]["decompose"]["flatHF"] / aux[c][B]["decompose"]["flatHF"] for c in REG5])),
            "stripeE_ratio": float(np.mean([aux[c][L]["decompose"]["stripeE"] / aux[c][B]["decompose"]["stripeE"] for c in REG5])),
            "warp_ratio": float(np.mean([tmp[c][f"{c}_{L}"]["warp"] / tmp[c][f"{c}_{B}"]["warp"] for c in DEV6])),
            "dseam": float(np.mean([tmp[c][f"{c}_{L}"]["seam"] / tmp[c][f"{c}_{L}"]["nonseam"]
                                    - tmp[c][f"{c}_{B}"]["seam"] / tmp[c][f"{c}_{B}"]["nonseam"] for c in DEV6])),
        }
        for k, v in mine.items():
            ref = s[k]
            dd = abs(v - ref)
            worst = max(worst, dd)
            print(f"{d:8s} {mname:12s} {k:16s} {v:12.6f} {ref:12.6f} {dd:9.2e}")
        print(f"{d:8s} {mname:12s} per-clip dREG_FRAME " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(REG5, dreg)))

print("\nabsolute REG_FRAME 5-clip means (registered clips):")
for mname, m in MOD.items():
    for d in ["stock", "ftmse", "ftema", "cd"]:
        print(f"  {mname:12s} {d:6s} {np.mean([reg[c][f'{m}__{d}']['lpips_clip']['REG_FRAME'] for c in REG5]):.4f}")
print("absolute NIQE 6-clip means:")
for mname, m in MOD.items():
    for d in ["stock", "ftmse"]:
        print(f"  {mname:12s} {d:6s} {np.mean([aux[c][f'{m}__{d}']['niqe'] for c in DEV6]):.3f}")
print(f"  GT (registered real right eye, score_aux) {np.mean([json.load(open(f'{O}/score_aux_dev_v1/{c}.json'))['GT']['niqe'] for c in DEV6]):.3f}")
print("absolute flatHF / edgeHF 5-clip means (deliverable):")
for d in ["stock", "ftmse", "ftema", "cd"]:
    print(f"  {d:6s} flatHF {np.mean([aux[c][f'deliv_cap__{d}']['decompose']['flatHF'] for c in REG5]):.5f} "
          f"edgeHF {np.mean([aux[c][f'deliv_cap__{d}']['decompose']['edgeHF'] for c in REG5]):.5f}")
gtd = {c: json.load(open(f"{O}/score_aux_dev_v1/{c}.json"))["GT"]["decompose"] for c in REG5}
print(f"  GT     flatHF {np.mean([gtd[c]['flatHF'] for c in REG5]):.5f} edgeHF {np.mean([gtd[c]['edgeHF'] for c in REG5]):.5f}")

print("\nheadroom (real right eye latents, 6 clips): my means from headroom_dev_v1/<clip>.json")
hr = {c: json.load(open(f"{O}/headroom_dev_v1/{c}.json")) for c in DEV6}
for r in ["stock", "stock32", "ftmse", "ftema", "cd", "sd15"]:
    lp = [hr[c]["rows"][r]["lpips"] for c in DEV6]
    ps = [hr[c]["rows"][r]["psnr"] for c in DEV6]
    better = sum(hr[c]["rows"][r]["lpips"] < hr[c]["rows"]["stock"]["lpips"] for c in DEV6)
    fl = np.nanmean([hr[c]["rows"][r]["ratio_to_GT"]["flatHF"] for c in DEV6])
    ed = np.nanmean([hr[c]["rows"][r]["ratio_to_GT"]["edgeHF"] for c in DEV6])
    print(f"  {r:8s} LPIPS {np.mean(lp):.4f}  PSNR {np.mean(ps):.3f}  better than stock {better}/6  flatHF/GT {fl:.3f}  edgeHF/GT {ed:.3f}")
lpm = {r: np.mean([hr[c]["rows"][r]["lpips"] for c in DEV6]) for r in ["stock", "ftmse"]}
print(f"  ftmse removes {100 * (1 - lpm['ftmse'] / lpm['stock']):.1f} % of the stock decoder's round-trip LPIPS-Alex")
print(f"\nworst |mine - selection| over all compared quantities: {worst:.2e}  -> {'AGREE (<= 1e-4)' if worst <= 1e-4 else 'MISMATCH'}")
