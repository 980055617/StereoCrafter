#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- DEV analysis and the pre-registered selection (PREREG.txt section 4).  CPU only.

cells: clip x model (origin_cap, deliv_cap).  Registered metrics on REGCLIPS (5; 0091 is outside the registration grid),
UNREG and NIQE on all 6 dev clips.  Every delta is candidate minus the stock decoder on the SAME latents (same clip, same
model).  Writes the table to stdout and SELECTION_DEV_<run>.json next to this file.
usage: python analyze_dev_v1.py <run> "<steps>"
"""
import json
import os
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
L = "scripts/distill/runs/deep_20261004/decoder_ft"
RUN, STEPS = sys.argv[1], [int(s) for s in sys.argv[2].split()]
O = f"outputs/deep_20261004/decoder_ft/score_reg_dev_{RUN}"
A = f"outputs/deep_20261004/decoder_ft/score_aux_dev_{RUN}"
DEV = "0040 0082 0091 0184 0245 0268".split()
REGCLIPS = "0040 0082 0184 0245 0268".split()
MODELS = ["origin_cap", "deliv_cap"]
CK = [f"s{s}" for s in STEPS]
US = [f"us_s{s}a{a}" for s in (1, 2) for a in ("0.15", "0.30", "0.50")]
R = {c: json.load(open(f"{O}/{c}.json"))["configs"] for c in DEV}
X = {c: json.load(open(f"{A}/{c}.json"))["labels"] for c in DEV}


def reg(c, lab, v="REG_FRAME"):
    return R[c][lab]["lpips_clip"][v]


# shared-left-half checks (make_rows_shared_v1.py): every label of a clip has the same left half and left alignment
ROWSJ = json.load(open(f"{L}/rows_dev_{RUN}.json"))
g0 = []
for c in DEV:
    md = {R[c][lab]["md5_left"] for lab in R[c]}
    dd = {(R[c][lab]["dy"], R[c][lab]["dx"]) for lab in R[c]}
    assert len(md) == 1 and len(dd) == 1, (c, md, dd)
    for lab in ("origin_cap__stock", "deliv_cap__stock"):
        g0.append(abs(R[c][lab]["lpips_clip"]["UNREG"] - ROWSJ["cells"][c][lab]["lpips"]))
print(f"[checks] every dev label of a clip: identical left-half md5 and left alignment (asserted); stock rows: registered-"
      f"scorer UNREG vs score_clip_ll ROW max |d| = {max(g0):.2e} (G0-type reproduction, must be <= 1e-6)")
assert max(g0) <= 1e-6


def rpsnr(c, lab):
    return R[c][lab]["rPSNR"]["REG_FRAME"]


def deltas(name):
    d = {k: [] for k in ("REG_FRAME", "REG_CLIP", "BLK_LOCAL", "UNREG", "rPSNR", "NIQE", "DISTS", "MUSIQ", "VGG", "sharp")}
    per = {}
    for m in MODELS:
        s, t = f"{m}__stock", f"{m}__{name}"
        for c in DEV:
            per.setdefault(m, {})[c] = {}
            u = reg(c, t, "UNREG") - reg(c, s, "UNREG")
            d["UNREG"].append(u)
            per[m][c]["UNREG"] = u
            n = X[c][t]["niqe"] - X[c][s]["niqe"]
            d["NIQE"].append(n)
            d["MUSIQ"].append(X[c][t]["musiq"] - X[c][s]["musiq"])
            d["sharp"].append(X[c][t]["sharp"] / X[c][s]["sharp"])
            if c in REGCLIPS:
                for v in ("REG_FRAME", "REG_CLIP", "BLK_LOCAL"):
                    d[v].append(reg(c, t, v) - reg(c, s, v))
                per[m][c]["REG_FRAME"] = reg(c, t) - reg(c, s)
                d["rPSNR"].append(rpsnr(c, t) - rpsnr(c, s))
                d["DISTS"].append(X[c][t]["dists"] - X[c][s]["dists"])
                d["VGG"].append(X[c][t]["lpips_vgg"] - X[c][s]["lpips_vgg"])
    return d, per


rows = {}
print(f"decoder_ft DEV ({RUN}); cells = 5 registered dev clips x 2 models (REG metrics, DISTS, rPSNR) / 6 clips x 2 (UNREG, NIQE)")
print(f"stock references (REG_FRAME, per model, 5-clip mean): " + ", ".join(
    f"{m} {np.mean([reg(c, m + '__stock') for c in REGCLIPS]):.4f}" for m in MODELS) +
    "; UNREG 6-clip: " + ", ".join(f"{m} {np.mean([reg(c, m + '__stock', 'UNREG') for c in DEV]):.4f}" for m in MODELS))
hdr = (f"{'cand':14s} {'dREG_FRAME':>10s} {'better':>6s} {'dREG_CLIP':>9s} {'dBLK_LOC':>9s} {'dUNREG':>8s} {'better':>6s} "
       f"{'drPSNR':>7s} {'dNIQE':>7s} {'dMUSIQ':>7s} {'dDISTS':>8s} {'dVGG':>8s} {'sharp':>6s}  eligible  [orig dREG / deliv dREG]")
print(hdr)
for name in CK + US:
    d, per = deltas(name)
    m = {k: float(np.mean(v)) for k, v in d.items()}
    nb = sum(x < 0 for x in d["REG_FRAME"])
    nu = sum(x < 0 for x in d["UNREG"])
    e1, e2, e3 = m["rPSNR"] >= -0.10, m["NIQE"] <= 0.05, m["UNREG"] <= 0.001
    elig = e1 and e2 and e3
    om = float(np.mean([per["origin_cap"][c]["REG_FRAME"] for c in REGCLIPS]))
    dm = float(np.mean([per["deliv_cap"][c]["REG_FRAME"] for c in REGCLIPS]))
    rows[name] = dict(mean=m, better_reg=nb, better_unreg=nu, eligible=elig, e=[e1, e2, e3], origin_dREG=om, deliv_dREG=dm,
                      per=per)
    print(f"{name:14s} {m['REG_FRAME']:+10.4f} {nb:3d}/10 {m['REG_CLIP']:+9.4f} {m['BLK_LOCAL']:+9.4f} {m['UNREG']:+8.4f} "
          f"{nu:3d}/12 {m['rPSNR']:+7.3f} {m['NIQE']:+7.3f} {m['MUSIQ']:+7.2f} {m['DISTS']:+8.4f} {m['VGG']:+8.4f} "
          f"{m['sharp']:6.3f}  {'YES' if elig else 'no ' + ''.join('x' if not e else '.' for e in (e1, e2, e3))}"
          f"  [{om:+.4f} / {dm:+.4f}]")
el = [n for n in CK if rows[n]["eligible"]]
sel = None
if el:
    best = min(rows[n]["mean"]["REG_FRAME"] for n in el)
    tied = [n for n in el if rows[n]["mean"]["REG_FRAME"] <= best + 0.0005]
    sel = min(tied, key=lambda n: int(n[1:]))
us_best = min(US, key=lambda n: rows[n]["mean"]["REG_FRAME"])
print()
print(f"ELIGIBLE checkpoints: {el or 'none'}")
print(f"SELECTED decoder checkpoint (lowest dREG_FRAME among eligible; ties within 0.0005 -> earlier step): {sel}")
print(f"SELECTED unsharp baseline (lowest dREG_FRAME, no eligibility filter): {us_best} "
      f"(dREG {rows[us_best]['mean']['REG_FRAME']:+.4f}, dDISTS {rows[us_best]['mean']['DISTS']:+.4f})")
print()
print("per-clip REG_FRAME deltas (candidate - stock), origin | deliverable:")
for name in CK + [us_best]:
    per = rows[name]["per"]
    print(f"  {name:14s} " + "  ".join(f"{c}:{per['origin_cap'][c]['REG_FRAME']:+.4f}|{per['deliv_cap'][c]['REG_FRAME']:+.4f}"
                                       for c in REGCLIPS) +
          "   UNREG 0091: " + f"{per['origin_cap']['0091']['UNREG']:+.4f}|{per['deliv_cap']['0091']['UNREG']:+.4f}")
json.dump(dict(run=RUN, steps=STEPS, selected=sel, eligible=el, unsharp_selected=us_best,
               rows={k: {kk: vv for kk, vv in v.items() if kk != "per"} | {"per": v["per"]} for k, v in rows.items()}),
          open(f"{L}/SELECTION_DEV_{RUN}.json", "w"), indent=1)
