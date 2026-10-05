#!/usr/bin/env python
"""Collect the external_models r2 scores (CPU path, deviation D3) and apply the PREREG_r2.txt decision rules.

usage: python analyze_r2.py <out_txt> [clips...]
Sources (every number printed comes from one of these files):
  UNREG rows (sharp, gtSharp, rightPSNR, lpips) : scripts/distill/runs/deep_20261004/external_models/unreg_cpu/<clip>.txt
        (score_clip_ll_cpu_r2.py; origin_ll re-scored in the same call)
  REGISTERED (UNREG/REG_CLIP/REG_FRAME/REG_FRAME_RAW/BLK_LOCAL) for origin, deliverable, M2SVid, all on CPU, paired:
        outputs/deep_20261004/external_models/score_reg_cpu_m2svid_fa_w16_r2/<clip>.json
  PUBLISHED GPU values (origin, deliverable, s25): outputs/more_20261004/eval_robustness/score_v1/<clip>.json
  NR: outputs/deep_20261004/external_models/nr_cpu_m2svid_fa_w16_r2/<clip>.json
  temporal: outputs/deep_20261004/external_models/temporal_cpu_m2svid_fa_w16_r2/<clip>.json
  render metadata: outputs/deep_20261004/external_models/clips/<clip>_m2svid_fa_w16_ll/run_<clip>.json
"""
import glob
import json
import os
import sys

import numpy as np

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
R = "scripts/distill/runs/deep_20261004/external_models"
O = "outputs/deep_20261004/external_models"
TAG = "m2svid_fa_w16"
LABEL = TAG + "_ll"
out_txt = sys.argv[1]
clips = sys.argv[2:] or ["0301", "0204", "0052", "0147", "0042", "0125", "0128", "0141", "0170", "0225", "0251", "0259"]
assert not os.path.exists(out_txt), f"refusing to overwrite {out_txt}"
REGIME = ["0301", "0204", "0052", "0147"]

rows = {}
for fn in sorted(glob.glob(f"{R}/unreg_cpu/*.txt")) + [f"{R}/SCORES_UNREG_CPU_{LABEL}.txt"]:
    if not os.path.exists(fn):
        continue
    for line in open(fn):
        if line.startswith("ROW "):
            kv = dict(t.split("=", 1) for t in line.split()[1:])
            rows[kv["tag"]] = kv
D = {}
for c in clips:
    rj = f"{O}/score_reg_cpu_{TAG}_r2/{c}.json"
    if not os.path.exists(rj) or f"{c}_{LABEL}" not in rows:
        continue
    j = json.load(open(rj))
    v1 = json.load(open(f"outputs/more_20261004/eval_robustness/score_v1/{c}.json"))
    cf = j["configs"]
    d = dict(clip=c)
    # D4: references = PUBLISHED GPU values for every clip; M2SVid = CPU; paired CPU refs kept where they exist
    for k, src, lab in (("origin", v1["configs"], "origin_ll"), ("deliv", v1["configs"], "mstudent2_step800_deliv_ll"),
                        ("m2", cf, LABEL), ("origin_cpu", cf, "origin_ll"),
                        ("deliv_cpu", cf, "mstudent2_step800_deliv_ll")):
        if lab not in src:
            continue
        d[k] = dict(src[lab]["lpips_clip"])
        d[k]["rPSNR_UNREG"] = src[lab]["rPSNR"]["UNREG"]
        d[k]["rPSNR_REG_FRAME"] = src[lab]["rPSNR"]["REG_FRAME"]
        d[k]["md5_left"] = src[lab]["md5_left"]
    for k, lab in (("pub_origin", "origin_ll"), ("pub_deliv", "mstudent2_step800_deliv_ll"), ("pub_s25", "s25_ll")):
        d[k] = dict(v1["configs"][lab]["lpips_clip"])
    d["backend_gate"] = {k: v["max_abs"] for k, v in j["gate"].items()}
    mrow, orow = rows[f"{c}_{LABEL}"], rows[f"{c}_origin_ll"]
    d["sharp"] = dict(m2=float(mrow["sharp"]), origin=float(orow["sharp"]), gt=float(mrow["gtSharp"]))
    d["unreg_rows"] = dict(m2=float(mrow["lpips"]), origin=float(orow["lpips"]))
    d["G0"] = dict(offset_equal=(mrow["dy"], mrow["dx"]) == (orow["dy"], orow["dx"]),
                   left_md5_equal_scored=d["m2"]["md5_left"] == d["origin"]["md5_left"],
                   unreg_two_scripts_equal=abs(float(mrow["lpips"]) - d["m2"]["UNREG"]) < 1e-5)
    rm = f"{O}/clips/{c}_{LABEL}/run_{c}.json"
    if os.path.exists(rm):
        q = json.load(open(rm))
        d["G0"]["left_md5_all_frames_equal"] = q["G0_left_md5_match"]
        d["G1"] = dict(frames=q["G1_frames_match"], finite=q["finite"], lossless=q["lossless_roundtrip"])
        d["sec_per_window"] = float(np.mean(q["sec_per_window"]))
        d["hole_frac_closed"] = q["mask_polarity"]["hole_frac_after_closing_dilation"]
    nj = f"{O}/nr_cpu_{TAG}_r2/{c}.json"
    if os.path.exists(nj):
        d["nr"] = json.load(open(nj))["means"]
    tj = f"{O}/temporal_cpu_{TAG}_r2/{c}.json"
    if os.path.exists(tj):
        t = json.load(open(tj))["rows"]
        d["temporal"] = {k: dict(tLP=t[k]["tLP"], seam_own=(t[k]["seam_ratio_m2grid"] if k == "m2svid"
                                                            else t[k]["seam_ratio_origgrid"]),
                                 seam_m2grid=t[k]["seam_ratio_m2grid"]) for k in t}
    D[c] = d

done = [c for c in clips if c in D]
L = []
P = L.append
P(f"external_models r2 -- M2SVid (full-attention weights, {TAG}: 576x1024, 16-frame windows, 1 step) vs origin / "
  "deliverable")
P("LPIPS-Alex vs the real right eye, SCORE_STEP=4, lossless renders.  origin / deliverable = PUBLISHED GPU values "
  "(score_v1); M2SVid scored on CPU (D3/D4; backend error <= 5e-5, see paired check below)")
P(f"clips scored: {len(done)} {done}")
P("")
P(f"{'clip':5s} | {'UNREG':^26s} | {'REG_FRAME':^26s} | {'M2-orig':^15s} | {'M2-deliv':^15s} | {'sharp/gtSharp':^15s} | "
  f"{'rPSNR REG':^11s}")
P(f"{'':5s} | {'origin':>8s} {'deliv':>8s} {'M2SVid':>8s} | {'origin':>8s} {'deliv':>8s} {'M2SVid':>8s} | "
  f"{'UNREG':>7s} {'REG':>7s} | {'UNREG':>7s} {'REG':>7s} | {'origin':>7s} {'M2SVid':>7s} | {'orig':>5s} {'M2':>5s}")
for c in done:
    d = D[c]
    o, v, m = d["origin"], d["deliv"], d["m2"]
    P(f"{c:5s} | {o['UNREG']:8.4f} {v['UNREG']:8.4f} {m['UNREG']:8.4f} | {o['REG_FRAME']:8.4f} {v['REG_FRAME']:8.4f} "
      f"{m['REG_FRAME']:8.4f} | {m['UNREG'] - o['UNREG']:+7.4f} {m['REG_FRAME'] - o['REG_FRAME']:+7.4f} | "
      f"{m['UNREG'] - v['UNREG']:+7.4f} {m['REG_FRAME'] - v['REG_FRAME']:+7.4f} | "
      f"{d['sharp']['origin'] / d['sharp']['gt']:7.3f} {d['sharp']['m2'] / d['sharp']['gt']:7.3f} | "
      f"{o['rPSNR_REG_FRAME']:5.2f} {m['rPSNR_REG_FRAME']:5.2f}")


def summary(sel, name):
    if not sel:
        return
    mean = lambda k, var: float(np.mean([D[c][k][var] for c in sel]))
    P("")
    P(f"--- {name}: {len(sel)} clips {sel}")
    for var in ("UNREG", "REG_CLIP", "REG_FRAME", "BLK_LOCAL"):
        P(f"  {var:9s} origin {mean('origin', var):.4f} (pub GPU {mean('pub_origin', var):.4f})  deliv "
          f"{mean('deliv', var):.4f}  M2SVid {mean('m2', var):.4f}  | M2-origin {mean('m2', var) - mean('origin', var):+.4f}"
          f" (better {sum(D[c]['m2'][var] < D[c]['origin'][var] for c in sel)}/{len(sel)})  M2-deliv "
          f"{mean('m2', var) - mean('deliv', var):+.4f} (better {sum(D[c]['m2'][var] < D[c]['deliv'][var] for c in sel)}"
          f"/{len(sel)})")
    P(f"  published s25 (GPU): UNREG {mean('pub_s25', 'UNREG'):.4f}  REG_FRAME {mean('pub_s25', 'REG_FRAME'):.4f}")
    pc = [c for c in sel if "origin_cpu" in D[c]]
    if pc:
        mp = lambda k, var: float(np.mean([D[c][k][var] for c in pc]))
        P(f"  PAIRED CPU check on {pc}: M2-origin UNREG {mp('m2', 'UNREG') - mp('origin_cpu', 'UNREG'):+.4f} REG_FRAME "
          f"{mp('m2', 'REG_FRAME') - mp('origin_cpu', 'REG_FRAME'):+.4f};  M2-deliv UNREG "
          f"{mp('m2', 'UNREG') - mp('deliv_cpu', 'UNREG'):+.4f} REG_FRAME {mp('m2', 'REG_FRAME') - mp('deliv_cpu', 'REG_FRAME'):+.4f}")
    P(f"  sharp/gtSharp: origin {np.mean([D[c]['sharp']['origin'] / D[c]['sharp']['gt'] for c in sel]):.3f}  "
      f"M2SVid {np.mean([D[c]['sharp']['m2'] / D[c]['sharp']['gt'] for c in sel]):.3f}")
    P(f"  rightPSNR REG_FRAME: origin {mean('origin', 'rPSNR_REG_FRAME'):.3f}  deliv {mean('deliv', 'rPSNR_REG_FRAME'):.3f}"
      f"  M2SVid {mean('m2', 'rPSNR_REG_FRAME'):.3f}")
    if all("nr" in D[c] for c in sel):
        for met, lb in (("musiq", "higher=better"), ("clipiqa", "higher=better"), ("niqe", "lower=better")):
            P(f"  NR {met:8s} ({lb}) " + "  ".join(
                f"{k} {np.mean([D[c]['nr'][k][met] for c in sel]):.4f}" for k in ("GT", "origin_ll", "deliv", "m2svid")))
    if all("temporal" in D[c] for c in sel):
        for k in ("GT", "origin", "deliv", "m2svid"):
            P(f"  temporal {k:7s} tLP {np.mean([D[c]['temporal'][k]['tLP'] for c in sel]):.4f}  seam ratio (own grid) "
              f"{np.mean([D[c]['temporal'][k]['seam_own'] for c in sel]):.3f}  (at M2SVid grid "
              f"{np.mean([D[c]['temporal'][k]['seam_m2grid'] for c in sel]):.3f})")
    for ref, rn in (("origin", "origin"), ("deliv", "deliverable")):
        du = mean("m2", "UNREG") - mean(ref, "UNREG")
        dr = mean("m2", "REG_FRAME") - mean(ref, "REG_FRAME")
        wu = sum(D[c]["m2"]["UNREG"] < D[c][ref]["UNREG"] for c in sel)
        wr = sum(D[c]["m2"]["REG_FRAME"] < D[c][ref]["REG_FRAME"] for c in sel)
        lu = sum(D[c]["m2"]["UNREG"] > D[c][ref]["UNREG"] for c in sel)
        lr = sum(D[c]["m2"]["REG_FRAME"] > D[c][ref]["REG_FRAME"] for c in sel)
        need = 3 if len(sel) == 4 else (10 if len(sel) == 12 else None)
        if need is None:
            verdict = "lane rule defined only for 4 or 12 clips (descriptive)"
        elif du <= -0.010 and dr <= -0.010 and wu >= need and wr >= need:
            verdict = "CLEARLY BETTER"
        elif du >= 0.010 and dr >= 0.010 and lu >= need and lr >= need:
            verdict = "CLEARLY WORSE"
        else:
            verdict = "NO CLEAR DIFFERENCE"
        P(f"  LANE RULE vs {rn}: dUNREG {du:+.4f} (better {wu}/{len(sel)}), dREG_FRAME {dr:+.4f} (better {wr}/{len(sel)})"
          f" -> {verdict}")
        if len(sel) == 12:
            prog = dr <= -0.005 and wr >= 9
            P(f"  PROGRAM RULE vs {rn}: REG_FRAME {dr:+.4f}, better on {wr}/12 -> {'MET' if prog else 'NOT MET'}")


summary([c for c in REGIME if c in D], "REGIME 4-clip set")
summary(done if len(done) == 12 else [], "ALL 12 TEST CLIPS")
P("")
P("backend reproduction (CPU vs published GPU, max |d| over 5 variants): " + ", ".join(
    f"{c} origin {D[c]['backend_gate'].get('origin_ll', float('nan')):.1e} deliv "
    f"{D[c]['backend_gate'].get('mstudent2_step800_deliv_ll', float('nan')):.1e}" for c in done if D[c]['backend_gate']))
P("gates G0/G1: " + "; ".join(f"{c} {D[c]['G0']} {D[c].get('G1')}" for c in done))
P("speed (s per 16-frame 576x1024 window on the RTX 4090, official scale-1.0 CFG batch, decode in 8-frame chunks): "
  + ", ".join(f"{c} {D[c].get('sec_per_window', float('nan')):.2f}" for c in done))
P("hole fraction after M2SVid closing+dilation (mean over clip): " + ", ".join(
    f"{c} {D[c].get('hole_frac_closed', float('nan')):.3f}" for c in done))
open(out_txt, "w").write("\n".join(L) + "\n")
json.dump(D, open(out_txt[:-4] + ".json", "w"), indent=1, default=str)
print("\n".join(L))
