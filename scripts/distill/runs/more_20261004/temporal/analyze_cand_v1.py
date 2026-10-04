#!/usr/bin/env python
"""more_20261004 / temporal lane: candidate table + PREREG.txt verdicts.  CPU only.
usage: analyze_cand_v1.py OUT.txt --clips 0170 0259 0042 0301 --cands T5nat_ov5 T5nat_ov7 ...
           --tjson a.json b.json ... --lpips s1.txt s2.txt ... [--ctx origin_g101_s8_dcs14 ...]
Each clip's temporal rows must all come from ONE scorer invocation (one JSON) that also contains origin_ll, so every
gap uses an origin value scored with the same flows.  If a clip appears in several JSONs, the LAST one that contains all
needed rows for that clip is used (named in the provenance block).
"""
import argparse
import json
import math
import os

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--clips", nargs="+", required=True)
ap.add_argument("--cands", nargs="+", required=True)
ap.add_argument("--ctx", nargs="*", default=[])
ap.add_argument("--tjson", nargs="+", required=True)
ap.add_argument("--lpips", nargs="+", required=True)
A = ap.parse_args()
CL = A.clips
REF = [("origin", "origin_ll"), ("deliv8", "mstudent2_step800_deliv_ll"), ("BASE", "deliv_g100_T5nat"),
       ("BASEpad", "deliv_g100_T5pad")]
ROWS = REF + [(c, c) for c in A.cands] + [(c, c) for c in A.ctx]
SPEED_DIRS = {"BASE": "outputs/finalcheck_20261004/speed/clips/{c}_deliv_g100_T5nat",
              "BASEpad": "outputs/finalcheck_20261004/speed/clips/{c}_deliv_g100_T5pad",
              "deliv8": None, "origin": None}

# ------------------------------------------------------------------ temporal rows
T, TSRC = {}, {}
for jf in A.tjson:
    J = json.load(open(jf))
    for c in CL:
        if c not in J:
            continue
        need = [f"{c}_origin_ll"] + [f"{c}_{t}" for lab, t in ROWS if f"{c}_{t}" in J[c]]
        if f"{c}_origin_ll" in J[c]:
            if c not in T:
                T[c] = {}
            for lab, t in ROWS:
                k = f"{c}_{t}"
                if k in J[c]:
                    T[c][lab] = dict(J[c][k], origin_same_json=J[c][f"{c}_origin_ll"]["warp"])
                    TSRC[(c, lab)] = jf
            T[c]["GT"] = J[c]["GT"]
# ------------------------------------------------------------------ LPIPS rows
LP, LSRC = {}, {}
for sf in A.lpips:
    for line in open(sf):
        if not line.startswith("ROW "):
            continue
        kv = dict(x.split("=", 1) for x in line.split()[1:])
        LP[(kv["clip"], kv["tag"])] = dict(lpips=float(kv["lpips"]), sharp=float(kv["sharp"]),
                                           gtSharp=float(kv["gtSharp"]), path=kv["path"])
        LSRC[(kv["clip"], kv["tag"])] = sf


def lp(c, t):
    return LP.get((c, f"{c}_{t}"))


def has_all(lab, t):
    return all(lab in T.get(c, {}) for c in CL) and all(lp(c, t) for c in CL)


L = []
P = L.append
P("TEMPORAL LANE -- candidate table (more_20261004/temporal, GPU 1).  Pre-registration: PREREG.txt in this directory.")
P(f"clips (n={len(CL)}): {' '.join(CL)}   LPIPS: score_clip_ll.py SCORE_STEP=4 (unchanged); temporal: "
  f"score_temporal_tr_v1.py STEP=1")
avail = [(lab, t) for lab, t in ROWS if has_all(lab, t) or (lab in ("deliv8", "BASEpad") and all(lab in T.get(c, {}) for c in CL))]
missing = [lab for lab, t in ROWS if (lab, t) not in avail]
if missing:
    P(f"NOT AVAILABLE on all {len(CL)} clips (not tabulated): {missing}")


def m(lab, key):
    return sum(T[c][lab][key] for c in CL) / len(CL)


o4 = sum(T[c]["origin"]["origin_same_json"] for c in CL) / len(CL)


def gap(lab):
    return (sum(T[c][lab]["warp"] for c in CL) / len(CL) / o4 - 1) * 100


def lpm(t):
    return sum(lp(c, t)["lpips"] for c in CL) / len(CL)


# origin consistency across JSONs (same scorer, recomputed flows) -- must be identical
ovals = {}
for jf in A.tjson:
    J = json.load(open(jf))
    for c in CL:
        if c in J and f"{c}_origin_ll" in J[c]:
            ovals.setdefault(c, set()).add(J[c][f"{c}_origin_ll"]["warp"])
P("origin warp identical across every scorer invocation that contains it: " +
  " ".join(f"{c}:{'yes' if len(v) == 1 else 'NO ' + str(sorted(v))}" for c, v in ovals.items()))

P("")
P("--- PER CLIP: warp error (pixel-pooled RAFT warp, lower = steadier) ---")
P(f"  {'row':22s} " + " ".join(f"{c:>8s}" for c in CL) + f" {'mean':>8s} {'gap%':>7s}  vsBASE per clip")
for lab, t in avail:
    vs = " ".join(f"{T[c][lab]['warp'] / T[c]['BASE']['warp']:.3f}" for c in CL)
    P(f"  {lab:22s} " + " ".join(f"{T[c][lab]['warp']:8.5f}" for c in CL) + f" {m(lab, 'warp'):8.5f} {gap(lab):+7.2f}  {vs}")
P(f"  {'GT':22s} " + " ".join(f"{T[c]['GT']['warp']:8.5f}" for c in CL))

P("")
P("--- PER CLIP: tLP (mean LPIPS between consecutive frames) and tLP/tLP_GT ---")
for lab, t in avail:
    P(f"  {lab:22s} " + " ".join(f"{T[c][lab]['tLP']:.4f}({T[c][lab]['tLP'] / T[c]['GT']['tLP']:.3f})" for c in CL)
      + f"  mean {m(lab, 'tLP'):.4f}")
P("")
P("--- PER CLIP: seam ratio at the render's OWN window seams (seamOwn/nonseamOwn) and at fixed 11k+2 ---")
for lab, t in avail:
    own = [T[c][lab]["seamOwn"] / T[c][lab]["nonseamOwn"] for c in CL]
    fix = [T[c][lab]["seam"] / T[c][lab]["nonseam"] for c in CL]
    P(f"  {lab:22s} own " + " ".join(f"{x:.3f}" for x in own) + f"  mean {sum(own) / len(own):.3f}   |  11k+2 "
      + " ".join(f"{x:.3f}" for x in fix) + f"  mean {sum(fix) / len(fix):.3f}")

P("")
P("--- decomposition of the pooled warp by transition class (own geometry): seam / decode-boundary / within-pair ---")
for lab, t in avail:
    parts = []
    for c in CL:
        tr = T[c][lab]["tr"]
        agg = {}
        for s, n, cl in zip(tr["warp_sum"], tr["warp_cnt"], tr["cls"]):
            a = agg.setdefault(cl, [0.0, 0.0, 0])
            a[0] += s; a[1] += n; a[2] += 1
        parts.append(f"{c} " + " ".join(f"{k}{agg[k][2]}:{agg[k][0] / agg[k][1]:.5f}" for k in ("seam", "bnd", "in")
                                         if k in agg))
    P(f"  {lab:22s} " + " | ".join(parts))

P("")
P("--- same positions for every overlap-3 row: warp at the decode_chunk_size-2 pair BOUNDARIES vs WITHIN pairs (seams excluded)")
P("    (classification of BASE's geometry applied to every overlap-3 row, so decode-14 rows are read at the positions")
P("     where decode-2 renders change decode chunk; ratio bnd/in = 1 means no pair-boundary excess) ---")
for lab, t in avail:
    if T[CL[0]][lab].get("geo", {"ov": 3}).get("ov", 3) != 3:
        continue
    parts, pooled = [], [0.0, 0.0, 0.0, 0.0]
    for c in CL:
        tr = T[c][lab]["tr"]
        cls2 = T[c]["BASE"]["tr"]["cls"]
        assert len(cls2) == len(tr["warp_sum"])
        agg = {}
        for s_, n_, cl in zip(tr["warp_sum"], tr["warp_cnt"], cls2):
            a = agg.setdefault(cl, [0.0, 0.0])
            a[0] += s_; a[1] += n_
        b, i_ = agg["bnd"][0] / agg["bnd"][1], agg["in"][0] / agg["in"][1]
        parts.append(f"{c} bnd {b:.5f} in {i_:.5f} ({b / i_:.3f})")
    P(f"  {lab:22s} " + " | ".join(parts))

import sys as _sys
_sys.path.insert(0, os.path.join(REPO, "scripts/distill/runs/more_20261004/temporal"))
from tlib import window_schedule as _ws  # noqa: E402
P("")
P("--- window-position profile (overlap-3 rows): pooled warp of transition p -> p+1 at window-LOCAL position p, over")
P("    full-length windows k >= 1 (positions 3..12; 13 -> next window = seam), all clips pooled by valid pixels.")
P("    decode_chunk_size 2 puts chunk boundaries at odd p (3, 5, 7, 9, 11) ---")
P(f"  {'row':22s} " + " ".join(f"{'p' + str(q):>8s}" for q in range(3, 13)) + "   odd/even")
for lab, t in avail:
    if T[CL[0]][lab].get("geo", {"ov": 3}).get("ov", 3) != 3:
        continue
    acc = {q: [0.0, 0.0] for q in range(3, 13)}
    for c in CL:
        r = T[c][lab]
        W = _ws(r["Nrender"], 14, 3)
        tr = r["tr"]
        nt = len(tr["warp_sum"])
        for k, w in enumerate(W):
            if k == 0 or w["nf"] != 14:
                continue
            for q in range(3, 13):
                tt = w["cur_i"] + q
                if tt < nt:
                    acc[q][0] += tr["warp_sum"][tt]
                    acc[q][1] += tr["warp_cnt"][tt]
    prof = {q: acc[q][0] / acc[q][1] for q in acc}
    odd = sum(acc[q][0] for q in acc if q % 2) / sum(acc[q][1] for q in acc if q % 2)
    even = sum(acc[q][0] for q in acc if not q % 2) / sum(acc[q][1] for q in acc if not q % 2)
    P(f"  {lab:22s} " + " ".join(f"{prof[q]:8.5f}" for q in range(3, 13)) + f"   {odd / even:.3f}")

P("")
P("--- PER CLIP: LPIPS vs GT (lower = better) and sharpness ---")
P(f"  {'row':22s} " + " ".join(f"{c:>8s}" for c in CL) + f" {'mean':>8s} {'dBASE':>8s} {'d origin':>8s}  sharp/BASE per clip (mean)")
for lab, t in avail:
    if not all(lp(c, t) for c in CL):
        continue
    sr = [lp(c, t)["sharp"] / lp(c, "deliv_g100_T5nat")["sharp"] for c in CL]
    P(f"  {lab:22s} " + " ".join(f"{lp(c, t)['lpips']:8.4f}" for c in CL) + f" {lpm(t):8.4f} {lpm(t) - lpm('deliv_g100_T5nat'):+8.4f} "
      f"{lpm(t) - lpm('origin_ll'):+8.4f}  " + " ".join(f"{x:.3f}" for x in sr) + f" ({sum(sr) / len(sr):.4f})")

# ------------------------------------------------------------------ verdicts
F = abs(gap("BASEpad") - gap("BASE")) if all("BASEpad" in T[c] for c in CL) else float("nan")
P("")
P("=== PRE-REGISTERED VERDICTS (PREREG.txt) ===")
P(f"  origin mean4 warp {o4:.5f};  BASE gap {gap('BASE'):+.2f}%;  BASEpad gap {gap('BASEpad'):+.2f}%;  seed floor F = {F:.2f} "
  f"points (R1 needs an improvement > {2 * F:.2f});  deliv8 gap {gap('deliv8'):+.2f}%")
for lab, t in avail:
    if lab not in A.cands:
        continue
    g = gap(lab)
    dL = lpm(t) - lpm("deliv_g100_T5nat")
    W = g <= gap("BASE") - 5.0
    Lok = dL <= 0.002
    imp = gap("BASE") - g
    better = [c for c in CL if T[c][lab]["warp"] < T[c]["BASE"]["warp"]]
    sr = sum(lp(c, t)["sharp"] / lp(c, "deliv_g100_T5nat")["sharp"] for c in CL) / len(CL)
    if not (W and Lok):
        v = "FAIL (" + ", ".join(x for x, ok in (("W", W), ("L", Lok)) if not ok) + " violated)"
    else:
        quals = []
        if not imp > 2 * F:
            quals.append("INCONCLUSIVE (inside the seed floor)")
        if len(better) < 3:
            quals.append(f"FLAGGED: carried by {' '.join(better)}")
        if sr < 0.97:
            quals.append("FLAGGED: bought by sharpness loss")
        v = "PASS" if not quals else "; ".join(quals)
    P(f"  {lab:22s} gap {g:+7.2f}% (change vs BASE {-imp:+6.2f} pts; W needs <= {gap('BASE') - 5:+.2f}) "
      f"LPIPS d {dL:+.4f} (L <= +0.002)  warp better than BASE on {len(better)}/{len(CL)} {better}  "
      f"sharp/BASE {sr:.4f}  ->  {v}")

# ------------------------------------------------------------------ context: decode setting matched for origin
OD = "origin_g101_s8_dcs14"
if OD in A.ctx and all(OD in T.get(c, {}) for c in CL):
    P("")
    P("=== CONTEXT (not gating): the same decode setting applied to origin (8 steps, guidance 1.01, decode_chunk_size 14) ===")
    od4 = m(OD, "warp")
    P(f"  origin dcs14 vs deployed origin_ll: warp mean4 {od4:.5f} vs {o4:.5f} ({(od4 / o4 - 1) * 100:+.2f}%), per clip "
      + " ".join(f"{c} {T[c][OD]['warp'] / T[c]['origin']['warp']:.3f}" for c in CL))
    if all(lp(c, OD) for c in CL):
        P(f"  origin dcs14 LPIPS mean4 {lpm(OD):.4f} vs origin_ll {lpm('origin_ll'):.4f} (d {lpm(OD) - lpm('origin_ll'):+.4f})")
    for lab in [x for x in A.cands if "dcs" in x]:
        if all(lab in T.get(c, {}) for c in CL):
            g2 = (m(lab, "warp") / od4 - 1) * 100
            P(f"  {lab}: gap vs origin at the SAME decode setting = {g2:+.2f}% (vs deployed origin {gap(lab):+.2f}%; "
              f"BASE vs deployed origin {gap('BASE'):+.2f}%)")
            if all(lp(c, OD) for c in CL):
                P(f"  {lab}: LPIPS vs origin at the same decode setting d {lpm(lab) - lpm(OD):+.4f} "
                  f"(BASE vs deployed origin d {lpm('deliv_g100_T5nat') - lpm('origin_ll'):+.4f})")

# ------------------------------------------------------------------ cost
P("")
P("=== COST (speed_log.json of each render; UNet s = CUDA-event UNet time; windows from the schedule) ===")
for lab, t in avail:
    row = []
    for c in CL:
        if lab in ("BASE", "BASEpad"):
            d = SPEED_DIRS[lab].format(c=c)
        elif lab in ("origin", "deliv8"):
            d = None
        else:
            d = f"outputs/more_20261004/temporal/clips/{c}_{t}"
        if d and os.path.exists(os.path.join(d, "speed_log.json")):
            S = json.load(open(os.path.join(d, "speed_log.json")))
            row.append(f"{c} win {S['n_windows']:2d} unet {S['unet_ms_sum'] / 1000:6.1f}s calls {S['unet_calls_total']} "
                       f"proc {S['hook_total_s']:6.1f}s")
    if row:
        P(f"  {lab:22s} " + " | ".join(row))

P("")
P("=== PROVENANCE ===")
for lab, t in avail:
    P(f"  {lab:22s} temporal: " + ", ".join(sorted(set(TSRC.get((c, lab), '?') for c in CL)))
      + "  lpips: " + ", ".join(sorted(set(LSRC.get((c, f'{c}_{t}'), '?') for c in CL))))
open(A.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
