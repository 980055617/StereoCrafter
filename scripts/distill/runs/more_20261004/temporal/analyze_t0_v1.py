#!/usr/bin/env python
"""more_20261004 / temporal lane: S1 scorer-equivalence check + diagnostics D1 (seam share) and D2 (decode-pair
boundaries) on the EXISTING origin / deliv8 / BASE / BASEpad renders (PREREG.txt).  CPU only, reads JSON.
usage: analyze_t0_v1.py <t0_baselines.json> <OUT.txt>
"""
import json
import sys

CLIPS = ["0170", "0259", "0042", "0301"]
V1 = "outputs/finalcheck_20261004/validate/v1_temporal/temporal_576_12clip.json"
ROWS = [("origin", "origin_ll"), ("deliv8", "mstudent2_step800_deliv_ll"),
        ("BASE", "deliv_g100_T5nat"), ("BASEpad", "deliv_g100_T5pad")]

src, out = sys.argv[1], sys.argv[2]
J = json.load(open(src))
v1 = json.load(open(V1))
L = []
P = L.append
P(f"T0 BASELINES -- source {src}")
P(f"scorer scripts/distill/runs/more_20261004/temporal/score_temporal_tr_v1.py (STEP=1), clips {' '.join(CLIPS)}")

# ---------------------------------------------------------------- S1 equivalence vs the validate lane JSON
P("")
P(f"=== S1 scorer equivalence vs {V1} (origin_ll, mstudent2_step800_deliv_ll) ===")
nexact, nall, maxd = 0, 0, 0.0
for c in CLIPS:
    for tag in ("origin_ll", "mstudent2_step800_deliv_ll"):
        a, b = J[c][f"{c}_{tag}"], v1[c][f"{c}_{tag}"]
        for k in ("tLP", "warp", "seam", "nonseam"):
            nall += 1
            d = abs(a[k] - b[k])
            maxd = max(maxd, d)
            nexact += (a[k] == b[k])
    for k in ("tLP", "warp", "seam", "nonseam"):
        nall += 1
        d = abs(J[c]["GT"][k] - v1[c]["GT"][k])
        maxd = max(maxd, d)
        nexact += (J[c]["GT"][k] == v1[c]["GT"][k])
s1 = maxd <= 1e-6
P(f"  values compared {nall} (incl. GT rows), exactly equal {nexact}, max |diff| {maxd:.3e} -> S1 {'PASS' if s1 else 'FAIL'}")

# ---------------------------------------------------------------- per-transition consistency
P("")
P("=== per-transition arrays reproduce the headline warp (sum(warp_sum)/sum(warp_cnt) vs warp) ===")
worst = 0.0
for c in CLIPS:
    for _, tag in ROWS:
        r = J[c][f"{c}_{tag}"]
        tr = r["tr"]
        w = sum(tr["warp_sum"]) / sum(tr["warp_cnt"])
        worst = max(worst, abs(w / r["warp"] - 1))
P(f"  worst relative difference {worst:.2e} (float32 per-batch vs float64 per-transition summation order)")


def pooled(r, classes):
    tr = r["tr"]
    s = sum(x for x, cl in zip(tr["warp_sum"], tr["cls"]) if cl in classes)
    n = sum(x for x, cl in zip(tr["warp_cnt"], tr["cls"]) if cl in classes)
    k = sum(1 for cl in tr["cls"] if cl in classes)
    return s, n, k


def decomp(r):
    S_seam, n_seam, k_seam = pooled(r, {"seam"})
    S_b, n_b, k_b = pooled(r, {"bnd"})
    S_i, n_i, k_i = pooled(r, {"in"})
    n_all = n_seam + n_b + n_i
    mean_ns = (S_b + S_i) / (n_b + n_i)
    mean_in = S_i / n_i
    return dict(warp=r["warp"], seam=S_seam / n_seam, bnd=S_b / n_b, inn=mean_in, nonseam=mean_ns,
                k_seam=k_seam, k_b=k_b, k_i=k_i,
                seamflat=mean_ns,
                bflat=(S_seam + S_i + n_b * mean_in) / n_all,
                dR=r["tr"]["dR"], cls=r["tr"]["cls"])


D = {c: {lab: decomp(J[c][f"{c}_{tag}"]) for lab, tag in ROWS} for c in CLIPS}
m4 = lambda lab, key: sum(D[c][lab][key] for c in CLIPS) / len(CLIPS)
o4 = m4("origin", "warp")
gap = lambda lab, key="warp": (m4(lab, key) / o4 - 1) * 100

P("")
P("=== headline per clip (warp = pixel-pooled RAFT warp error, all transitions) ===")
P(f"  {'row':10s} " + " ".join(f"{c:>9s}" for c in CLIPS) + f" {'mean4':>9s} {'gap%':>7s}")
for lab, _ in ROWS:
    P(f"  {lab:10s} " + " ".join(f"{D[c][lab]['warp']:9.5f}" for c in CLIPS) + f" {m4(lab, 'warp'):9.5f} {gap(lab):+7.2f}")
P(f"  GT         " + " ".join(f"{J[c]['GT']['warp']:9.5f}" for c in CLIPS))
F = abs(gap("BASEpad") - gap("BASE"))
P(f"  seed floor F = |gap(BASEpad) - gap(BASE)| = {F:.2f} points  -> R1 requires an improvement > 2F = {2 * F:.2f} points")
P(f"  per-clip warp(BASEpad)/warp(BASE): " + " ".join(f"{c} {D[c]['BASEpad']['warp'] / D[c]['BASE']['warp']:.4f}" for c in CLIPS))

P("")
P("=== D1 seam share: pooled warp at window seams vs the rest (own geometry, overlap 3 -> t = 11k+2) ===")
P(f"  {'row':10s} {'clip':>5s} {'k_seam':>6s} {'seam':>9s} {'nonseam':>9s} {'seam/ns':>8s}")
for lab, _ in ROWS:
    for c in CLIPS:
        d = D[c][lab]
        P(f"  {lab:10s} {c:>5s} {d['k_seam']:6d} {d['seam']:9.5f} {d['nonseam']:9.5f} {d['seam'] / d['nonseam']:8.3f}")
P("")
P(f"  {'row':10s} {'gap%':>7s} {'gap_seamflat%':>14s} {'bound_seam(pts)':>16s}")
for lab, _ in ROWS:
    P(f"  {lab:10s} {gap(lab):+7.2f} {gap(lab, 'seamflat'):+14.2f} {gap(lab) - gap(lab, 'seamflat'):16.2f}")
bs = gap("BASE") - gap("BASE", "seamflat")
P(f"  bound_seam(BASE) = {bs:.2f} points (the most a seam-only fix could remove if within-window transitions are "
  f"unchanged) -> {'< 5: seam-targeted knobs cannot meet W by construction (prediction)' if bs < 5 else '>= 5: W reachable in principle'}")

P("")
P("=== D2 decode-pair boundaries (decode_chunk_size 2; window-LOCAL position; non-seam transitions only) ===")
P(f"  {'row':10s} {'clip':>5s} {'k_bnd':>6s} {'k_in':>6s} {'bnd':>9s} {'within':>9s} {'bnd/in':>8s}")
for lab, _ in ROWS:
    for c in CLIPS:
        d = D[c][lab]
        P(f"  {lab:10s} {c:>5s} {d['k_b']:6d} {d['k_i']:6d} {d['bnd']:9.5f} {d['inn']:9.5f} {d['bnd'] / d['inn']:8.3f}")
P("")
P(f"  {'row':10s} {'gap%':>7s} {'gap_bflat%':>11s} {'bound_dcs(pts)':>15s}")
for lab, _ in ROWS:
    P(f"  {lab:10s} {gap(lab):+7.2f} {gap(lab, 'bflat'):+11.2f} {gap(lab) - gap(lab, 'bflat'):15.2f}")
bd = gap("BASE") - gap("BASE", "bflat")
P(f"  bound_dcs(BASE) = {bd:.2f} points -> dcs14 TRIGGER (>= 2.5): {'YES' if bd >= 2.5 else 'NO'}")

# |dR| by class (raw frame difference, no flow) -- descriptive
P("")
P("=== descriptive: mean |R_t+1 - R_t| by transition class (no flow) ===")
for lab, _ in ROWS:
    row = []
    for c in CLIPS:
        d = D[c][lab]
        by = {}
        for x, cl in zip(d["dR"], d["cls"]):
            by.setdefault(cl, []).append(x)
        mm = {k: sum(v) / len(v) for k, v in by.items()}
        row.append(f"{c} seam {mm['seam']:.4f} bnd {mm['bnd']:.4f} in {mm['in']:.4f}")
    P(f"  {lab:10s} " + " | ".join(row))
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
