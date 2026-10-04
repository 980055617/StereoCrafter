#!/usr/bin/env python
"""ays_20261004/ays5 (PREREG ADDENDUM 1): the PRIMARY contrast under a registered GT, computed READ-ONLY from the
ays_20261004 "robust" lane's per-clip JSONs outputs/ays_20261004/robust/score_reg_ays5_v1/<clip>.json (written by
scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py, unchanged).  CPU only; descriptive; gates
nothing.  Their UNREG value of every row must reproduce this lane's S2 ROW value within 1e-6, else nothing is quoted.
usage: analyze_robust_reg_primary_v1.py OUT.txt"""
import glob, itertools, json, math, os, sys
import numpy as np
os.chdir("/home/kawa/master_project/StereoCrafter")
L = "scripts/distill/runs/ays_20261004/ays5"
RD = "outputs/ays_20261004/robust/score_reg_ays5_v1"
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
VARS = ["UNREG", "REG_CLIP", "REG_FRAME", "REG_FRAME_RAW", "BLK_FRAME", "BLK_LOCAL"]
SEED, NBOOT = 20261004, 10000
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
S2 = {}
for f in sorted(glob.glob(f"{L}/SCORES_AYS5_S2_*.txt")):
    for ln in open(f):
        if ln.startswith("ROW "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:])
            S2[(d["clip"], d["tag"][5:])] = (float(d["lpips"]), os.path.normpath(d["path"]))
R = {c: json.load(open(f"{RD}/{c}.json")) for c in CLIPS if os.path.exists(f"{RD}/{c}.json")}
Lines = [f"PRIMARY contrast under registered GT -- read-only from {RD} ({len(R)}/12 clips; robust lane, score_registered_v1.py unchanged)"]
bad, ncmp = [], 0
for c, r in R.items():
    for lab, cfg in r["configs"].items():
        if (c, lab) in S2:
            ncmp += 1
            dv = cfg["lpips_clip"]["UNREG"] - S2[(c, lab)][0]
            if abs(dv) > 1e-6 or os.path.normpath(cfg["path"]) != S2[(c, lab)][1]:
                bad.append(f"{c} {lab} d={dv:+.1e} path_same={os.path.normpath(cfg['path']) == S2[(c, lab)][1]}")
ok = not bad and len(R) == 12
Lines.append(f"G0 their UNREG vs this lane's S2 ROW values (same file, |d| <= 1e-6): {ncmp} rows compared, "
             f"{'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}" + ("" if len(R) == 12 else "  [not all 12 clips present -> NOT quoted]"))


def boot(d):
    rng = np.random.default_rng(SEED)
    m = d[rng.integers(0, len(d), size=(NBOOT, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def sign_p(d):
    nz = d[d != 0]; n, k = len(nz), int((nz < 0).sum()); m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0


def flip_p(d):
    S = np.array(list(itertools.product([1, -1], repeat=len(d))), dtype=float)
    return float(np.mean(np.abs((S * d).mean(axis=1)) >= abs(d.mean()) - 1e-15))


def contrast(a, b, var, drop=()):
    cl = [c for c in CLIPS if c in R and c not in drop]
    d = np.array([R[c]["configs"][a]["lpips_clip"][var] - R[c]["configs"][b]["lpips_clip"][var] for c in cl])
    lo, hi = boot(d)
    tag = var + ("-0125" if drop else "")
    Lines.append(f"  {tag:14s} n={len(cl):2d} mean {d.mean():+.5f} neg {int((d < 0).sum())}/{len(cl)} boot95 [{lo:+.5f},{hi:+.5f}] "
                 f"p_sign {sign_p(d):.5f} p_flip {flip_p(d):.5f} | " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(cl, d)))


if ok:
    for a, b, lab in [("deliv_g100_T5pad", "AYS5pad8_origin_g100", "PRIMARY deliverable T5pad - AYS5 origin (5 evals each, same noise)"),
                      ("AYS5pad8_origin_g100", "origin_g100_T5pad", "secondary S-a: AYS5 origin - Karras-T5 origin"),
                      ("AYS5pad8_origin_g100", "origin_ll", "secondary S-b: AYS5 origin (5 evals) - deployed origin (16 evals)")]:
        Lines.append(f"{lab}   [{a} - {b}]")
        for v in VARS:
            contrast(a, b, v)
        contrast(a, b, "REG_FRAME", drop=("0125",))
    Lines.append("note: eval_robustness flagged 0125's REG_FRAME as ILL-POSED (search boundary); the REG_FRAME-0125 line is the n=11 sensitivity.")
open(OUT, "w").write("\n".join(Lines) + "\n")
print("\n".join(Lines))
