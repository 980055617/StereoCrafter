#!/usr/bin/env python
"""ays_20261004/ays5 OPTIONAL registered-GT check: contrasts from outputs/ays_20261004/ays5/registered_v1/<clip>.json
(score_registered_v1.py UNCHANGED).  Descriptive only (gates nothing).  CPU only.  usage: analyze_registered_v1.py OUT.txt
UNREG must reproduce this lane's score_clip_ll.py values (|d| <= 1e-6: the rows JSON holds the 6-dp ROW values).
eval_robustness flagged 0125's REG_FRAME as ILL-POSED (search boundary), so an n=11 line without 0125 is added."""
import itertools, json, math, os, sys
import numpy as np
os.chdir("/home/kawa/master_project/StereoCrafter")
O = "outputs/ays_20261004/ays5/registered_v1"
ROWS = json.load(open("scripts/distill/runs/ays_20261004/ays5/ROWS_registered_v1.json"))
CLIPS = ROWS["clips"]
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
SEED, NBOOT = 20261004, 10000
VARS = ["UNREG", "REG_CLIP", "REG_FRAME", "REG_FRAME_RAW", "BLK_FRAME", "BLK_LOCAL"]
R = {c: json.load(open(f"{O}/{c}.json")) for c in CLIPS if os.path.exists(f"{O}/{c}.json")}
Lines = [f"ays5 OPTIONAL registered-GT check (score_registered_v1.py unchanged; {len(R)}/12 clips scored; descriptive only)"]


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


bad = []
for c, r in R.items():
    for lab, cfg in r["configs"].items():
        if lab not in ROWS["cells"][c]:
            continue
        dv = cfg["lpips_clip"]["UNREG"] - ROWS["cells"][c][lab]["lpips"]
        if abs(dv) > 1e-6:
            bad.append(f"{c} {lab} {dv:+.1e}")
Lines.append(f"G0 (UNREG reproduces this lane's ROW values within 1e-6): {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}")
# cross-check: labels already scored by eval_robustness (outputs/more_20261004/eval_robustness/score_v1/<clip>.json, same
# unchanged scorer, model-independent registration) must give the same per-variant clip LPIPS
EV = "outputs/more_20261004/eval_robustness/score_v1"
devs, ncmp = [], 0
for c, r in R.items():
    if not os.path.exists(f"{EV}/{c}.json"):
        continue
    e = json.load(open(f"{EV}/{c}.json"))
    for lab, cfg in r["configs"].items():
        if lab in e["configs"]:
            for v in VARS:
                devs.append(abs(cfg["lpips_clip"][v] - e["configs"][lab]["lpips_clip"][v])); ncmp += 1
Lines.append(f"cross-check vs eval_robustness score_v1 (shared labels, all 6 variants): {ncmp} values, max |dev| "
             f"{max(devs) if devs else float('nan'):.1e} -> {'PASS' if devs and max(devs) <= 1e-6 else 'CHECK'}")


def contrast(a, b, label, var, drop=()):
    cl = [c for c in CLIPS if c in R and a in R[c]["configs"] and b in R[c]["configs"] and c not in drop]
    if len(cl) < 2:
        return
    d = np.array([R[c]["configs"][a]["lpips_clip"][var] - R[c]["configs"][b]["lpips_clip"][var] for c in cl])
    lo, hi = boot(d)
    Lines.append(f"  {var:13s} n={len(cl):2d} mean {d.mean():+.5f} neg {int((d < 0).sum())}/{len(cl)} boot95 [{lo:+.5f},{hi:+.5f}] "
                 f"p_sign {sign_p(d):.4f} p_flip {flip_p(d):.4f} | " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(cl, d)))


for a, b, label in [("deliv_g100_T5pad", "AYS5pad8_origin_g100", "PRIMARY deliverable T5pad - AYS5 origin (5 evals each)"),
                    ("deliv_g100_T5pad", "AYS8_origin_g100", "3a deliverable T5pad (5) - AYS8 origin @1.00 (8)"),
                    ("AYS5pad8_origin_g100", "origin_ll", "AYS5 origin (5) - deployed origin (16)"),
                    ("deliv_g100_T5pad", "origin_ll", "context deliverable T5pad - deployed origin"),
                    ("mstudent2_step800_deliv_ll", "origin_ll", "anchor: deployed deliverable - deployed origin (eval_robustness REG_FRAME -0.0132, 11/12)")]:
    if not any(a in r["configs"] and b in r["configs"] for r in R.values()):
        continue
    Lines.append(f"{label}   [{a} - {b}]")
    for v in VARS:
        contrast(a, b, label, v)
    contrast(a, b, label, "REG_FRAME", drop=("0125",))
    Lines[-1] = Lines[-1].replace("REG_FRAME    ", "REG_FRAME-0125", 1)
open(OUT, "w").write("\n".join(Lines) + "\n")
print("\n".join(Lines))
