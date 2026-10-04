#!/usr/bin/env python
"""judge J3c: paired AYS contrasts over every clip scored so far (SCORES_J3c_AYS_*.txt ROW lines; CPU).
usage: analyze_ays_v1.py OUT.txt"""
import glob, itertools, math, os, sys
import numpy as np
os.chdir("/home/kawa/master_project/StereoCrafter")
J = "scripts/distill/runs/more_20261004/judge"
V, S, SRC = {}, {}, {}
for f in sorted(glob.glob(f"{J}/SCORES_J3c_AYS_*.txt")):
    for ln in open(f):
        if not ln.startswith("ROW "): continue
        d = dict(kv.split("=", 1) for kv in ln.split()[1:]); c = d["clip"]; t = d["tag"][len(c) + 1:]
        if (c, t) in V: assert abs(V[(c, t)] - float(d["lpips"])) < 1e-9, (c, t, f)
        V[(c, t)] = float(d["lpips"]); S[(c, t)] = float(d["sharp"]); SRC.setdefault(c, set()).add(os.path.basename(f))
ORDER = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
# context rows not re-scored in this lane (0301 smoke had no mstudent2/s25 row): taken from the published cells that the
# eval_robustness lane re-scored with the same unchanged scorer (G0: 72/72 within 4.9e-07); labelled in the sources line.
import json
PUB = json.load(open("scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json"))
for c in list(SRC):
    for t in ("mstudent2_step800_deliv_ll", "s25_ll"):
        if (c, t) not in V and t in PUB["cells"].get(c, {}):
            V[(c, t)] = float(PUB["cells"][c][t]["lpips"]); S[(c, t)] = float(PUB["cells"][c][t]["sharp"])
            SRC[c].add(f"{t} from eval_robustness/PUBLISHED_ROWS.json")
def exact_sign_p(d):
    n = int((d != 0).sum()); k = int((d < 0).sum()); m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n)
def signflip_p(d):
    obs = abs(d.mean()); return float(np.mean([abs((d * np.array(s)).mean()) >= obs - 1e-15 for s in itertools.product([1, -1], repeat=len(d))]))
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}
L = ["judge J3c -- Align-Your-Steps schedule on ORIGIN, paired contrasts (score_clip_ll.py SCORE_STEP=4, FFV1 renders)",
     "sources: " + "; ".join(f"{c}: {sorted(s)}" for c, s in sorted(SRC.items())), ""]
def contrast(a, b, label, rule=None):
    cl = [c for c in ORDER if (c, a) in V and (c, b) in V]
    if not cl: return
    d = np.array([V[(c, a)] - V[(c, b)] for c in cl]); n = len(d)
    ci = (d.mean() - T975[n - 1] * d.std(ddof=1) / math.sqrt(n), d.mean() + T975[n - 1] * d.std(ddof=1) / math.sqrt(n)) if n > 1 else (float("nan"),) * 2
    L.append(f"{label}   [{a} - {b}]  n={n}")
    L.append("  per clip: " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(cl, d)))
    L.append(f"  mean {d.mean():+.5f}  negative on {int((d < 0).sum())}/{n}  worst {cl[int(d.argmax())]} {d.max():+.4f}  t95 [{ci[0]:+.4f},{ci[1]:+.4f}]"
             f"  p_sign {exact_sign_p(d):.4f}  p_signflip {signflip_p(d):.4f}")
    if rule:
        ok = d.mean() <= -0.002 and d.max() <= 0.005
        L.append(f"  RULE ({rule}): mean <= -0.002 and no clip > +0.005 -> {'PASS (free gain)' if ok else 'FAIL'}")
    sh = [S[(c, a)] / S[(c, b)] for c in cl]; L.append(f"  sharpness ratio a/b mean {np.mean(sh):.3f} (per clip " + " ".join(f"{x:.3f}" for x in sh) + ")")
    L.append(f"  means: {a} {np.mean([V[(c, a)] for c in cl]):.4f}  {b} {np.mean([V[(c, b)] for c in cl]):.4f}")
contrast("AYS8_origin_g101", "origin_ll", "AYS8 origin @1.01 vs deployed origin (Karras-8 @1.01): same cost, same noise", rule="pre-registered 4-/12-clip rule")
contrast("AYS5pad8_origin_g100", "origin_g100_T5pad", "AYS5 origin @1.00 pad8 vs origin T5 @1.00 pad8: same cost, same noise", rule="pre-registered 4-/12-clip rule")
contrast("AYS5pad8_origin_g100", "origin_ll", "AYS5 origin @1.00 pad8 vs deployed origin (5x1 vs 8x2 evals)")
L.append(""); L.append("--- THESIS-DECIDING CONTRASTS (deliverable vs a schedule-optimised origin at equal cost) ---")
contrast("mstudent2_step800_deliv_ll", "AYS8_origin_g101", "deliverable 8x2@1.01 (published headline config) vs AYS8 origin @1.01: same 16 evals/window, same noise")
contrast("deliv_g100_T5pad", "AYS5pad8_origin_g100", "deliverable T5@1.00 pad8 vs AYS5 origin @1.00 pad8: same 5 evals/window, same noise")
contrast("deliv_g100_T5pad", "AYS8_origin_g101", "deliverable T5@1.00 pad8 (5 evals) vs AYS8 origin @1.01 (16 evals)")
L.append(""); L.append("--- context ---")
contrast("s25_ll", "AYS8_origin_g101", "origin s25 (teacher, 50 evals) vs AYS8 origin (16 evals)")
contrast("mstudent2_step800_deliv_ll", "origin_ll", "deliverable 8x2@1.01 vs deployed origin (published headline, same clips)")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
