#!/usr/bin/env python
"""judge J3b + J4a (CPU, read-only): paired per-clip LPIPS contrasts from the finalcheck speed lane's ROW lines
(score_clip_ll.py, SCORE_STEP=4, FFV1) and the eval_robustness lane's PUBLISHED_ROWS.json (for s25_ll).
usage: analyze_matched_v1.py OUT.txt"""
import re, sys, os, json, itertools, math
import numpy as np
os.chdir("/home/kawa/master_project/StereoCrafter")
FC = "scripts/distill/runs/finalcheck_20261004/speed"
files = [f"{FC}/{f}" for f in ["SCORES_STEP1.txt", "SCORES_STEP2_EXT12.txt", "SCORES_STEP2_4clip.txt",
                                "SCORES_STEP2_T5G100_4clip.txt", "SCORES_POSTHOC_T5G100NAT.txt"]]
CL = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
V = {}; SRC = {}
for f in files:
    for ln in open(f):
        if not ln.startswith("ROW "): continue
        d = dict(kv.split("=", 1) for kv in ln.split()[1:])
        clip = d["clip"]; tag = d["tag"][len(clip) + 1:]
        val = float(d["lpips"])
        if (clip, tag) in V:
            assert abs(V[(clip, tag)] - val) < 1e-9, (clip, tag, V[(clip, tag)], val, f)
        V[(clip, tag)] = val; SRC.setdefault(tag, set()).add(os.path.basename(f))
pub = json.load(open("scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json"))
# PUBLISHED_ROWS.json: cells[clip][label]['lpips']
def find_s25():
    return {c: float(pub['cells'][c]['s25_ll']['lpips']) for c in pub['cells'] if 's25_ll' in pub['cells'][c]}
s25 = find_s25()
for c in CL:
    if c in s25: V[(c, "s25_ll")] = s25[c]
SRC["s25_ll"] = {"eval_robustness/PUBLISHED_ROWS.json"}
def vec(tag): return np.array([V[(c, tag)] for c in CL])
def exact_sign_p(d):
    n = int((d != 0).sum()); k = int((d < 0).sum()); m = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, m + 1)) / 2 ** n * 2
    return min(1.0, p)
def signflip_p(d):
    obs = abs(d.mean()); cnt = 0; tot = 0
    for s in itertools.product([1, -1], repeat=len(d)):
        tot += 1; cnt += abs((d * np.array(s)).mean()) >= obs - 1e-15
    return cnt / tot
def tci(d):
    from math import sqrt
    t975 = {11: 2.200985}[len(d) - 1]
    se = d.std(ddof=1) / sqrt(len(d)); return d.mean() - t975 * se, d.mean() + t975 * se
L = []
def contrast(a, b, label, note=""):
    d = vec(a) - vec(b)
    lo, hi = tci(d)
    L.append(f"{label}\n  {a} - {b}   [{note}]")
    L.append("  per clip: " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(CL, d)))
    L.append(f"  mean {d.mean():+.5f}  improved {int((d<0).sum())}/12  worst {CL[int(d.argmax())]} {d.max():+.4f}  best {CL[int(d.argmin())]} {d.min():+.4f}"
             f"  t95 [{lo:+.4f},{hi:+.4f}]  p_sign {exact_sign_p(d):.5f}  p_signflip {signflip_p(d):.5f}")
    L.append(f"  means: {a} {vec(a).mean():.4f}  {b} {vec(b).mean():.4f}")
    return d
L.append("judge J3b/J4a -- paired 12-clip contrasts recomputed from ROW lines (score_clip_ll.py SCORE_STEP=4, FFV1 renders)")
L.append("sources: " + "; ".join(f"{t}: {sorted(s)}" for t, s in sorted(SRC.items())))
L.append("")
L.append("=== J3b: origin's own sampler levers (literature lane quotes T6 +0.00018, T5 +0.00140, T5@1.00 +0.00215) ===")
contrast("origin_g101_T6pad", "origin_ll", "origin T6@1.01 (padded) vs deployed origin", "paired: pad 8")
contrast("origin_g101_T5pad", "origin_ll", "origin T5@1.01 (padded) vs deployed origin", "paired: pad 8")
contrast("origin_g100_T5pad", "origin_ll", "origin T5@1.00 (padded) vs deployed origin", "paired: pad 8")
contrast("origin_g100_s8", "origin_ll", "origin 8-step@1.00 vs deployed origin", "paired by construction")
L.append("")
L.append("=== J4a: MATCHED SAMPLER (same schedule, same guidance, same noise) deliverable vs origin ===")
contrast("deliv_g100_T5pad", "origin_g100_T5pad", "T5@1.00, both RNG-padded to 8 (PRIMARY matched comparison)", "paired: identical per-window noise")
contrast("deliv_g100_s8", "origin_g100_s8", "8 steps @1.00, both", "paired")
contrast("mstudent2_step800_deliv_ll", "origin_ll", "8x2 @1.01 both (= the published headline)", "paired")
contrast("deliv_g101_T5pad", "origin_g101_T5pad", "T5@1.01, both padded", "paired")
L.append("")
L.append("=== context: shipped config vs deployed origin, and vs the teacher ===")
contrast("deliv_g100_T5nat", "origin_ll", "deliverable T5@1.00 UNPADDED (what ships) vs deployed origin", "window 0 paired only")
contrast("deliv_g100_T5nat", "origin_g100_T5pad", "deliverable T5@1.00 UNPADDED vs origin T5@1.00 padded", "window 0 paired only; descriptive")
if all((c, "s25_ll") in V for c in CL):
    contrast("s25_ll", "origin_ll", "origin s25 (teacher) vs deployed origin", "s25 shares window-0 noise")
    contrast("mstudent2_step800_deliv_ll", "s25_ll", "deliverable 8x2@1.01 vs s25", "")
txt = "\n".join(L) + "\n"
open(sys.argv[1], "w").write(txt); print(txt)
