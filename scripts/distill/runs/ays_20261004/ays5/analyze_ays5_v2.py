#!/usr/bin/env python
# [ays5 v2 = analyze_ays5_v1.py with ONLY: a score row whose tag ends in _r1 (a PREREG retry dir) is filed under its label
#  without the suffix, and the real path is printed (PREREG ADDENDUM 2).]
"""ays_20261004/ays5: reproduction gates + paired contrasts from this lane's SCORES_AYS5_<TAG>_<clip>.txt ROW lines
(score_clip_ll.py UNCHANGED, SCORE_STEP=4, FFV1 renders).  CPU only, read-only.  Definitions: PREREG.txt (same dir).
usage: analyze_ays5_v1.py OUT.txt [OUT.json]       (refuses to overwrite either file)"""
import glob, itertools, json, math, os, sys
import numpy as np
os.chdir("/home/kawa/master_project/StereoCrafter")
L = "scripts/distill/runs/ays_20261004/ays5"
J = "scripts/distill/runs/more_20261004/judge"
F = "scripts/distill/runs/finalcheck_20261004/speed"
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
REGIME = ["0052", "0147", "0204", "0301"]
SEED, NBOOT, TOL = 20261004, 10000, 2e-6
OUT_TXT = sys.argv[1]
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else OUT_TXT.rsplit(".", 1)[0] + ".json"
for p in (OUT_TXT, OUT_JSON):
    assert not os.path.exists(p), f"refusing to overwrite {p}"


def rows(path):
    for ln in open(path):
        if ln.startswith("ROW "):
            yield dict(kv.split("=", 1) for kv in ln.split()[1:])


def load_stage(tag):
    """this lane's scores of one stage -> V[(clip, label)] (label = tag minus the clip prefix)."""
    V = {}
    for f in sorted(glob.glob(f"{L}/SCORES_AYS5_{tag}_*.txt")):
        txt = open(f).read()
        if "SCORE_DONE rc=0" not in txt:
            print(f"INCOMPLETE score file ignored: {f}")
            continue
        for d in rows(f):
            c = d["clip"]
            lab = d["tag"][len(c) + 1:]
            if lab.endswith("_r1"):                       # [v2: a PREREG retry dir counts as its label; real path kept]
                lab = lab[:-3]
                RETRY.append(f"{tag} {c} {lab}: read from {d['path']} (retry _r1, PREREG ADDENDUM 2)")
            V[(c, lab)] = dict(lpips=float(d["lpips"]), sharp=float(d["sharp"]), dy=int(d["dy"]),
                                                  dx=int(d["dx"]), n=int(d["n"]), path=os.path.normpath(d["path"]),
                                                  src=os.path.relpath(f))
    return V


# ---------------------------------------------------------------- reference ROW values (other lanes' score files)
REF_FILES = [f"{F}/SCORES_STEP1.txt", f"{F}/SCORES_STEP2_4clip.txt", f"{F}/SCORES_STEP2_EXT12.txt",
             f"{F}/SCORES_STEP2_T5G100_4clip.txt", f"{F}/SCORES_POSTHOC_T5G100NAT.txt"] + sorted(glob.glob(f"{J}/SCORES_J3c_AYS_*.txt"))
REF = {}
for f in REF_FILES:
    for d in rows(f):
        REF.setdefault(os.path.normpath(d["path"]), []).append(
            dict(lpips=float(d["lpips"]), dy=int(d["dy"]), dx=int(d["dx"]), n=int(d["n"]), src=os.path.relpath(f)))

RETRY = []
S2, S3 = load_stage("S2"), load_stage("S3")
L_OUT, JS = [], dict(seed=SEED, nboot=NBOOT, tol=TOL, gates={}, contrasts={})


def gate(V, stage, extra=None):
    """per clip: every scored row whose exact file was scored before must reproduce |dLPIPS| <= TOL, same (dy,dx), n."""
    ok_clips = {}
    for c in CLIPS:
        cells = {k[1]: v for k, v in V.items() if k[0] == c}
        if not cells:
            continue
        msgs, ok = [], True
        for lab, v in cells.items():
            refs = list(REF.get(v["path"], []))
            if extra:
                refs += [dict(lpips=e["lpips"], dy=e["dy"], dx=e["dx"], n=e["n"], src=e["src"])
                         for e in extra.values() if e["path"] == v["path"]]
            for r in refs:
                dv = v["lpips"] - r["lpips"]
                good = abs(dv) <= TOL and (v["dy"], v["dx"], v["n"]) == (r["dy"], r["dx"], r["n"])
                ok &= good
                msgs.append(f"{lab} vs {r['src']}: d={dv:+.1e} off=({v['dy']},{v['dx']}) n={v['n']} {'ok' if good else 'MISMATCH'}")
        nref = len(msgs)
        ok_clips[c] = ok
        JS["gates"][f"{stage}_{c}"] = dict(ok=ok, checks=msgs)
        L_OUT.append(f"  {stage} {c}: {'PASS' if ok else 'FAIL'} ({nref} reference comparisons)"
                     + ("" if ok else "  " + " | ".join(m for m in msgs if "MISMATCH" in m)))
    return ok_clips


def exact_sign_p(d):
    nz = d[d != 0]
    n, k = len(nz), int((nz < 0).sum())
    m = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n) if n else 1.0


def signflip_p(d):
    obs = abs(d.mean())
    S = np.array(list(itertools.product([1, -1], repeat=len(d))), dtype=float)
    return float(np.mean(np.abs((S * d).mean(axis=1)) >= obs - 1e-15))


def boot_ci(d):
    rng = np.random.default_rng(SEED)                      # fresh generator per contrast -> order-independent
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def contrast(V, a, b, label, clips, valid=None, key=None):
    cl = [c for c in clips if (c, a) in V and (c, b) in V and (valid is None or valid.get(c, False))]
    if not cl:
        L_OUT.append(f"{label}: no clips")
        return None
    d = np.array([V[(c, a)]["lpips"] - V[(c, b)]["lpips"] for c in cl])
    n = len(d)
    t = (d.mean() - T975[n - 1] * d.std(ddof=1) / math.sqrt(n), d.mean() + T975[n - 1] * d.std(ddof=1) / math.sqrt(n)) if n > 1 else (float("nan"),) * 2
    lo, hi = boot_ci(d) if n > 1 else (float("nan"),) * 2
    res = dict(a=a, b=b, clips=cl, delta=[float(x) for x in d], mean=float(d.mean()), n=n, neg=int((d < 0).sum()),
               worst_clip=cl[int(d.argmax())], worst=float(d.max()), boot95=[lo, hi], t95=list(map(float, t)),
               p_sign=exact_sign_p(d), p_signflip=signflip_p(d),
               mean_a=float(np.mean([V[(c, a)]["lpips"] for c in cl])), mean_b=float(np.mean([V[(c, b)]["lpips"] for c in cl])),
               sharp_ratio=[V[(c, a)]["sharp"] / V[(c, b)]["sharp"] for c in cl],
               src={c: sorted({V[(c, a)]["src"], V[(c, b)]["src"]}) for c in cl},
               path_a={c: V[(c, a)]["path"] for c in cl}, path_b={c: V[(c, b)]["path"] for c in cl})
    L_OUT.append(f"{label}   [{a} - {b}]  n={n}")
    L_OUT.append("  per clip: " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(cl, d)))
    L_OUT.append(f"  mean {d.mean():+.5f}  negative on {res['neg']}/{n}  worst {res['worst_clip']} {d.max():+.4f}  "
                 f"boot95 [{lo:+.5f},{hi:+.5f}]  t95 [{t[0]:+.5f},{t[1]:+.5f}]  p_sign {res['p_sign']:.5f}  p_signflip {res['p_signflip']:.5f}")
    L_OUT.append(f"  means: {a} {res['mean_a']:.4f}  {b} {res['mean_b']:.4f}   sharpness ratio a/b mean {np.mean(res['sharp_ratio']):.3f}")
    JS["contrasts"][key or f"{a}-{b}"] = res
    return res


L_OUT += ["ays_20261004 / ays5 -- deliverable T5@1.00 vs origin with the Align-Your-Steps 5-evaluation schedule (AYS5)",
          "scorer scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py UNCHANGED, SCORE_STEP=4, FFV1 renders; "
          f"bootstrap: percentile, {NBOOT} resamples of clips, numpy default_rng({SEED}) fresh per contrast; "
          "sign test exact two-sided (ties dropped); sign-flip exact over all 2^n sign assignments", ""]
MODE = open(f"{L}/MODE_S1.txt").read().split()[0] if os.path.exists(f"{L}/MODE_S1.txt") else "?"
L_OUT.append(f"AYS5-origin render mode (C0 control outcome): {MODE}")
JS["mode"] = MODE

if S2:
    L_OUT.append("REPRODUCTION GATES stage S2 (|dLPIPS| <= 2e-6 on the 6-dp ROW values, same offset and n, against every earlier score of the same file)")
    g2 = gate(S2, "S2")
    L_OUT.append("")
    L_OUT.append("=== STEP 2 PRIMARY (pre-registered): deliverable T5@1.00 pad8 vs AYS5 origin @1.00 pad8 -- same 5 evaluations/window, same per-window noise ===")
    P = contrast(S2, "deliv_g100_T5pad", "AYS5pad8_origin_g100", "PRIMARY", CLIPS, valid=g2, key="PRIMARY")
    if P:
        void = [c for c in CLIPS if not g2.get(c, False)]
        holds = P["neg"] >= 10 and P["boot95"][1] < 0
        L_OUT.append(f"  VERDICT: negative on {P['neg']}/12 (need >= 10; VOID clips count as not negative: {void or 'none'}), "
                     f"bootstrap 95% CI upper {P['boot95'][1]:+.5f} (need < 0)  ->  CLAIM {'HOLDS' if holds else 'DOES NOT HOLD'}")
        JS["verdict_step2"] = dict(holds=bool(holds), neg=P["neg"], void=void, boot_hi=P["boot95"][1])
    L_OUT.append("")
    L_OUT.append("--- step 2 secondary (descriptive; the judge's 12-clip rule is applied to AYS5 vs Karras-T5 on origin) ---")
    R = contrast(S2, "AYS5pad8_origin_g100", "origin_g100_T5pad", "AYS5 vs Karras-T5, both origin @1.00 pad8 (same cost, same noise)", CLIPS, valid=g2)
    if R:
        ok = R["mean"] <= -0.002 and R["worst"] <= 0.005
        L_OUT.append(f"  judge 12-clip rule (mean12 <= -0.002 and no clip > +0.005): {'PASS (free gain)' if ok else 'FAIL'}"
                     + ("" if R["n"] == 12 else f"  [n={R['n']}, rule defined for 12]"))
    contrast(S2, "AYS5pad8_origin_g100", "origin_ll", "AYS5 origin @1.00 pad8 (5 evals) vs deployed origin 8x2@1.01 (16 evals)", CLIPS, valid=g2)
    contrast(S2, "deliv_g100_T5nat", "AYS5pad8_origin_g100", "deliverable T5@1.00 UNPADDED (what ships) vs AYS5 origin pad8 (window 0 paired only)", CLIPS, valid=g2)
    contrast(S2, "deliv_g100_T5pad", "origin_g100_T5pad", "consistency: deliverable T5pad vs origin Karras-T5pad (judge J4a: -0.01357, 12/12)", CLIPS, valid=g2)
    contrast(S2, "deliv_g100_T5pad", "origin_ll", "context: deliverable T5pad (5 evals) vs deployed origin (16 evals)", CLIPS, valid=g2)

if S3:
    L_OUT.append("")
    L_OUT += [f"note: {r}" for r in RETRY]
    L_OUT.append("REPRODUCTION GATES stage S3 (also against this lane's S2 scores of the same files)")
    g3 = gate(S3, "S3", extra={k: v for k, v in S2.items()})
    L_OUT.append("")
    L_OUT.append("=== STEP 3a: deliverable T5@1.00 pad8 (5 evals) vs AYS8 origin @1.00 (8 evals, batch 1) -- same per-window noise ===")
    A = contrast(S3, "deliv_g100_T5pad", "AYS8_origin_g100", "3a", CLIPS, valid=g3, key="3a")
    if A:
        alg = A["mean"] <= 0
        ni = A["boot95"][1] <= 0.0007
        L_OUT.append(f"  3a (pre-registered): mean{A['n']} {A['mean']:+.5f} {'<=' if alg else '>'} 0 -> deliverable at 5 evaluations is "
                     f"{'AT LEAST AS GOOD as' if alg else 'WORSE than'} AYS8 origin at 8 -> equal-quality evaluation ratio "
                     f"{'>= 8/5 = 1.6' if alg else 'between 1 (AYS5 origin, beaten) and 1.6'}")
        L_OUT.append(f"  3a secondary non-inferiority (bootstrap upper <= +0.0007, re-seed proxy median): {'MET' if ni else 'NOT MET'} (upper {A['boot95'][1]:+.5f})")
        JS["verdict_3a"] = dict(at_least_as_good=bool(alg), noninferior_0007=bool(ni))
    contrast(S3, "AYS8_origin_g100", "AYS8_origin_g101", "context: AYS8 origin @1.00 (8 evals) vs AYS8 origin @1.01 (16 evals)", CLIPS, valid=g3)
    V23 = dict(S2)
    V23.update({k: v for k, v in S3.items() if k[1] == "AYS8_origin_g100"})
    g23 = {c: g2.get(c, False) and g3.get(c, False) for c in CLIPS}
    contrast(V23, "AYS8_origin_g100", "AYS5pad8_origin_g100", "context: AYS8 origin @1.00 (8 evals) vs AYS5 origin @1.00 (5 evals) [S3 vs S2 scores]", CLIPS, valid=g23)
    contrast(V23, "AYS8_origin_g100", "origin_ll", "context: AYS8 origin @1.00 (8 evals) vs deployed origin (16 evals) [S3 vs S2 scores]", CLIPS, valid=g23)
    L_OUT.append("")
    L_OUT.append("=== STEP 3b: deliverable AYS5@1.00 pad8 vs deliverable T5@1.00 pad8 (4 regime clips) -- do the two gains stack? ===")
    Bb = contrast(S3, "AYS5pad8_deliv_g100", "deliv_g100_T5pad", "3b", REGIME, valid=g3, key="3b")
    if Bb:
        st = Bb["mean"] <= -0.002 and Bb["worst"] <= 0.005
        L_OUT.append(f"  3b (judge's 4-clip rule, mean4 <= -0.002 and no clip > +0.005): {'STACKS (free gain on the deliverable)' if st else 'DOES NOT STACK'}"
                     f"  [n={Bb['n']}: direction check, the exact sign test cannot go below p = 0.125]")
        JS["verdict_3b"] = dict(stacks=bool(st))
    contrast(S3, "AYS5pad8_deliv_g100", "AYS5pad8_origin_g100", "context: deliverable vs origin, both AYS5@1.00 pad8 (matched AYS5 sampler)", REGIME, valid=g3)

open(OUT_TXT, "w").write("\n".join(L_OUT) + "\n")
json.dump(JS, open(OUT_JSON, "w"), indent=1)
print("\n".join(L_OUT))
