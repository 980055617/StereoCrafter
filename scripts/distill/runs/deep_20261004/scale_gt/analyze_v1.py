#!/usr/bin/env python
"""scale_gt analysis: dev curve + selection (PREREG.txt) or the one-time test table.
Inputs: per-clip json of score_registered_scalegt_v1.py (<reg_dir>/<clip>.json) and score_clip_ll.py ROW logs (sharpness).
usage:
  python analyze_v1.py dev  <reg_dir> <out_txt> <rowlog> [<rowlog> ...]
  python analyze_v1.py test <reg_dir> <out_txt> <rowlog> [<rowlog> ...]
dev : reference label origin_s8; registered metric = mean REG_FRAME over DEV_REG (5 clips; 0091 unregistered only);
      every other label is a checkpoint "<run>_s<step>_s8".
test: pairs (sel_s8 vs origin_ll) and (sel_s25 vs s25_ll) on the 12 test clips (+ n=11 without 0125).
"""
import json, math, os, re, sys
import numpy as np
MODE, REG_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
logs = sys.argv[4:]
DEV6 = ["0040", "0082", "0091", "0184", "0245", "0268"]; DEV_REG = ["0040", "0082", "0184", "0245", "0268"]
TEST = ["0042", "0052", "0125", "0128", "0141", "0147", "0170", "0204", "0225", "0251", "0259", "0301"]
VARS = ["UNREG", "REG_CLIP", "REG_FRAME"]
sharp, rows_lp = {}, {}
for lg in logs:
    if lg.endswith(".json"):          # a rows json (make_rows_v1.py): sharpness / lpips of base rows such as origin_ll, s25_ll
        rj = json.load(open(lg))
        for clip, cc in rj["cells"].items():
            for lab, r in cc.items():
                sharp[(clip, lab)] = float(r["sharp"]); rows_lp[(clip, lab)] = float(r["lpips"]); sharp[(clip, "GT")] = float(r["gtSharp"])
        continue
    for line in open(lg):
        if not line.startswith("ROW "): continue
        kv = dict(t.split("=", 1) for t in line.split()[1:])
        lab = kv["tag"][len(kv["clip"]) + 1:] if kv["tag"].startswith(kv["clip"] + "_") else kv["tag"]
        sharp[(kv["clip"], lab)] = float(kv["sharp"]); rows_lp[(kv["clip"], lab)] = float(kv["lpips"])
        sharp[(kv["clip"], "GT")] = float(kv["gtSharp"])
R = {}
for c in (DEV6 if MODE == "dev" else TEST):
    p = os.path.join(REG_DIR, f"{c}.json")
    if os.path.exists(p): R[c] = json.load(open(p))
out = []
def P(*a): out.append(" ".join(str(x) for x in a))
def val(c, lab, var): return R[c]["configs"][lab]["lpips_clip"][var]
def rps(c, lab, var): return R[c]["configs"][lab]["rPSNR"][var]
def boot_ci(d, n=20000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1); return float(np.quantile(m, .025)), float(np.quantile(m, .975))
def sign_p(d):
    d = [x for x in d if x != 0]; k = sum(x < 0 for x in d); n = len(d)
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0
# G0-like gate: UNREG column must equal the score_clip_ll lpips of the same render
g0 = []
for c in R:
    for lab, cf in R[c]["configs"].items():
        if (c, lab) in rows_lp: g0.append(abs(cf["lpips_clip"]["UNREG"] - rows_lp[(c, lab)]))
P(f"GATE G0 (registered scorer UNREG == score_clip_ll lpips): {len(g0)} cells, max |d| = {max(g0) if g0 else float('nan'):.2e}",
  "PASS" if g0 and max(g0) < 1e-5 else "FAIL/EMPTY")
if MODE == "dev":
    ref = "origin_s8"
    labs = sorted({l for c in R for l in R[c]["configs"]} - {ref},
                  key=lambda l: (l.split("_s")[0], int(re.search(r"_s(\d+)_s8$", l).group(1)) if re.search(r"_s(\d+)_s8$", l) else 0))
    P(f"\nDEV  ({len(R)} clips scored: {sorted(R)}; registered metric over {DEV_REG}; 0091 UNREG only)")
    P(f"{'label':22s} {'REG_FRAME5':>10s} {'d':>8s} {'impr':>5s} | {'REG_CLIP5':>9s} {'d':>8s} | {'UNREG6':>8s} {'d':>8s} {'impr':>5s} | {'UNREG5':>8s} {'d':>8s} | {'sharp6':>7s} {'ratio':>6s} | {'rPSNRreg5':>9s} {'d':>6s}")
    def row(lab):
        r5 = [c for c in DEV_REG if c in R and lab in R[c]["configs"]]; r6 = [c for c in DEV6 if c in R and lab in R[c]["configs"]]
        if not r5: return None
        f = np.mean([val(c, lab, "REG_FRAME") for c in r5]); fo = np.mean([val(c, ref, "REG_FRAME") for c in r5])
        k = np.mean([val(c, lab, "REG_CLIP") for c in r5]); ko = np.mean([val(c, ref, "REG_CLIP") for c in r5])
        u = np.mean([val(c, lab, "UNREG") for c in r6]); uo = np.mean([val(c, ref, "UNREG") for c in r6])
        u5 = np.mean([val(c, lab, "UNREG") for c in r5]); u5o = np.mean([val(c, ref, "UNREG") for c in r5])
        imp = sum(val(c, lab, "REG_FRAME") < val(c, ref, "REG_FRAME") for c in r5); impu = sum(val(c, lab, "UNREG") < val(c, ref, "UNREG") for c in r6)
        sh = [sharp.get((c, lab)) for c in r6]; sho = [sharp.get((c, ref)) for c in r6]
        shm = np.mean(sh) if all(x is not None for x in sh) else float("nan"); shom = np.mean(sho) if all(x is not None for x in sho) else float("nan")
        rp = np.mean([rps(c, lab, "REG_FRAME") for c in r5]); rpo = np.mean([rps(c, ref, "REG_FRAME") for c in r5])
        return dict(lab=lab, f=f, df=f - fo, imp=imp, n5=len(r5), k=k, dk=k - ko, u=u, du=u - uo, impu=impu, n6=len(r6), u5=u5, du5=u5 - u5o,
                    sh=shm, shr=shm / shom if shom == shom else float("nan"), rp=rp, drp=rp - rpo)
    allrows = {}
    for lab in [ref] + labs:
        r = row(lab)
        if r is None: continue
        allrows[lab] = r
        P(f"{lab:22s} {r['f']:10.4f} {r['df']:+8.4f} {r['imp']:2d}/{r['n5']} | {r['k']:9.4f} {r['dk']:+8.4f} | {r['u']:8.4f} {r['du']:+8.4f} {r['impu']:2d}/{r['n6']} | "
          f"{r['u5']:8.4f} {r['du5']:+8.4f} | {r['sh']:7.4f} {r['shr']:6.3f} | {r['rp']:9.3f} {r['drp']:+6.3f}")
    P("\nPER-CLIP REG_FRAME deltas vs origin_s8 (0091: UNREG delta)")
    P(f"{'label':22s} " + " ".join(f"{c:>8s}" for c in DEV6))
    for lab in labs:
        cells = []
        for c in DEV6:
            if c not in R or lab not in R[c]["configs"]: cells.append(f"{'-':>8s}"); continue
            v = "UNREG" if c == "0091" else "REG_FRAME"
            cells.append(f"{val(c, lab, v) - val(c, ref, v):+8.4f}")
        P(f"{lab:22s} " + " ".join(cells))
    main = {int(re.search(r"_s(\d+)_s8$", l).group(1)): r for l, r in allrows.items() if l.startswith("main_v1_s") and re.search(r"_s(\d+)_s8$", l)}
    con = {int(re.search(r"_s(\d+)_s8$", l).group(1)): r for l, r in allrows.items() if l.startswith("contrast8_v1_s") and re.search(r"_s(\d+)_s8$", l)}
    if main:
        best = min(sorted(main), key=lambda s: (round(main[s]["f"], 10), s))
        # tie rule: within 0.0005 of the minimum -> the earliest such step
        mn = min(main[s]["f"] for s in main); best = min(s for s in main if main[s]["f"] <= mn + 0.0005)
        P(f"\nSELECTION (PREREG): evaluated MAIN steps {sorted(main)}; lowest registered dev metric {mn:.4f} at step "
          f"{min(main, key=lambda s: main[s]['f'])}; tie rule (<= min+0.0005 -> earliest) selects step {best} "
          f"(REG_FRAME5 {main[best]['f']:.4f}, delta {main[best]['df']:+.4f})")
        worse_all = all(main[s]["df"] > 0 for s in main)
        P(f"STILL-DEGRADES test (MAIN worse than origin on the registered dev metric at EVERY evaluated checkpoint): {worse_all}")
        nb = [s for s in (best - 250, best + 250) if 0 < s <= 3000 and s not in main]
        P(f"Stage-2 neighbours of the best Stage-1 step still to evaluate: {nb}")
    if main and con:
        P("\nSCALE CONTRAST at matched steps (registered dev delta vs origin; lower = better):")
        for s in sorted(set(main) & set(con)):
            P(f"  step {s:5d}: MAIN {main[s]['df']:+.4f} (UNREG {main[s]['du']:+.4f})   CONTRAST8 {con[s]['df']:+.4f} (UNREG {con[s]['du']:+.4f})   "
              f"MAIN better: {main[s]['df'] < con[s]['df']}")
else:
    ER = "outputs/more_20261004/eval_robustness/score_v1"
    gr = []
    for c in R:
        e = json.load(open(os.path.join("/home/kawa/master_project/StereoCrafter", ER, f"{c}.json")))["configs"]
        for lab in ("origin_ll", "s25_ll"):
            if lab in R[c]["configs"] and lab in e:
                for v in ("UNREG", "REG_CLIP", "REG_FRAME"):
                    gr.append(abs(R[c]["configs"][lab]["lpips_clip"][v] - e[lab]["lpips_clip"][v]))
    P(f"GATE G-REG (origin_ll / s25_ll UNREG, REG_CLIP, REG_FRAME == eval_robustness score_v1): {len(gr)} cells, max |d| = "
      f"{max(gr) if gr else float('nan'):.2e}", "PASS" if gr and max(gr) < 1e-5 else "FAIL/EMPTY")
    pairs = [(os.environ.get("SEL8", "sel_s8"), "origin_ll", "8 steps"), (os.environ.get("SEL25", "sel_s25"), "s25_ll", "25 steps"),
             (os.environ.get("SEL8", "sel_s8") + "_colrm", "origin_ll", "8 steps COLOUR-REMOVED (diagnostic, PREREG_ADDENDUM_2)"),
             (os.environ.get("SEL25", "sel_s25") + "_colrm", "s25_ll", "25 steps COLOUR-REMOVED (diagnostic)")]
    for lab, ref, nm in pairs:
        cl = [c for c in TEST if c in R and lab in R[c]["configs"] and ref in R[c]["configs"]]
        if not cl: continue
        P(f"\nTEST {nm}: {lab} vs {ref}  ({len(cl)} clips)")
        P(f"{'clip':6s} " + " ".join(f"{v+' '+ref[:6]:>16s} {v+' '+lab[:7]:>16s} {'d':>8s}" for v in VARS) + f" {'sharp ref':>9s} {'sharp':>7s} {'rPSNRreg d':>10s}")
        D = {v: [] for v in VARS}
        for c in cl:
            cells = []
            for v in VARS:
                a, b = val(c, ref, v), val(c, lab, v); D[v].append(b - a); cells.append(f"{a:16.4f} {b:16.4f} {b-a:+8.4f}")
            P(f"{c:6s} " + " ".join(cells) + f" {sharp.get((c, ref), float('nan')):9.4f} {sharp.get((c, lab), float('nan')):7.4f} "
              f"{rps(c, lab, 'REG_FRAME') - rps(c, ref, 'REG_FRAME'):+10.3f}" + ("   (REG_FRAME ill-posed, eval_robustness)" if c == "0125" else ""))
        for v in VARS:
            d = D[v]; lo, hi = boot_ci(d)
            d11 = [x for c, x in zip(cl, d) if c != "0125"]
            P(f"  {v:9s} mean {np.mean([val(c, ref, v) for c in cl]):.4f} -> {np.mean([val(c, lab, v) for c in cl]):.4f}  delta {np.mean(d):+.4f}  "
              f"pct-boot95 [{lo:+.4f},{hi:+.4f}]  improved {sum(x < 0 for x in d)}/{len(d)}  p_sign {sign_p(d):.4f}   n=11 (no 0125): "
              f"{np.mean(d11):+.4f} {sum(x < 0 for x in d11)}/{len(d11)}")
        shr = np.mean([sharp[(c, lab)] for c in cl]) / np.mean([sharp[(c, ref)] for c in cl])
        drp = np.mean([rps(c, lab, "REG_FRAME") - rps(c, ref, "REG_FRAME") for c in cl])
        P(f"  sharpness ratio (12-clip mean sharp {lab}/{ref}) = {shr:.3f};  registered rPSNR delta = {drp:+.3f} dB")
        if nm.startswith("8 steps"):
            d = D["REG_FRAME"]
            works = np.mean(d) <= -0.005 and sum(x < 0 for x in d) >= 9 and shr >= 0.90 and drp >= -0.5
            P(f"  PREREG bar 'WORKS AT SCALE' (mean <= -0.005, >= 9/12, sharp >= 0.90x, rPSNRreg >= -0.5 dB) on [{nm}]: {works}")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
