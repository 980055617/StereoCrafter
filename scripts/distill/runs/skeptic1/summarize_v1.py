#!/usr/bin/env python
"""Build the skeptic tables from the ROW lines of my own re-scoring runs."""
import re, sys, os, collections

def rows(path):
    out = collections.defaultdict(dict)
    gt = {}
    if not os.path.exists(path): return out, gt
    for ln in open(path):
        if not ln.startswith("ROW "): continue
        d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
        c = d["clip"]; tag = d["tag"]
        out[c][tag] = dict(lpips=float(d["lpips"]), sharp=float(d["sharp"]),
                           gtSharp=float(d["gtSharp"]), leftPSNR=float(d["leftPSNR"]),
                           rightPSNR=float(d["rightPSNR"]), n=int(d["n"]),
                           dy=int(d["dy"]), dx=int(d["dx"]), path=d["path"])
        gt[c] = float(d["gtSharp"])
    return out, gt

MP = "scripts/distill/runs/skeptic1/RESCORE_12clip_mp4v.txt"
LL = "scripts/distill/runs/skeptic1/SCORES_STACK_lossless.txt"
mp, gtm = rows(MP)
ll, gtl = rows(LL)
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
STU = {"0301": "0301_smoke1_step800", "0204": "0204_smoke1_step800"}
def stag(c):
    if c in STU: return STU[c]
    return f"{c}_heldout_smoke1s800" if c in ("0052", "0128", "0125", "0170") else f"{c}_heldout_s800rest"

print("=" * 100)
print("TABLE 1  INDEPENDENT RE-SCORE OF THE STEP-DISTILLATION LANE (mp4v, on-disk outputs, my own run)")
print("=" * 100)
print(f"{'clip':5s} {'gtSharp':>8s} | {'origin':>8s} {'s25':>8s} {'student':>8s} {'mamba':>8s} | "
      f"{'d_s25':>8s} {'d_stud':>8s} {'d_mamba':>8s} | {'frac':>6s} | {'regime':>6s} | offsets/nf/leftPSNR agree?")
agg = collections.defaultdict(list)
for c in CLIPS:
    r = mp.get(c, {})
    o = r.get(f"{c}_origin"); s = r.get(f"{c}_origin_s25"); t = r.get(stag(c)); m = r.get(f"{c}_all_8k_v2")
    if not all((o, s, t, m)):
        print(f"{c} INCOMPLETE {sorted(r)}"); continue
    ds, dt, dm = s["lpips"] - o["lpips"], t["lpips"] - o["lpips"], m["lpips"] - o["lpips"]
    frac = dt / ds if abs(ds) > 1e-4 else float('nan')
    reg = "over" if o["sharp"] > o["gtSharp"] else "under"
    ok = (len({(x["dy"], x["dx"]) for x in (o, s, t, m)}) == 1 and
          len({x["n"] for x in (o, s, t, m)}) == 1 and
          len({round(x["leftPSNR"], 2) for x in (o, s, t, m)}) == 1)
    print(f"{c:5s} {o['gtSharp']:8.4f} | {o['lpips']:8.4f} {s['lpips']:8.4f} {t['lpips']:8.4f} {m['lpips']:8.4f} | "
          f"{ds:+8.4f} {dt:+8.4f} {dm:+8.4f} | {frac*100:5.1f}% | {reg:>6s} | {'yes' if ok else 'NO'}")
    agg["o"].append(o["lpips"]); agg["s"].append(s["lpips"]); agg["t"].append(t["lpips"]); agg["m"].append(m["lpips"])
    agg["reg"].append(reg); agg["clip"].append(c)
    agg["shr"].append(t["sharp"] / o["sharp"])
n = len(agg["o"])
mean = lambda k: sum(agg[k]) / len(agg[k])
if n:
    print(f"{'MEAN':5s} {'':8s} | {mean('o'):8.4f} {mean('s'):8.4f} {mean('t'):8.4f} {mean('m'):8.4f} | "
          f"{mean('s')-mean('o'):+8.4f} {mean('t')-mean('o'):+8.4f} {mean('m')-mean('o'):+8.4f} | "
          f"{(mean('t')-mean('o'))/(mean('s')-mean('o'))*100:5.1f}%")
    print(f"\n  n={n}  student improved {sum(1 for o,t in zip(agg['o'],agg['t']) if t<o)}/{n}, "
          f"regressed {sum(1 for o,t in zip(agg['o'],agg['t']) if t>o)}/{n}; "
          f"mean sharpness ratio student/origin {sum(agg['shr'])/n:.4f}, min {min(agg['shr']):.4f}")
    ho = [i for i, c in enumerate(agg["clip"]) if c not in STU]
    mo = lambda k, idx: sum(agg[k][i] for i in idx) / len(idx)
    print(f"  TRAINED CLIPS (0301,0204): origin {mo('o',[i for i,c in enumerate(agg['clip']) if c in STU]):.5f} "
          f"student {mo('t',[i for i,c in enumerate(agg['clip']) if c in STU]):.5f} "
          f"s25 {mo('s',[i for i,c in enumerate(agg['clip']) if c in STU]):.5f}")
    dS = mo('s', ho) - mo('o', ho); dT = mo('t', ho) - mo('o', ho)
    print(f"  HELD-OUT ONLY (n={len(ho)}): origin {mo('o',ho):.5f} s25 {mo('s',ho):.5f} student {mo('t',ho):.5f} "
          f"=> d_s25 {dS:+.5f}  d_student {dT:+.5f}  fraction {dT/dS*100:.1f}%")
    for reg in ("under", "over"):
        idx = [i for i, r in enumerate(agg["reg"]) if r == reg]
        dS = mo('s', idx) - mo('o', idx); dT = mo('t', idx) - mo('o', idx)
        print(f"  regime {reg:5s} (n={len(idx)}): d_s25 {dS:+.5f} d_student {dT:+.5f} fraction {dT/dS*100:.1f}%")

print()
print("=" * 100)
print("TABLE 2  STACKING WITH THE SHIPPED MAMBA DELIVERABLE -- ALL LOSSLESS (FFV1)")
print("=" * 100)
LAB = [("origin", "{c}_origin_ll"), ("origin+s25", "{c}_s25_ll"),
       ("mamba", "{c}_mamba_ll"), ("mamba+s25", "{c}_mamba_s25_ll"),
       ("origin+student", "{c}_student_ll"), ("mamba+student", "{c}_mamba_student_ll")]
SC = "0301 0204 0052 0147".split()
acc = collections.defaultdict(list)
for c in SC:
    r = ll.get(c, {})
    if not r: print(f"{c}: no rows yet"); continue
    o = r.get(f"{c}_origin_ll")
    print(f"--- {c}  (GT sharp {gtl.get(c, float('nan')):.4f}) ---")
    print(f"  {'row':16s} {'LPIPS':>8s} {'delta':>9s} {'sharp':>8s} {'sh/orig':>8s} {'leftPSNR':>9s} {'rPSNR':>8s} {'nf':>4s} {'offset':>9s}")
    for lab, pat in LAB:
        x = r.get(pat.format(c=c))
        if x is None: continue
        d = (x["lpips"] - o["lpips"]) if o else float('nan')
        print(f"  {lab:16s} {x['lpips']:8.4f} {d:+9.4f} {x['sharp']:8.4f} "
              f"{x['sharp']/o['sharp'] if o else float('nan'):8.4f} {x['leftPSNR']:9.2f} {x['rightPSNR']:8.3f} "
              f"{x['n']:4d} {f'({x[chr(100)+chr(121)]},{x[chr(100)+chr(120)]})':>9s}")
        acc[lab].append(x["lpips"])
    if o and all(f"{c}_{k}" in r for k in ("mamba_ll", "mamba_s25_ll", "s25_ll")):
        dm = r[f"{c}_mamba_ll"]["lpips"] - o["lpips"]
        ds = r[f"{c}_s25_ll"]["lpips"] - o["lpips"]
        dms = r[f"{c}_mamba_s25_ll"]["lpips"] - r[f"{c}_s25_ll"]["lpips"]
        print(f"  COMPOSITION: mamba cost at 8 steps {dm:+.4f} vs mamba cost on top of s25 {dms:+.4f} "
              f"(difference {dms-dm:+.4f});  s25 gain on origin {ds:+.4f}, "
              f"on mamba {r[f'{c}_mamba_s25_ll']['lpips']-r[f'{c}_mamba_ll']['lpips']:+.4f}")
if acc.get("origin"):
    print("\n  MEANS over the scored clips:")
    base = sum(acc["origin"]) / len(acc["origin"])
    for lab, _ in LAB:
        if lab in acc and len(acc[lab]) == len(acc["origin"]):
            v = sum(acc[lab]) / len(acc[lab])
            print(f"    {lab:16s} {v:.4f}  delta vs origin {v-base:+.4f}  (n={len(acc[lab])})")
