#!/usr/bin/env python
"""All skeptic tables from the ROW lines of my own scoring runs."""
import os, collections
D = "scripts/distill/runs/skeptic1"

def rows(path):
    out = collections.defaultdict(dict); gt = {}
    if not os.path.exists(path): return out, gt
    for ln in open(path):
        if not ln.startswith("ROW "): continue
        d = dict(kv.split("=", 1) for kv in ln.split()[1:] if "=" in kv)
        out[d["clip"]][d["tag"]] = {k: (float(d[k]) if k not in ("n",) else int(d[k]))
                                    for k in ("lpips", "sharp", "gtSharp", "leftPSNR", "rightPSNR", "n")}
        out[d["clip"]][d["tag"]]["off"] = (int(d["dy"]), int(d["dx"]))
        gt[d["clip"]] = float(d["gtSharp"])
    return out, gt

FLOOR = 0.001
SC = ["0301", "0204", "0052", "0147"]
REG = {"0301": "under", "0204": "under", "0052": "over", "0147": "over"}

ll, gtl = rows(f"{D}/SCORES_STACK_lossless.txt")
print("=" * 104)
print("TABLE 2  STACKING THE WINNER ON THE SHIPPED MAMBA DELIVERABLE -- ALL LOSSLESS (FFV1), real-GT LPIPS")
print("=" * 104)
LAB = [("origin", "{c}_origin_ll"), ("origin+s25", "{c}_s25_ll"), ("mamba", "{c}_mamba_ll"),
       ("mamba+s25", "{c}_mamba_s25_ll"), ("origin+student", "{c}_student_ll"),
       ("mamba+student", "{c}_mamba_student_ll")]
acc = collections.defaultdict(list); comp = []
for c in SC:
    r = ll.get(c, {})
    if not r: print(f"{c}: MISSING"); continue
    o = r.get(f"{c}_origin_ll")
    print(f"--- {c}  regime={REG[c]}  GT sharp {gtl.get(c, float('nan')):.4f} ---")
    print(f"  {'row':16s} {'LPIPS':>8s} {'d vs orig':>10s} {'sharp':>8s} {'sh/orig':>8s} "
          f"{'leftPSNR':>9s} {'nf':>4s} {'offset':>9s}")
    for lab, pat in LAB:
        x = r.get(pat.format(c=c))
        if x is None: continue
        print(f"  {lab:16s} {x['lpips']:8.4f} {x['lpips']-o['lpips']:+10.4f} {x['sharp']:8.4f} "
              f"{x['sharp']/o['sharp']:8.4f} {x['leftPSNR']:9.2f} {x['n']:4d} {str(x['off']):>9s}")
        acc[lab].append(x["lpips"])
    need = [f"{c}_{k}" for k in ("mamba_ll", "s25_ll", "mamba_s25_ll")]
    if o and all(k in r for k in need):
        dm = r[f"{c}_mamba_ll"]["lpips"] - o["lpips"]
        dms = r[f"{c}_mamba_s25_ll"]["lpips"] - r[f"{c}_s25_ll"]["lpips"]
        ds_o = r[f"{c}_s25_ll"]["lpips"] - o["lpips"]
        ds_m = r[f"{c}_mamba_s25_ll"]["lpips"] - r[f"{c}_mamba_ll"]["lpips"]
        comp.append((c, dm, dms, dms - dm, ds_o, ds_m))
if comp:
    print("\n  COMPOSITION TEST  (measurement floor ~%.4f; runs are bit-deterministic)" % FLOOR)
    print(f"  {'clip':5s} {'mamba cost @8st':>16s} {'mamba cost @s25':>16s} {'difference':>11s} "
          f"{'s25 gain on orig':>17s} {'s25 gain on mamba':>18s} {'verdict':>9s}")
    for c, dm, dms, dd, dso, dsm in comp:
        print(f"  {c:5s} {dm:+16.4f} {dms:+16.4f} {dd:+11.4f} {dso:+17.4f} {dsm:+18.4f} "
              f"{'NULL' if abs(dd) < FLOOR else 'REAL':>9s}")
    n = len(comp)
    print(f"  {'MEAN':5s} {sum(x[1] for x in comp)/n:+16.4f} {sum(x[2] for x in comp)/n:+16.4f} "
          f"{sum(x[3] for x in comp)/n:+11.4f} {sum(x[4] for x in comp)/n:+17.4f} "
          f"{sum(x[5] for x in comp)/n:+18.4f}")
if acc.get("origin"):
    base = sum(acc["origin"]) / len(acc["origin"])
    print("\n  4-CLIP MEANS:")
    for lab, _ in LAB:
        if lab in acc and len(acc[lab]) == len(acc["origin"]):
            v = sum(acc[lab]) / len(acc[lab])
            print(f"    {lab:16s} {v:.4f}   delta vs origin {v-base:+.4f}")
        elif lab in acc:
            print(f"    {lab:16s} (only {len(acc[lab])}/{len(acc['origin'])} clips: "
                  f"{', '.join(f'{v:.4f}' for v in acc[lab])})")

# ---------------- 12-clip lossless ----------------
l12, gt12 = rows(f"{D}/SCORES_12CLIP_LOSSLESS.txt")
CL12 = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
if l12:
    print()
    print("=" * 104)
    print("TABLE 3  THE MEASUREMENT NEITHER LANE MADE: 12-clip LOSSLESS origin / s25 / student")
    print("=" * 104)
    print(f"{'clip':5s} {'origin':>8s} {'s25':>8s} {'student':>8s} | {'d_s25':>8s} {'d_stud':>8s} "
          f"{'frac':>7s} | {'shR stu':>8s} {'regime':>6s} {'nf':>4s} {'left=':>6s}")
    A = collections.defaultdict(list)
    for c in CL12:
        r = l12.get(c, {})
        o = r.get(f"{c}_origin_ll"); s = r.get(f"{c}_s25_ll"); t = r.get(f"{c}_student_ll")
        if not all((o, s, t)):
            print(f"{c:5s} INCOMPLETE {sorted(r)}"); continue
        ds, dt = s["lpips"] - o["lpips"], t["lpips"] - o["lpips"]
        reg = "over" if o["sharp"] > o["gtSharp"] else "under"
        same = (len({x["off"] for x in (o, s, t)}) == 1 and len({x["n"] for x in (o, s, t)}) == 1
                and len({round(x["leftPSNR"], 2) for x in (o, s, t)}) == 1)
        print(f"{c:5s} {o['lpips']:8.4f} {s['lpips']:8.4f} {t['lpips']:8.4f} | {ds:+8.4f} {dt:+8.4f} "
              f"{(dt/ds*100 if abs(ds)>1e-4 else float('nan')):6.1f}% | {t['sharp']/o['sharp']:8.4f} "
              f"{reg:>6s} {o['n']:4d} {'yes' if same else 'NO':>6s}")
        for k, v in (("o", o), ("s", s), ("t", t)): A[k].append(v["lpips"])
        A["reg"].append(reg); A["clip"].append(c); A["shr"].append(t["sharp"] / o["sharp"])
    if A["o"]:
        n = len(A["o"]); m = lambda k: sum(A[k]) / n
        print(f"{'MEAN':5s} {m('o'):8.4f} {m('s'):8.4f} {m('t'):8.4f} | {m('s')-m('o'):+8.4f} "
              f"{m('t')-m('o'):+8.4f} {(m('t')-m('o'))/(m('s')-m('o'))*100:6.1f}%")
        print(f"  n={n}  s25 improved {sum(1 for o,s in zip(A['o'],A['s']) if s<o)}/{n}, "
              f"student improved {sum(1 for o,t in zip(A['o'],A['t']) if t<o)}/{n}; "
              f"student worst {max(t-o for o,t in zip(A['o'],A['t'])):+.4f}, "
              f"s25 worst {max(s-o for o,s in zip(A['o'],A['s'])):+.4f}")
        ho = [i for i, c in enumerate(A["clip"]) if c not in ("0301", "0204")]
        mo = lambda k, idx: sum(A[k][i] for i in idx) / len(idx)
        dS, dT = mo('s', ho) - mo('o', ho), mo('t', ho) - mo('o', ho)
        print(f"  HELD-OUT ONLY (n={len(ho)}): origin {mo('o',ho):.5f} s25 {mo('s',ho):.5f} "
              f"student {mo('t',ho):.5f} => d_s25 {dS:+.5f} d_student {dT:+.5f} frac {dT/dS*100:.1f}%")
        for reg in ("under", "over"):
            idx = [i for i, r in enumerate(A["reg"]) if r == reg]
            if not idx: continue
            dS, dT = mo('s', idx) - mo('o', idx), mo('t', idx) - mo('o', idx)
            print(f"  regime {reg:5s} (n={len(idx)}): d_s25 {dS:+.5f} d_student {dT:+.5f} frac {dT/dS*100:.1f}%")

# ---------------- down0 ----------------
d0, gtd = rows(f"{D}/SCORES_DOWN0.txt")
if d0:
    print()
    print("=" * 104)
    print("TABLE 4  CAN THE TWO WINS COEXIST?  2-slot Mamba (down_blocks.0 only) + the distilled up3 tensors")
    print("=" * 104)
    L4 = [("origin", "{c}_origin_ll"), ("origin+student", "{c}_student_ll"),
          ("mamba 5-slot", "{c}_mamba_ll"), ("mamba 2-slot", "{c}_mamba_down0_ll"),
          ("mamba2+student", "{c}_mamba_down0_student_ll")]
    B = collections.defaultdict(list)
    for c in SC:
        r = d0.get(c, {})
        if not r: print(f"{c}: MISSING"); continue
        o = r.get(f"{c}_origin_ll")
        print(f"--- {c} regime={REG[c]} ---")
        for lab, pat in L4:
            x = r.get(pat.format(c=c))
            if x is None: continue
            print(f"  {lab:16s} {x['lpips']:8.4f} {x['lpips']-o['lpips']:+10.4f} sharp {x['sharp']:.4f} "
                  f"nf {x['n']}")
            B[lab].append(x["lpips"])
    if B.get("origin"):
        base = sum(B["origin"]) / len(B["origin"])
        print("\n  4-CLIP MEANS:")
        for lab, _ in L4:
            if lab in B and len(B[lab]) == len(B["origin"]):
                v = sum(B[lab]) / len(B[lab])
                print(f"    {lab:16s} {v:.4f}   delta vs origin {v-base:+.4f}")
