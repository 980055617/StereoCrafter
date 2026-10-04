"""Summarise the V1 diagnostics D1/D2 (PREREG.txt ADDENDUM 1) next to the main V1 numbers.
usage: python d_analyze.py MAIN.json DIAG.json OUT.txt"""
import json, sys
MAIN, DIAG, OUT = sys.argv[1:4]
m = json.load(open(MAIN)); d = json.load(open(DIAG))
CLIPS = [c for c in sorted(d.keys()) if c in m]
SUF = {"origin": "_origin_ll", "shipped": "_mamba_ll", "deliv": "_mstudent2_step800_deliv_ll", "s25": "_s25_ll"}
def tag(c, k):
    return next(t for t in m[c] if t.endswith(SUF[k]))
L = []; p = L.append
mean = lambda v: sum(v) / len(v)
p(f"V1 DIAGNOSTICS D1/D2 (interpretation only; F1-F3 decided by the main run) -- sources {MAIN} , {DIAG}")
p(f"clips (n={len(CLIPS)}): {' '.join(CLIPS)}")
allok = all(d[c]["check"][k]["identical"] for c in CLIPS for k in ("origin", "deliv"))
p(f"built-in check (recomputed unregistered origin/deliv warp == main JSON): {'ALL IDENTICAL' if allok else 'NOT ALL IDENTICAL -- see per clip'}")
for c in CLIPS:
    ck = d[c]["check"]
    if not all(ck[k]["identical"] for k in ck):
        p(f"  {c}: " + "  ".join(f"{k} {ck[k]['recomputed']:.6f} vs {ck[k]['main']:.6f}" for k in ck))
p("")
# D1 sanity rule (fixed before v1_diag.py ran): hole_frac <= 0.10 on every clip AND |ddx - GEOMETRY ddx| <= 6 px on the
# six clips outputs/review_20261001/GEOMETRY.txt covers; otherwise D1 is DROPPED (D2 is unaffected).
GEOM = {"0147": -41, "0141": -59, "0042": -36, "0128": -34, "0301": -15, "0204": -17}
d1_ok = all(d[c]["D1"].get("hole_frac", 1.0) <= 0.10 for c in CLIPS) and all(abs(d[c]["D1"]["ddx"] - GEOM[c]) <= 6 for c in CLIPS if c in GEOM)
p("--- D1 sanity (hole fraction, ddx vs GEOMETRY.txt) ---")
for c in CLIPS:
    x = d[c]["D1"]
    p(f"  {c}: hole_frac {x.get('hole_frac', float('nan')):.4f}  ddx {x['ddx']:4d}  GEOMETRY {GEOM.get(c, '-')}")
p(f"  D1 sanity: {'PASS -- D1 reportable' if d1_ok else 'FAIL -- D1 DROPPED (numbers below are not to be quoted)'}")
p("")
p("--- D1 registration of the real right eye to the model input (holes excluded) ---")
p(f"  {'clip':6s} {'ddy':>4s} {'ddx':>5s} {'PSNR@0':>7s} {'PSNRreg':>8s} {'validFlow unreg':>15s} {'reg':>6s}")
for c in CLIPS:
    x = d[c]["D1"]
    p(f"  {c:6s} {x['ddy']:4d} {x['ddx']:5d} {x['maskedPSNR_at0']:7.2f} {x['maskedPSNR_best']:8.2f} {x['valid_frac_unreg']:15.3f} {x['valid_frac_reg']:6.3f}")
p("")
p("--- D1 warp error with REGISTERED GT flows (and the main unregistered values for comparison) ---")
ks = [k for k in ("origin", "shipped", "deliv", "s25") if all(f"{k}_warp_reg" in d[c]["D1"] for c in CLIPS)]
p(f"  {'clip':6s} " + " ".join(f"{k + '_reg':>11s}" for k in ks) + f" {'GT_reg':>8s} | " + " ".join(f"{k + '_unreg':>13s}" for k in ks))
for c in CLIPS:
    x = d[c]["D1"]
    p(f"  {c:6s} " + " ".join(f"{x[k + '_warp_reg']:11.5f}" for k in ks) + f" {x['GT_reg_warp']:8.5f} | " +
      " ".join(f"{m[c][tag(c, k)]['warp']:13.5f}" for k in ks))
mr = {k: mean([d[c]["D1"][f"{k}_warp_reg"] for c in CLIPS]) for k in ks}
mu = {k: mean([m[c][tag(c, k)]["warp"] for c in CLIPS]) for k in ks}
p(f"  {'MEAN':6s} " + " ".join(f"{mr[k]:11.5f}" for k in ks) + f" {mean([d[c]['D1']['GT_reg_warp'] for c in CLIPS]):8.5f} | " + " ".join(f"{mu[k]:13.5f}" for k in ks))
for k in ks:
    if k == "origin": continue
    wr = sum(d[c]["D1"][f"{k}_warp_reg"] > d[c]["D1"]["origin_warp_reg"] for c in CLIPS)
    p(f"  {k:8s} vs origin: registered mean {(mr[k]/mr['origin']-1)*100:+.1f}% (worse on {wr}/{len(CLIPS)}) | unregistered mean {(mu[k]/mu['origin']-1)*100:+.1f}%")
p("")
p("--- D2 sharpness-matched origin (unsharp mask, k bisected to the deliverable's sharpness) ---")
p(f"  {'clip':6s} {'k':>6s} {'sh orig':>8s} {'sh deliv':>9s} {'sh R_k':>8s} | {'tLP o':>7s} {'tLP R_k':>8s} {'tLP d':>7s} | {'warp o':>7s} {'warp R_k':>9s} {'warp d':>7s} | {'wreg R_k':>9s} {'wreg d':>7s} | {'ratio R_k':>9s} {'ratio d':>8s}")
rows = []
for c in CLIPS:
    x = d[c]["D2"]; o = m[c][tag(c, "origin")]; dl = m[c][tag(c, "deliv")]
    r = dict(k=x["k"], so=x["sharp_origin"], sd=x["sharp_target_deliv"], sk=x["sharp_achieved"],
             to=o["tLP"], tk=x["tLP"], td=dl["tLP"], wo=o["warp"], wk=x["warp"], wd=dl["warp"],
             wrk=x["warp_reg"], wrd=d[c]["D1"]["deliv_warp_reg"], rk=x["seam"] / x["nonseam"], rd=dl["seam"] / dl["nonseam"])
    rows.append(r)
    p(f"  {c:6s} {r['k']:6.3f} {r['so']:8.5f} {r['sd']:9.5f} {r['sk']:8.5f} | {r['to']:7.4f} {r['tk']:8.4f} {r['td']:7.4f} | "
      f"{r['wo']:7.5f} {r['wk']:9.5f} {r['wd']:7.5f} | {r['wrk']:9.5f} {r['wrd']:7.5f} | {r['rk']:9.3f} {r['rd']:8.3f}")
M = {k: mean([r[k] for r in rows]) for k in rows[0]}
p(f"  {'MEAN':6s} {M['k']:6.3f} {M['so']:8.5f} {M['sd']:9.5f} {M['sk']:8.5f} | {M['to']:7.4f} {M['tk']:8.4f} {M['td']:7.4f} | "
  f"{M['wo']:7.5f} {M['wk']:9.5f} {M['wd']:7.5f} | {M['wrk']:9.5f} {M['wrd']:7.5f} | {M['rk']:9.3f} {M['rd']:8.3f}")
p("")
p("=== ADDENDUM-1 READING (fixed before the aggregate was seen) ===")
for name, kd, kk in (("warp (unregistered flows, as V1)", "wd", "wk"), ("warp (registered flows, D1)", "wrd", "wrk"), ("tLP", "td", "tk"), ("seam ratio", "rd", "rk")):
    rel = M[kd] / M[kk] - 1
    nw = sum(r[kd] > r[kk] for r in rows)
    verdict = ("deliverable EXCEEDS the sharpness-matched origin by >5% -> instability beyond its sharpness" if rel > 0.05
               else "deliverable within +5% of (or below) the sharpness-matched origin -> increase is what added sharpness alone produces")
    p(f"  {name:34s}: deliv {M[kd]:.5f} vs R_k {M[kk]:.5f} ({rel*100:+.1f}%; deliv higher on {nw}/{len(rows)}) -> {verdict}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
