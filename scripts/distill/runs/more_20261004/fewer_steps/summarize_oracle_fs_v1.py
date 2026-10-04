#!/usr/bin/env python
"""Summarise an oracle render's oracle_diag.json (written by oracle_fs_v1.py) -- the PREREG R2 readout.
Per window and mean: for every substituted step k, dx0 = ||x0t - x0o||/||x0o||, tgterr = ||x0o - x0t||/||x0t||,
trunc = ||x_target - euler(y, x0o)||/||x_target||; for the merged step also its prefix (the 8-grid k5 target on the same
window) and the endpoint errors of the teacher's 1-coarse-step (T4a-like) and 2-coarse-step (T5-like) paths vs the fine one.
usage: summarize_oracle_fs_v1.py OUT.txt <render dir>
"""
import json, os, sys
OUT, D = sys.argv[1], sys.argv[2]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
j = json.load(open(os.path.join(D, "oracle_diag.json")))
L = []
P = L.append
P(f"oracle_diag: {D}")
P(f"unet_state={j['unet_state']}  guid={j['guid']}  subst={j['subst']}  M={j['M']}  prefix={j['prefix']}  e2={j['e2']}")
P(f"usage={j['usage']}")
P("")
P("CONTROLS (window 0; bf16 floor expected, < 1e-2):")
for c in j["controls"]:
    P("  " + "  ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in c.items()))
P("")
keys = ["dx0", "tgterr", "trunc", "dx0_prefix", "tgterr_prefix", "trunc_prefix", "end_err_1coarse", "end_err_2coarse"]
for k in j["subst"]:
    rs = [r for r in j["windows"] if r["k"] == k]
    if not rs:
        continue
    r0 = rs[0]
    P(f"STEP k={k}: sigma {r0['sigma']:.6g} -> {r0['sigma_next']:.6g}, M={r0['M']}, substeps {[round(s, 6) for s in r0['subs']]}")
    have = [x for x in keys if x in r0]
    P("  win " + "".join(f"{x:>17s}" for x in have))
    for r in rs:
        P(f"  {r['win']:3d} " + "".join(f"{r[x]:17.4e}" for x in have))
    means = {x: sum(r[x] for r in rs) / len(rs) for x in have}
    P("  mean" + "".join(f"{means[x]:17.4e}" for x in have))
    if "dx0_prefix" in means:
        P(f"  R  = mean dx0(merged) / mean dx0(prefix = 8-grid k5 target)   = {means['dx0'] / means['dx0_prefix']:.3f}")
        P(f"  R' = mean trunc(merged) / mean trunc(prefix)                   = {means['trunc'] / means['trunc_prefix']:.3f}")
        per = [r['dx0'] / r['dx0_prefix'] for r in rs]
        P(f"  per-window dx0 ratio min/median/max = {min(per):.3f} / {sorted(per)[len(per) // 2]:.3f} / {max(per):.3f}")
    if "end_err_1coarse" in means:
        P(f"  endpoint error vs the fine teacher path: 1 coarse step (T4a-like) {means['end_err_1coarse']:.4e}, "
          f"2 coarse steps (T5-like) {means['end_err_2coarse']:.4e}, ratio {means['end_err_1coarse'] / means['end_err_2coarse']:.3f}")
    P("")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
