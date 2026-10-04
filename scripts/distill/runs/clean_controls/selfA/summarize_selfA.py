"""CONTROL A pre-registered readout from scores_selfA.txt (score_clip_ll.py ROW lines, SCORE_STEP=4, lossless path).
Reference = THIS chain's no-hook lossless origin render (0301_origin_nohook_ll / 0204_origin_nohook_ll).
Old (mp4v-scored, minift/scores_null.txt) P1-null drift on 0301: +0.0307 / +0.0483 / +0.0596 LPIPS at steps 100/200/300,
sharp 0.0235 -> 0.0203 / 0.0190 / 0.0184.  Pass = |dLPIPS| <= 0.005 and |dsharp| <= 5 % at all three steps;
Fail = dLPIPS(step300) >= +0.03; otherwise PARTIAL with the fraction of the old drift that remains.
usage: python summarize_selfA.py scores_selfA.txt
"""
import sys, re
rows = {}
for line in open(sys.argv[1]):
    if not line.startswith("ROW "): continue
    kv = dict(re.findall(r"(\w+)=(\S+)", line)); rows[(kv["clip"], kv["tag"])] = kv
OLD = {"0301": {"origin": (0.4445, 0.0235), "100": (0.4752, 0.0203), "200": (0.4928, 0.0190), "300": (0.5041, 0.0184), "e1high": (0.4472, 0.0230), "e1low": (0.5001, 0.0187)}}
def get(clip, tag): r = rows.get((clip, tag)); return (float(r["lpips"]), float(r["sharp"]), r["dy"], r["dx"], r["leftPSNR"]) if r else None
print(f"{'clip':5s} {'variant':30s} {'LPIPS':>7s} {'dLPIPS':>8s} {'sharp':>7s} {'dsharp%':>8s} {'offset':>9s} {'leftPSNR':>8s} | {'old mp4v LPIPS/sharp':>22s} {'old dLPIPS':>10s} {'frac of old drift left':>22s}")
verdict = {}
for clip in ("0301", "0204"):
    ref = get(clip, f"{clip}_origin_nohook_ll")
    if ref is None: print(f"{clip}: no origin reference row"); continue
    L0, S0 = ref[0], ref[1]
    for tag, variant in ((f"{clip}_origin_nohook_ll", "origin (no hook, lossless) REF"), (f"{clip}_origin_ll", "origin beyond4 render (2026-10-01)"), (f"{clip}_originall_hook_ll", "origin through hook (originall)"),
                         (f"{clip}_llnull_step100", "llnull step100"), (f"{clip}_llnull_step200", "llnull step200"), (f"{clip}_llnull_step300", "llnull step300"),
                         (f"{clip}_llnull_step300_e1high", "llnull step300 e1high (sigma>=103)"), (f"{clip}_llnull_step300_e1low", "llnull step300 e1low (sigma<=31)")):
        r = get(clip, tag)
        if r is None: continue
        L, S = r[0], r[1]; dL = L - L0; dS = (S - S0) / S0 * 100
        key = tag.split("_")[-1] if "step" in tag else ("origin" if "origin" in tag else "")
        key = {"step100": "100", "step200": "200", "step300": "300"}.get(key, key)
        old = OLD.get(clip, {}).get(key); oldref = OLD.get(clip, {}).get("origin")
        if old and oldref and key not in ("origin", ""):
            odL = old[0] - oldref[0]; frac = dL / odL if odL else float("nan")
            extra = f"| {old[0]:.4f}/{old[1]:.4f}{'':>9s} {odL:+10.4f} {frac:22.3f}"
            if clip == "0301" and key in ("100", "200", "300"): verdict[key] = (dL, dS, frac)
        else: extra = "|"
        print(f"{clip:5s} {variant:30s} {L:7.4f} {dL:+8.4f} {S:7.4f} {dS:+8.2f} {f'({r[2]},{r[3]})':>9s} {float(r[4]):8.2f} {extra}")
if len(verdict) == 3:
    d300, s300, f300 = verdict["300"]
    passed = all(abs(v[0]) <= 0.005 and abs(v[1]) <= 5.0 for v in verdict.values())
    failed = d300 >= 0.03
    print("\nPRE-REGISTERED READOUT (0301, steps 100/200/300 vs this chain's lossless origin):")
    for k in ("100", "200", "300"): print(f"   step {k}: dLPIPS {verdict[k][0]:+.4f}  dsharp {verdict[k][1]:+.2f} %  fraction of old drift remaining {verdict[k][2]:.3f}")
    if passed: print(f"   -> PASS: CODEC WAS THE CAUSE: residual drift {d300:+.4f} of +0.060 ({f300*100:.0f} %)")
    elif failed: print(f"   -> FAIL: COLLAPSE CONFIRMED ON CLEAN TARGET: drift {d300:+.4f} ({f300*100:.0f} % of the old +0.060)")
    else: print(f"   -> PARTIAL: {f300:.2f} of the old drift remains (dLPIPS {d300:+.4f} at step 300, dsharp {s300:+.1f} %)")
