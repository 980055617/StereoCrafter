#!/usr/bin/env python
"""judge J3c (AYS smoke) + J5 (dev clips) tables from the ROW lines (CPU).  usage: analyze_J3c_J5_v1.py OUT.txt"""
import os, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
J = "scripts/distill/runs/more_20261004/judge"
def rows(p):
    out = {}
    if not os.path.exists(p): return out
    for ln in open(p):
        if ln.startswith("ROW "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:]); out[d["tag"][len(d["clip"]) + 1:]] = d
    return out
L = []
a = rows(f"{J}/SCORES_J3c_AYS_0301.txt")
if a:
    g = lambda t: float(a[t]["lpips"]); s = lambda t: float(a[t]["sharp"])
    L.append("J3c AYS smoke, 0301 (score_clip_ll.py SCORE_STEP=4; SCORES_J3c_AYS_0301.txt)")
    for t in a: L.append(f"  {t:28s} LPIPS {g(t):.6f}  sharp {s(t):.6f}  rightPSNR {float(a[t]['rightPSNR']):.3f}")
    L.append(f"  reproduction: origin_ll {g('origin_ll'):.4f} (published 0.4351); origin_g100_T5pad {g('origin_g100_T5pad'):.4f} (published 0.4416)")
    d8 = g("AYS8_origin_g101") - g("origin_ll"); d5 = g("AYS5pad8_origin_g100") - g("origin_g100_T5pad")
    L.append(f"  AYS8 @1.01 - deployed origin (8x2, Karras)      {d8:+.4f}  -> futility rule (<= -0.003): {'EXTEND' if d8 <= -0.003 else 'STOP'}")
    L.append(f"  AYS5 @1.00 pad8 - origin T5 @1.00 pad8           {d5:+.4f}  -> futility rule (<= -0.003): {'EXTEND' if d5 <= -0.003 else 'STOP'}")
    L.append(f"  context: deliverable T5@1.00 pad8 - AYS5 origin  {g('deliv_g100_T5pad') - g('AYS5pad8_origin_g100'):+.4f}; "
             f"- AYS8 origin {g('deliv_g100_T5pad') - g('AYS8_origin_g101'):+.4f}")
    L.append(f"  sharpness ratio AYS8/origin_ll {s('AYS8_origin_g101')/s('origin_ll'):.3f}; AYS5/originT5pad {s('AYS5pad8_origin_g100')/s('origin_g100_T5pad'):.3f}")
    L.append("")
for c in ["0268", "0082"]:
    r = rows(f"{J}/SCORES_J5_DEV_{c}.txt")
    if not r: continue
    g = lambda t: float(r[t]["lpips"])
    L.append(f"J5 dev clip {c} (score_clip_ll.py SCORE_STEP=4; SCORES_J5_DEV_{c}.txt), GT sharpness {float(next(iter(r.values()))['gtSharp']):.4f}")
    for t in r: L.append(f"  {t:24s} LPIPS {g(t):.6f}  sharp {float(r[t]['sharp']):.6f}  rightPSNR {float(r[t]['rightPSNR']):.3f}  offset ({r[t]['dy']},{r[t]['dx']}) n={r[t]['n']}")
    for lab, x, y in [("C1 headline   d8 - o8  ", "d8_deliv_g101_s8", "o8_origin_g101_s8"),
                      ("C2 matched    d5p - o5p", "d5p_deliv_g100_T5pad", "o5p_origin_g100_T5pad"),
                      ("C3 shipped    d5n - o8 ", "d5n_deliv_g100_T5nat", "o8_origin_g101_s8"),
                      ("   origin own lever o5p - o8", "o5p_origin_g100_T5pad", "o8_origin_g101_s8")]:
        d = g(x) - g(y); L.append(f"  {lab}  {d:+.4f}  {'(<0: holds)' if d < 0 else '(>=0: FAILS)'}")
    L.append("")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
