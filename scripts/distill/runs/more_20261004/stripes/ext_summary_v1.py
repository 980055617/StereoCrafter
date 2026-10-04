#!/usr/bin/env python
"""E1-E3 of PREREG_ADDENDUM_2 / 2b: deliverable A (rowlin) on all 12 test clips vs its own unfilled baseline
(deliv_g100_T5nat) and vs deployed origin (origin_ll).  Every LPIPS is read from the given SCORES files (ROW lines);
crack fractions from each render's stripe_fill_log.json.
usage: ext_summary_v1.py OUT.txt TRIGGER_STATUS SCORES1 [SCORES2 ...]
"""
import json
import os
import sys

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT, TRIG = sys.argv[1], sys.argv[2]
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}")
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
lp, src = {}, {}
for f in sys.argv[3:]:
    for line in open(f):
        if line.startswith("ROW "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            lab = kv["tag"].split("_", 1)[1]
            key = (kv["clip"], lab)
            v = float(kv["lpips"])
            if key in lp and abs(lp[key] - v) > 1e-9:
                sys.exit(f"conflicting LPIPS for {key}")
            lp[key] = v
            src.setdefault(key, f)
crack = {}
for c in CLIPS:
    p = f"outputs/more_20261004/stripes/clips/{c}_deliv_g100_T5nat_A_rowlin/stripe_fill_log.json"
    if os.path.exists(p):
        crack[c] = json.load(open(p))["crack_frac_window"]
L = []
P = L.append
P("E1-E3 (PREREG_ADDENDUM_2 / 2b) -- deliverable T5 @1.00 unpadded, crack fill A (rowlin/keep) vs unfilled, 12 test clips")
P(f"TRIGGER STATUS: {TRIG}")
P(f"{'clip':6s} {'crack%':>7s} {'origin_ll':>10s} {'deliv':>10s} {'deliv+A':>10s} {'A-deliv':>9s} {'A-origin':>9s}")
rows = []
for c in CLIPS:
    o, d, a = lp.get((c, "origin_ll")), lp.get((c, "deliv_g100_T5nat")), lp.get((c, "deliv_g100_T5nat_A_rowlin"))
    if None in (o, d, a):
        P(f"{c:6s} missing: origin_ll={o} deliv={d} A={a}")
        continue
    rows.append((c, crack.get(c, float('nan')), o, d, a))
    P(f"{c:6s} {100*crack.get(c, float('nan')):7.3f} {o:10.6f} {d:10.6f} {a:10.6f} {a-d:+9.6f} {a-o:+9.6f}")
n = len(rows)
if n:
    md = sum(r[4] - r[3] for r in rows) / n
    mo = sum(r[2] for r in rows) / n
    mdl = sum(r[3] for r in rows) / n
    ma = sum(r[4] for r in rows) / n
    better = sum(r[4] < r[3] for r in rows)
    P(f"\nE1 n={n}: mean LPIPS origin {mo:.4f} | deliverable {mdl:.4f} | deliverable+A {ma:.4f}; mean delta A-deliv {md:+.5f}; "
      f"A better on {better}/{n} clips; worst clip {max(rows, key=lambda r: r[4]-r[3])[0]} "
      f"{max(r[4]-r[3] for r in rows):+.4f}; best {min(rows, key=lambda r: r[4]-r[3])[0]} {min(r[4]-r[3] for r in rows):+.4f}")
    P(f"   rule 'fill helps on 12 clips' (mean delta < 0 AND >= 9/12 better): "
      f"{'YES' if (n == 12 and md < 0 and better >= 9) else 'NO'}")
    P(f"E2 deliverable+A vs deployed origin: mean delta {ma-mo:+.4f}, better on {sum(r[4] < r[2] for r in rows)}/{n}; "
      f"unfilled deliverable vs origin: {mdl-mo:+.4f}, better on {sum(r[3] < r[2] for r in rows)}/{n}")
    # Spearman rank correlation (average ranks for ties)
    def ranks(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            for k in range(i, j + 1):
                r[idx[k]] = (i + j) / 2.0
            i = j + 1
        return r
    xs = [r[1] for r in rows]
    ys = [r[4] - r[3] for r in rows]
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    P(f"E3 Spearman(crack fraction, delta A-deliv) over {n} clips: {cov / (sx * sy):+.3f}  "
      f"(negative = more cracks -> larger gain)")
P("\nPROVENANCE: " + "; ".join(sorted(set(src.values()))))
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
