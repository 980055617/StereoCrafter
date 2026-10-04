#!/usr/bin/env python
"""Build the stripes-lane table and apply the PREREG.txt criteria (P1-P3, K5 LPIPS reproduction).

usage: table_v1.py OUT.txt --scores S1 [S2 ...] --decomp D1.json [D2.json ...] [--temporal T1.json ...]
                   [--clips 0301,0204,0052,0147]
Every number printed is read from the given files (score ROW lines, decomposition JSON rows, temporal JSON rows);
the file each row came from is listed in the PROVENANCE block.
"""
import json
import os
import re
import sys
from collections import defaultdict

os.chdir("/home/kawa/master_project/StereoCrafter")
argv = sys.argv[1:]
OUT = argv[0]
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}")
groups = defaultdict(list)
cur = None
for a in argv[1:]:
    if a.startswith("--"):
        cur = a[2:]
        continue
    groups[cur].append(a)
CLIPS = (groups.get("clips") or ["0301,0204,0052,0147"])[0].split(",")

CONFIGS = [("origin", "origin_ll", "origin_g101_s8", "origin 8x2 @1.01 (deployed)"),
           ("deliv", "deliv_g100_T5nat", "deliv_g100_T5nat", "deliverable T5 @1.00 unpadded")]
VARIANTS = [("A_rowlin", "A rowlin/keep"), ("B_telea", "B telea/keep"), ("C_rowlinshrink", "C rowlin/shrink")]
PUBLISHED = {  # K5: TABLE_HEADLINE_12CLIP.txt (origin_ll) and TABLE_POSTHOC_T5G100_UNPADDED.txt (deliv T5nat)
    "origin_ll": {"0301": 0.4351, "0204": 0.2053, "0052": 0.4467, "0147": 0.5269},
    "deliv_g100_T5nat": {"0301": 0.3994, "0204": 0.1874, "0052": 0.4399, "0147": 0.5251},
}
TEMPORAL_PUB = {"0301": 0.0493, "0204": 0.0105, "0052": 0.0217, "0147": 0.0083}   # V1_TEMPORAL_576_12CLIP.txt origin

lp, sh, gsh, prov = {}, {}, {}, defaultdict(set)
for f in groups.get("scores", []):
    for line in open(f):
        if line.startswith("ROW "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            tag = kv["tag"]
            lab = tag.split("_", 1)[1]
            key = (kv["clip"], lab)
            if key in lp and abs(lp[key] - float(kv["lpips"])) > 1e-9:
                sys.exit(f"conflicting LPIPS for {key} in {f}")
            lp[key] = float(kv["lpips"]); sh[key] = float(kv["sharp"]); gsh[kv["clip"]] = float(kv["gtSharp"])
            prov[lab].add(f)
dc = {}
for f in groups.get("decomp", []):
    d = json.load(open(f))
    for clip, v in d.items():
        for lab, row in v["rows"].items():
            dc[(clip, lab)] = row
            prov["decomp:" + lab.split(" ")[0]].add(f)
tp = {}
for f in groups.get("temporal", []):
    d = json.load(open(f))
    for clip, v in d.items():
        for tag, row in v.items():
            lab = "GT" if tag == "GT" else tag.split("_", 1)[1]
            tp[(clip, lab)] = row
            prov["temporal:" + lab].add(f)

L = []
P = L.append
P("=" * 118)
P("STRIPES LANE -- crack fill of the warped right-eye input, 4 regime clips " + " ".join(CLIPS))
P("LPIPS: score_clip_ll.py SCORE_STEP=4 (FFV1 only).  Decomposition: reviewlib operators, whole 576x1024 window,")
P("registered GT regions, frames step 8 (n=19).  Temporal: score_temporal_ll.py (RAFT flow on the GT window).")
P("All deltas/ratios are a variant vs ITS OWN config's unfilled baseline (RNG-paired: same initial noise).")
P("=" * 118)


def g(dct, key, fld=None):
    v = dct.get(key)
    if v is None:
        return None
    return v if fld is None else v.get(fld)


def fmt(x, w=8, p=4):
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and x == x else f"{'n/a':>{w}s}"


# ---- K5 LPIPS reproduction
P("\nK5 scorer reproduction (re-scored baselines vs published per-clip LPIPS, tolerance +-0.0001):")
k5_ok = True
for lab, pub in PUBLISHED.items():
    for c in CLIPS:
        v = lp.get((c, lab))
        ok = v is not None and abs(round(v, 4) - pub[c]) <= 0.0001 + 1e-12
        k5_ok &= ok
        P(f"  {lab:20s} {c}  ours {fmt(v, 8, 6)}  published {pub[c]:.4f}  {'PASS' if ok else 'FAIL'}")
P(f"  K5 LPIPS: {'PASS' if k5_ok else 'FAIL'}")
P("K6 temporal reproduction (origin_ll warp error vs V1_TEMPORAL_576_12CLIP.txt, 4 decimals):")
k6_ok = True
for c in CLIPS:
    v = g(tp, (c, "origin_ll"), "warp")
    ok = v is not None and round(v, 4) == TEMPORAL_PUB[c]
    k6_ok &= ok
    P(f"  {c}  ours {fmt(v, 8, 5)}  published {TEMPORAL_PUB[c]:.4f}  {'PASS' if ok else ('n/a' if v is None else 'FAIL')}")
P(f"  K6 temporal: {'PASS' if k6_ok else 'FAIL or incomplete'}")

verdicts = {}
for ck, base, vpref, cname in CONFIGS:
    P("\n" + "=" * 118)
    P(f"CONFIG {cname}   baseline label {base}")
    P("=" * 118)
    hdr = f"  {'row':34s}" + "".join(f"{c:>10s}" for c in CLIPS) + f"{'MEAN':>10s}"
    rows_all = [(base, "baseline (unfilled)")] + [(f"{vpref}_{v}", vn) for v, vn in VARIANTS]
    for title, fn, prec in (
        ("LPIPS (lower is better)", lambda c, lab: lp.get((c, lab)), 4),
        ("delta LPIPS vs baseline", lambda c, lab: (lp[(c, lab)] - lp[(c, base)]) if (c, lab) in lp and (c, base) in lp else None, 4),
        ("sharp / GT sharp", lambda c, lab: sh[(c, lab)] / gsh[c] if (c, lab) in sh else None, 3),
        ("stripeE (flat-region |dx|)", lambda c, lab: g(dc, (c, lab), "stripeE"), 5),
        ("stripeE / baseline", lambda c, lab: g(dc, (c, lab), "stripeE") / g(dc, (c, base), "stripeE") if g(dc, (c, lab)) and g(dc, (c, base)) else None, 4),
        ("  stripeE near holes / baseline", lambda c, lab: g(dc, (c, lab), "stripeE_nearHole") / g(dc, (c, base), "stripeE_nearHole") if g(dc, (c, lab)) and g(dc, (c, base)) else None, 4),
        ("  stripeE far from holes / baseline", lambda c, lab: g(dc, (c, lab), "stripeE_farHole") / g(dc, (c, base), "stripeE_farHole") if g(dc, (c, lab)) and g(dc, (c, base)) else None, 4),
        ("stripeE / GT", lambda c, lab: g(dc, (c, lab), "stripeE") / g(dc, (c, "GT"), "stripeE") if g(dc, (c, lab)) else None, 3),
        ("edgeHF (detail at GT edges)", lambda c, lab: g(dc, (c, lab), "edgeHF"), 5),
        ("edgeHF / baseline", lambda c, lab: g(dc, (c, lab), "edgeHF") / g(dc, (c, base), "edgeHF") if g(dc, (c, lab)) and g(dc, (c, base)) else None, 4),
        ("edgeHF / GT", lambda c, lab: g(dc, (c, lab), "edgeHF") / g(dc, (c, "GT"), "edgeHF") if g(dc, (c, lab)) else None, 3),
        ("flatHF / baseline", lambda c, lab: g(dc, (c, lab), "flatHF") / g(dc, (c, base), "flatHF") if g(dc, (c, lab)) and g(dc, (c, base)) else None, 4),
        ("haloFrac %", lambda c, lab: g(dc, (c, lab), "haloFrac"), 2),
        ("warp error (RAFT, GT flow)", lambda c, lab: g(tp, (c, lab), "warp"), 5),
        ("warp / baseline", lambda c, lab: g(tp, (c, lab), "warp") / g(tp, (c, base), "warp") if g(tp, (c, lab)) and g(tp, (c, base)) else None, 4),
        ("tLP", lambda c, lab: g(tp, (c, lab), "tLP"), 4),
        ("seam ratio", lambda c, lab: (g(tp, (c, lab), "seam") / g(tp, (c, lab), "nonseam")) if g(tp, (c, lab)) else None, 3),
    ):
        P(f"\n--- {title} ---")
        P(hdr)
        for lab, nm in rows_all:
            vals = [fn(c, lab) for c in CLIPS]
            ok = [v for v in vals if v is not None]
            mean = sum(ok) / len(ok) if len(ok) == len(CLIPS) else None
            P(f"  {nm:34s}" + "".join(fmt(v, 10, prec) for v in vals) + fmt(mean, 10, prec))
    # ---- criteria
    P(f"\n--- PRE-REGISTERED CRITERIA (PREREG.txt), {cname} ---")
    for v, vn in VARIANTS:
        lab = f"{vpref}_{v}"
        have = all((c, lab) in lp and (c, base) in lp and g(dc, (c, lab)) and g(dc, (c, base)) for c in CLIPS)
        if not have:
            P(f"  {vn:18s} INCOMPLETE (missing scores or decomposition rows)")
            verdicts[(ck, v)] = None
            continue
        dl = [lp[(c, lab)] - lp[(c, base)] for c in CLIPS]
        sr = [g(dc, (c, lab), "stripeE") / g(dc, (c, base), "stripeE") for c in CLIPS]
        er = [g(dc, (c, lab), "edgeHF") / g(dc, (c, base), "edgeHF") for c in CLIPS]
        p1 = sum(d < 0 for d in dl) >= 3
        p2 = sum(r < 1 for r in sr) >= 3 and sum(sr) / 4 < 1.0
        p3 = sum(er) / 4 >= 0.995 and min(er) >= 0.990
        ok = p1 and p2 and p3
        verdicts[(ck, v)] = ok
        md = sum(dl) / 4
        size = "negligible" if abs(md) < 0.0005 else "non-negligible"
        P(f"  {vn:18s} {'PASS' if ok else 'FAIL'}   P1 LPIPS better on {sum(d < 0 for d in dl)}/4 (mean {md:+.4f}, {size}): "
          f"{'PASS' if p1 else 'FAIL'} | P2 stripeE lower on {sum(r < 1 for r in sr)}/4, mean ratio {sum(sr)/4:.4f}: "
          f"{'PASS' if p2 else 'FAIL'} | P3 edgeHF mean ratio {sum(er)/4:.4f}, min {min(er):.4f}: {'PASS' if p3 else 'FAIL'}")
        wr = [g(tp, (c, lab), "warp") / g(tp, (c, base), "warp") for c in CLIPS if g(tp, (c, lab)) and g(tp, (c, base))]
        if len(wr) == len(CLIPS):
            P(f"  {'':18s} temporal: mean warp ratio {sum(wr)/4:.4f} ({'FLAG > +5%' if sum(wr)/4 > 1.05 else 'no flag'}), "
              f"worse on {sum(w > 1 for w in wr)}/4")

P("\n" + "=" * 118)
P("LANE VERDICT (PREREG: YES = a variant passes for BOTH configs; PARTIAL = one config; NO otherwise)")
best = "NO"
for v, vn in VARIANTS:
    a, b = verdicts.get(("origin", v)), verdicts.get(("deliv", v))
    P(f"  {vn:18s} origin: {('PASS' if a else 'FAIL') if a is not None else 'n/a'}   deliverable: {('PASS' if b else 'FAIL') if b is not None else 'n/a'}")
    if a and b:
        best = "YES"
    elif (a or b) and best != "YES":
        best = "PARTIAL"
P(f"  => {best}")
P("\nPROVENANCE (file(s) each label's numbers came from)")
for k in sorted(prov):
    P(f"  {k:44s} {sorted(prov[k])}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
