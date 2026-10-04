#!/usr/bin/env python
"""Build a scored table + pre-registered criteria verdicts from score_clip_ll.py ROW lines.

usage: table_v1.py OUT.txt SPEC.json SCORES.txt [SCORES.txt ...]

SPEC.json keys
  title        str
  clips        [clip ids]
  origin_ref   label of the deployed origin row (delta / clips-improved / worst-clip reference)
  rows         [{label, name, ref}]   ref = label of the same model's reference config (or null)
  criteria     [{name, row, ref, mean_max, clip_max, sharp_band (or null), gating}]
  headline     optional {file, map: {label: "row name in TABLE_HEADLINE_12CLIP.txt"}, tol}
A label is the render dir name minus the leading "<clip>_" (score_clip_ll.py's tag = the dir name).
Numbers are taken ONLY from the ROW lines of the given score files; the file each row came from is printed.
"""
import json
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(\S+) dx=(\S+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                 r"gtSharp=(\S+) rightPSNR=(\S+) n=(\S+) path=(\S+)")
OUT, SPEC = sys.argv[1], json.load(open(sys.argv[2]))
FILES = sys.argv[3:]
CLIPS, OREF = SPEC["clips"], SPEC["origin_ref"]
D, GT, SRC = defaultdict(dict), {}, defaultdict(dict)
for f in FILES:
    for line in open(f):
        m = ROW.match(line.strip())
        if not m:
            continue
        cl, tag = m.group(1), m.group(2)
        lab = tag[len(cl) + 1:] if tag.startswith(cl + "_") else tag
        rec = dict(lpips=float(m.group(6)), sharp=float(m.group(7)), leftPSNR=float(m.group(5)),
                   rightPSNR=float(m.group(9)), n=int(m.group(10)), off=(m.group(3), m.group(4)), path=m.group(11))
        if lab in D[cl] and abs(D[cl][lab]["lpips"] - rec["lpips"]) > 0:
            print(f"[warn] {cl} {lab} scored twice with different LPIPS: {D[cl][lab]['lpips']} vs {rec['lpips']}")
        D[cl][lab] = rec
        SRC[cl][lab] = f
        GT[cl] = float(m.group(8))

L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


rows = SPEC["rows"]
P("=" * 130)
P(SPEC["title"])
P("scorer scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=4 (unchanged), FFV1 renders only")
P("=" * 130)
have = [c for c in CLIPS if c in D]
missing = [(c, r["label"]) for c in CLIPS for r in rows if r["label"] not in D.get(c, {})]
if missing:
    P(f"MISSING scores: {missing}")
W = 9


def cells(fn):
    out = []
    for c in have:
        try:
            out.append(fn(c))
        except KeyError:
            out.append(f"{'--':>{W}s}")
    return "".join(out)


P()
P("--- PER CLIP: LPIPS (lower is better) ---")
P(f"  {'row':34s} " + "".join(f"{c:>{W}s}" for c in have))
P(f"  {'GT sharpness':34s} " + cells(lambda c: f"{GT[c]:{W}.4f}"))
for r in rows:
    P(f"  {r['name']:34s} " + cells(lambda c: f"{D[c][r['label']]['lpips']:{W}.4f}"))
P()
P(f"--- PER CLIP: delta vs {OREF} (negative = better) ---")
for r in rows:
    if r["label"] == OREF:
        continue
    P(f"  {r['name']:34s} " + cells(lambda c: f"{D[c][r['label']]['lpips'] - D[c][OREF]['lpips']:+{W}.4f}"))
P()
P("--- PER CLIP: delta vs the SAME MODEL's reference config (ref) ---")
for r in rows:
    if not r.get("ref"):
        continue
    P(f"  {r['name'] + ' - ref':34s} " + cells(
        lambda c: f"{D[c][r['label']]['lpips'] - D[c][r['ref']]['lpips']:+{W}.4f}"))
P()
P("--- PER CLIP: sharpness / GT sharpness (1.0 = matches the real right eye) ---")
for r in rows:
    P(f"  {r['name']:34s} " + cells(lambda c: f"{D[c][r['label']]['sharp'] / GT[c]:{W}.3f}"))
P()
P("--- PER CLIP: sharpness ratio vs the same model's reference config ---")
for r in rows:
    if not r.get("ref"):
        continue
    P(f"  {r['name'] + ' / ref':34s} " + cells(
        lambda c: f"{D[c][r['label']]['sharp'] / D[c][r['ref']]['sharp']:{W}.4f}"))
P()


def full(lab):
    return all(lab in D[c] for c in have)


P("=" * 130)
P(f"MEANS over n={len(have)} clips: {' '.join(have)}")
P("=" * 130)
P(f"  {'row':34s}{'meanLPIPS':>10s}{'d vs ' + OREF[:10]:>17s}{'d vs ref':>10s}{'improved':>9s}"
  f"{'sh/GT':>7s}{'sh/ref':>8s}{'worst vs origin':>18s}{'worst vs ref':>18s}{'rPSNR':>8s}")
om = sum(D[c][OREF]["lpips"] for c in have) / len(have) if full(OREF) else float("nan")
for r in rows:
    lab = r["label"]
    if not full(lab):
        P(f"  {r['name']:34s}  (incomplete)")
        continue
    m = sum(D[c][lab]["lpips"] for c in have) / len(have)
    nimp = sum(1 for c in have if D[c][lab]["lpips"] < D[c][OREF]["lpips"])
    shg = sum(D[c][lab]["sharp"] / GT[c] for c in have) / len(have)
    rp = sum(D[c][lab]["rightPSNR"] for c in have) / len(have)
    wc = max(have, key=lambda c: D[c][lab]["lpips"] - D[c][OREF]["lpips"])
    wd = D[wc][lab]["lpips"] - D[wc][OREF]["lpips"]
    if r.get("ref") and full(r["ref"]):
        mr = sum(D[c][r["ref"]]["lpips"] for c in have) / len(have)
        shr = sum(D[c][lab]["sharp"] / D[c][r["ref"]]["sharp"] for c in have) / len(have)
        wr = max(have, key=lambda c: D[c][lab]["lpips"] - D[c][r["ref"]]["lpips"])
        wrd = D[wr][lab]["lpips"] - D[wr][r["ref"]]["lpips"]
        dref, shref, wref = f"{m - mr:+10.4f}", f"{shr:8.4f}", f"{wr} {wrd:+.4f}"
    else:
        dref, shref, wref = f"{'':>10s}", f"{'':>8s}", ""
    P(f"  {r['name']:34s}{m:10.4f}{m - om:+17.4f}{dref}{nimp:>6d}/{len(have):<2d}{shg:7.3f}{shref}"
      f"{wc + ' ' + format(wd, '+.4f'):>18s}{wref:>18s}{rp:8.3f}")
P()

verdicts = []
P("=" * 130)
P("PRE-REGISTERED CRITERIA (scripts/distill/runs/finalcheck_20261004/speed/PREREG.txt)")
P("=" * 130)
for cr in SPEC.get("criteria", []):
    lab, ref = cr["row"], cr["ref"]
    if not (full(lab) and full(ref)):
        P(f"  {cr['name']}: NOT EVALUATED (missing scores)")
        verdicts.append((cr["name"], "NOT EVALUATED"))
        continue
    d = {c: D[c][lab]["lpips"] - D[c][ref]["lpips"] for c in have}
    md = sum(d.values()) / len(have)
    wc = max(have, key=lambda c: d[c])
    ok_a = md <= cr["mean_max"]
    ok_b = d[wc] <= cr["clip_max"]
    parts = [f"(a) mean delta {md:+.4f} <= +{cr['mean_max']:.3f}: {'PASS' if ok_a else 'FAIL'}",
             f"(b) worst clip {wc} {d[wc]:+.4f} <= +{cr['clip_max']:.3f}: {'PASS' if ok_b else 'FAIL'}"]
    ok = ok_a and ok_b
    if cr.get("sharp_band"):
        lo, hi = cr["sharp_band"]
        shr = sum(D[c][lab]["sharp"] / D[c][ref]["sharp"] for c in have) / len(have)
        ok_c = lo <= shr <= hi
        parts.append(f"(c) mean per-clip sharp ratio {shr:.4f} in [{lo}, {hi}]: {'PASS' if ok_c else 'FAIL'}")
        ok = ok and ok_c
    nb = sum(1 for c in have if d[c] < 0)
    tag = "GATING" if cr.get("gating") else "reported"
    P(f"  [{tag}] {cr['name']}  ({lab} vs {ref}, n={len(have)}, better on {nb}/{len(have)}): "
      f"{'PASS' if ok else 'FAIL'}")
    for p in parts:
        P(f"        {p}")
    verdicts.append((cr["name"], "PASS" if ok else "FAIL"))
P()

hl = SPEC.get("headline")
if hl:
    P("=" * 130)
    P(f"G8 SCORER REPRODUCTION vs {hl['file']} (tolerance +-{hl['tol']})")
    P("=" * 130)
    txt = open(hl["file"]).read().splitlines()
    hdr = next(i for i, s in enumerate(txt) if s.startswith("--- PER CLIP: LPIPS"))
    hclips = txt[hdr + 1].split()[1:]
    allok = True
    for lab, rname in hl["map"].items():
        line = next(s for s in txt[hdr:] if s.strip().startswith(rname))
        vals = [float(x) for x in line.strip()[len(rname):].split("[")[0].split()]
        ref = dict(zip(hclips, vals))
        bad = [(c, ref[c], round(D[c][lab]["lpips"], 4)) for c in have if c in ref and lab in D[c]
               and abs(round(D[c][lab]["lpips"], 4) - ref[c]) > hl["tol"] + 1e-9]
        mean_now = sum(D[c][lab]["lpips"] for c in have) / len(have) if full(lab) else float("nan")
        ok = not bad
        allok &= ok
        P(f"  {lab:34s} headline row '{rname}': per-clip {'MATCH' if ok else 'MISMATCH ' + str(bad)}; "
          f"rescored mean {mean_now:.4f}")
    verdicts.append(("G8 scorer reproduction", "PASS" if allok else "FAIL"))
    P()

P("=" * 130)
P("PROVENANCE (score file per row; render dir root)")
P("=" * 130)
for r in rows:
    files = sorted({SRC[c][r["label"]] for c in have if r["label"] in SRC[c]})
    roots = sorted({D[c][r["label"]]["path"].rsplit("/clips/", 1)[0] for c in have if r["label"] in D[c]})
    P(f"  {r['name']:34s} label={r['label']:30s} roots={roots} scores={files}")
P()
P("VERDICTS: " + "; ".join(f"{n}: {v}" for n, v in verdicts))
open(OUT, "w").write("\n".join(L) + "\n")
print(f"\nwrote {OUT}")
