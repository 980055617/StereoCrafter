#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- hi-res analysis v2 (CPU only).
v2 of analyze_hires_v1.py (never run for an official output): merges ALL given score files FIRST, then
  (1) reproduction: compares ONLY validate's renders (origin_ll_*, deliv_ll_*) with validate's
      outputs/finalcheck_20261004/validate/v2_scores/V2_hires_scores.txt ROW lines;
  (2) per resolution: validate's transfer rule for the deliverable 8x2, and this lane's H2 rule for the deliverable
      T5@1.00 RNG-padded renders (PREREG.txt section H2), with pair checks.
usage: analyze_hires_v2.py OUT.txt SCORES_HIRES_1792.txt SCORES_HIRES_1920.txt
"""
import json, os, re, statistics as st, sys

os.chdir("/home/kawa/master_project/StereoCrafter")
outp = sys.argv[1]
assert not os.path.exists(outp), "refusing to overwrite"
CL = ["0170", "0204", "0042", "0052"]
RES = ["1024x1792", "1024x1920"]
L = []; P = L.append


def rows(path):
    out = {}
    for line in open(path):
        if line.startswith("ROW "):
            d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            k = (d["clip"], d["tag"][len(d["clip"]) + 1:])
            assert k not in out, f"duplicate {k} in {path}"
            out[k] = dict(dy=int(d["dy"]), dx=int(d["dx"]), left=float(d["leftPSNR"]), lp=float(d["lpips"]),
                          sh=float(d["sharp"]), gs=float(d["gtSharp"]), n=int(d["n"]), path=d["path"], src=path)
    return out


def shape(path):
    for line in open(os.path.join(os.path.dirname(path), "writer_md5.txt")):
        if "_sbs" in line:
            return re.search(r"\(.*?\)", line).group(0)


mine = {}
for f in sys.argv[2:]:
    r = rows(f)
    assert not (set(r) & set(mine)), "overlapping keys across score files"
    mine.update(r)
lane = rows("outputs/finalcheck_20261004/validate/v2_scores/V2_hires_scores.txt")
P("=" * 118)
P("HI-RES -- this lane's re-score (score_clip_ll.py unchanged, SCORE_STEP=4) of validate's 16 renders + this lane's 8")
P("deliverable T5@1.00 RNG-padded renders.  Sources: " + ", ".join(sys.argv[2:]))
P("=" * 118)
P("\n[1] reproduction of validate's V2_hires_scores.txt ROW lines (validate renders only; |dLPIPS| <= 0.0001, same offset/n)")
nm = 0; nc = 0; mx = 0.0
for k, r in sorted(lane.items()):
    if not re.match(r"(origin|deliv)_ll_1024x(1792|1920)$", k[1]):
        continue
    m = mine.get(k)
    if m is None:
        P(f"  NOT RE-SCORED {k}"); continue
    nc += 1
    d = abs(m["lp"] - r["lp"]); mx = max(mx, d)
    ok = d <= 1e-4 and abs(m["sh"] - r["sh"]) <= 1e-4 and (m["dy"], m["dx"], m["n"]) == (r["dy"], r["dx"], r["n"])
    nm += ok
    if not ok:
        P(f"  MISMATCH {k}: validate {r['lp']:.6f} ({r['dy']},{r['dx']},n{r['n']}) vs mine {m['lp']:.6f} ({m['dy']},{m['dx']},n{m['n']})")
P(f"  reproduced {nm}/{nc} (max |dLPIPS| {mx:.6f})")
P("\n[2] PER RESOLUTION (pair check = identical (dy,dx), n, leftPSNR and writer shape for origin / deliverable 8x2 / T5@1.00)")
verdict = {}
for res in RES:
    P(f"\n  --- {res} ---")
    P(f"  {'clip':6s} {'offset':>9s} {'n':>3s} {'origin 8x2':>11s} {'deliv 8x2':>10s} {'d':>8s} {'T5@1.00 pad':>12s} {'T5-origin':>10s} {'T5-deliv':>9s}  sharp/GT o/d/T5      pair")
    D, DT, DTD = [], [], []
    for c in CL:
        o, dl, t = mine[(c, f"origin_ll_{res}")], mine[(c, f"deliv_ll_{res}")], mine[(c, f"deliv_g100_T5pad_{res}")]
        g = [o, dl, t]
        pair_ok = len({(x["dy"], x["dx"], x["n"], round(x["left"], 4)) for x in g}) == 1 and len({shape(x["path"]) for x in g}) == 1
        D.append(dl["lp"] - o["lp"]); DT.append(t["lp"] - o["lp"]); DTD.append(t["lp"] - dl["lp"])
        P(f"  {c:6s} {'(%d,%d)' % (o['dy'], o['dx']):>9s} {o['n']:3d} {o['lp']:11.4f} {dl['lp']:10.4f} {dl['lp'] - o['lp']:+8.4f} "
          f"{t['lp']:12.4f} {t['lp'] - o['lp']:+10.4f} {t['lp'] - dl['lp']:+9.4f}  {o['sh'] / o['gs']:.3f}/{dl['sh'] / dl['gs']:.3f}/{t['sh'] / t['gs']:.3f}"
          f"   {'OK ' + shape(o['path']) if pair_ok else 'PAIR FAIL'}")
    mo = st.mean(mine[(c, f"origin_ll_{res}")]["lp"] for c in CL)
    md = st.mean(mine[(c, f"deliv_ll_{res}")]["lp"] for c in CL)
    mt = st.mean(mine[(c, f"deliv_g100_T5pad_{res}")]["lp"] for c in CL)
    tr = (md < mo) and max(D) <= 0.005
    i = (mt < mo) and max(DT) <= 0.005
    ii = st.mean(DTD) <= 0.002 and max(DTD) <= 0.005
    P(f"  4-clip means: origin {mo:.4f} | deliverable 8x2 {md:.4f} ({md - mo:+.4f}, {sum(x < 0 for x in D)}/4, worst {max(D):+.4f}) | "
      f"T5@1.00 {mt:.4f} ({mt - mo:+.4f} vs origin, {sum(x < 0 for x in DT)}/4, worst {max(DT):+.4f}; {mt - md:+.4f} vs deliverable 8x2, worst {max(DTD):+.4f})")
    P(f"  validate transfer rule (deliverable 8x2): {'TRANSFERS' if tr else 'DOES NOT TRANSFER'}")
    P(f"  H2 rule (T5@1.00): (i) vs origin {'ok' if i else 'FAIL'}; (ii) vs deliverable 8x2 {'ok' if ii else 'FAIL'} "
      f"=> T5@1.00 {'KEEPS' if (i and ii) else 'DOES NOT KEEP'} THE HI-RES QUALITY CLAIM at {res}")
    verdict[res] = dict(origin=mo, deliv=md, t5=mt, deliv_transfers=tr, t5_keeps=(i and ii),
                        t5_minus_origin=mt - mo, t5_minus_deliv=mt - md, worst_t5_minus_origin=max(DT), worst_t5_minus_deliv=max(DTD),
                        deliv_improved=sum(x < 0 for x in D), t5_improved=sum(x < 0 for x in DT))
open(outp, "w").write("\n".join(L) + "\n")
json.dump(dict(verdict=verdict, rows={f"{k[0]}|{k[1]}": v for k, v in mine.items()}), open(outp.replace(".txt", ".json"), "w"), indent=1)
print("\n".join(L))
