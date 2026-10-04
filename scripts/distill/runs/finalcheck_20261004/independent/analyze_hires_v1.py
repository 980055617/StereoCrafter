#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- R2 (re-score of validate's 16 hi-res renders) and H2 (deliverable T5@1.00
RNG-padded at hi-res) analysis.  CPU only.
usage: analyze_hires_v1.py OUT.txt SCORES_R2.txt [SCORES_H2.txt]
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
            out[(d["clip"], d["tag"][len(d["clip"]) + 1:])] = dict(dy=int(d["dy"]), dx=int(d["dx"]), left=float(d["leftPSNR"]),
                                                                   lp=float(d["lpips"]), sh=float(d["sharp"]), gs=float(d["gtSharp"]),
                                                                   n=int(d["n"]), path=d["path"])
    return out


def shape(path):
    d = os.path.dirname(path)
    for line in open(os.path.join(d, "writer_md5.txt")):
        if "_sbs" in line:
            return re.search(r"\(.*?\)", line).group(0)


mine = rows(sys.argv[2])
lane = rows("outputs/finalcheck_20261004/validate/v2_scores/V2_hires_scores.txt")
P("=" * 118)
P("R2 -- independent re-score of validate's 16 hi-res renders (score_clip_ll.py unchanged, SCORE_STEP=4); source " + sys.argv[2])
P("=" * 118)
nm = 0; mx = 0.0
for k, m in sorted(mine.items()):
    r = lane.get(k)
    if r is None:
        P(f"  no validate ROW for {k}"); continue
    d = abs(m["lp"] - r["lp"]); mx = max(mx, d)
    ok = d <= 1e-4 and abs(m["sh"] - r["sh"]) <= 1e-4 and (m["dy"], m["dx"], m["n"]) == (r["dy"], r["dx"], r["n"])
    nm += ok
    if not ok:
        P(f"  MISMATCH {k}: validate {r['lp']:.6f} ({r['dy']},{r['dx']},n{r['n']}) vs mine {m['lp']:.6f} ({m['dy']},{m['dx']},n{m['n']})")
P(f"  reproduced {nm}/{len(mine)} validate ROW lines (max |dLPIPS| {mx:.6f})")
if len(sys.argv) > 3:
    mine.update(rows(sys.argv[3]))
P("\nPER RESOLUTION (deltas vs origin 8x2@1.01 at the same resolution; pair check = same (dy,dx), n, leftPSNR, writer shape)")
verdict = {}
for res in RES:
    P(f"\n  --- {res} ---")
    P(f"  {'clip':6s} {'offset':>9s} {'n':>3s} {'LPIPS origin':>13s} {'deliv 8x2':>10s} {'d':>8s} {'T5@1.00 pad':>12s} {'d vs origin':>12s} {'d vs deliv':>11s}  sharp/GT o/d/T5   pair")
    have_t5 = all((c, f"deliv_g100_T5pad_{res}") in mine for c in CL)
    D, DT, DTD = [], [], []
    for c in CL:
        o, dl = mine[(c, f"origin_ll_{res}")], mine[(c, f"deliv_ll_{res}")]
        t = mine.get((c, f"deliv_g100_T5pad_{res}"))
        group = [o, dl] + ([t] if t else [])
        pair_ok = len({(g["dy"], g["dx"], g["n"], round(g["left"], 4)) for g in group}) == 1 and len({shape(g["path"]) for g in group}) == 1
        D.append(dl["lp"] - o["lp"])
        line = f"  {c:6s} {'(%d,%d)' % (o['dy'], o['dx']):>9s} {o['n']:3d} {o['lp']:13.4f} {dl['lp']:10.4f} {dl['lp'] - o['lp']:+8.4f}"
        if t:
            DT.append(t["lp"] - o["lp"]); DTD.append(t["lp"] - dl["lp"])
            line += f" {t['lp']:12.4f} {t['lp'] - o['lp']:+12.4f} {t['lp'] - dl['lp']:+11.4f}"
            line += f"  {o['sh'] / o['gs']:.3f}/{dl['sh'] / dl['gs']:.3f}/{t['sh'] / t['gs']:.3f}"
        else:
            line += f" {'-':>12s} {'-':>12s} {'-':>11s}  {o['sh'] / o['gs']:.3f}/{dl['sh'] / dl['gs']:.3f}/-"
        P(line + f"   {'OK' if pair_ok else 'PAIR FAIL'}")
    mo = st.mean(mine[(c, f'origin_ll_{res}')]['lp'] for c in CL); md = st.mean(mine[(c, f'deliv_ll_{res}')]['lp'] for c in CL)
    tr = (md < mo) and max(D) <= 0.005
    P(f"  4-clip mean: origin {mo:.4f}  deliverable 8x2 {md:.4f}  delta {md - mo:+.4f}  improved {sum(x < 0 for x in D)}/4  "
      f"worst {max(D):+.4f} -> validate transfer rule: {'TRANSFERS' if tr else 'DOES NOT TRANSFER'}")
    verdict[res] = dict(deliv_transfers=tr)
    if have_t5:
        mt = st.mean(mine[(c, f'deliv_g100_T5pad_{res}')]['lp'] for c in CL)
        i = (mt < mo) and max(DT) <= 0.005
        ii = st.mean(DTD) <= 0.002 and max(DTD) <= 0.005
        P(f"  T5@1.00 4-clip mean {mt:.4f}: (i) vs origin delta {mt - mo:+.4f}, improved {sum(x < 0 for x in DT)}/4, worst {max(DT):+.4f} -> "
          f"{'ok' if i else 'FAIL'};  (ii) vs deliverable 8x2 mean {st.mean(DTD):+.4f}, worst {max(DTD):+.4f} -> {'ok' if ii else 'FAIL'}"
          f"  => T5@1.00 {'KEEPS' if (i and ii) else 'DOES NOT KEEP'} THE HI-RES QUALITY CLAIM at {res}")
        verdict[res].update(t5_keeps=(i and ii), t5_mean=mt, origin_mean=mo, deliv_mean=md)
open(outp, "w").write("\n".join(L) + "\n")
json.dump(dict(verdict=verdict, rows={f"{k[0]}|{k[1]}": v for k, v in mine.items()}), open(outp.replace(".txt", ".json"), "w"), indent=1)
print("\n".join(L))
