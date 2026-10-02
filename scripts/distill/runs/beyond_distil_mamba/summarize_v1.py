#!/usr/bin/env python
"""Build the readout tables from score_clip_ll.py ROW lines (LOSSLESS FFV1 only).

usage: summarize_v1.py <out.txt> <scores1.txt> [scores2.txt ...]
env    SUM_CLIPS  ordered clip list for the means (default 0301,0204,0052,0147)
"""
import os, re, sys, collections

out_path = sys.argv[1]
files = sys.argv[2:]
CLIPS = os.environ.get("SUM_CLIPS", "0301,0204,0052,0147").split(",")

ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(\S+) dx=(\S+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                 r"gtSharp=(\S+) rightPSNR=(\S+) n=(\S+) path=(\S+)")
D = collections.defaultdict(dict)          # clip -> label -> dict
GT = {}
for f in files:
    for line in open(f):
        m = ROW.match(line.strip())
        if not m:
            continue
        cl, tag, dy, dx, lp, lpips, sh, gts, rp, n, path = m.groups()
        lab = tag[len(cl) + 1:] if tag.startswith(cl + "_") else tag
        D[cl][lab] = dict(lpips=float(lpips), sharp=float(sh), leftPSNR=float(lp), rightPSNR=float(rp),
                          n=int(n), off=f"({dy},{dx})", path=path)
        GT[cl] = float(gts)

PRETTY = [("origin_ll", "origin"),
          ("s25_ll", "origin+s25"),
          ("student_ll", "origin+student (ref)"),
          ("mamba_ll", "mamba 5-slot (SHIPPED)"),
          ("mamba_s25_ll", "mamba+s25"),
          ("mamba_oracleALL_ll", "mamba+oracle(all 8)"),
          ("mamba_oracle456_ll", "mamba+oracle456")]
extra = sorted({l for cl in D for l in D[cl] if l.startswith("mstudent")},
               key=lambda s: (s.split("_step")[0], int(re.search(r"step(\d+)", s).group(1))))
PRETTY += [(l, "mamba+" + l.replace("_ll", "")) for l in extra]

L = []
def P(s=""):
    L.append(s)
    print(s, flush=True)

P("=" * 108)
P("MAMBA-SIDE BEYOND-DISTIL -- ALL ROWS LOSSLESS (FFV1), real-GT LPIPS, deployed config 8 steps / g 1.01")
P("scorer scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py  SCORE_STEP=4")
P("=" * 108)
for cl in CLIPS:
    if cl not in D:
        continue
    base = D[cl].get("mamba_ll", {}).get("lpips")
    orig = D[cl].get("origin_ll", {}).get("lpips")
    s25 = D[cl].get("mamba_s25_ll", {}).get("lpips")
    head = (base - s25) if (base is not None and s25 is not None) else None
    P(f"--- {cl}  GT sharp {GT.get(cl, float('nan')):.4f} ---")
    P(f"  {'row':26s} {'LPIPS':>8s} {'d vs mamba':>11s} {'frac head':>10s} {'sharp':>8s} "
      f"{'sh/mamba':>9s} {'leftPSNR':>9s} {'rPSNR':>8s} {'nf':>4s}")
    for lab, name in PRETTY:
        r = D[cl].get(lab)
        if not r:
            continue
        dv = "" if base is None else f"{r['lpips'] - base:+.4f}"
        fr = ""
        if head and head > 1e-6 and lab not in ("origin_ll", "s25_ll", "student_ll", "mamba_ll"):
            fr = f"{100.0 * (base - r['lpips']) / head:7.1f}%"
        shr = "" if base is None else f"{r['sharp'] / D[cl]['mamba_ll']['sharp']:9.4f}"
        P(f"  {name:26s} {r['lpips']:8.4f} {dv:>11s} {fr:>10s} {r['sharp']:8.4f} {shr:>9s} "
          f"{r['leftPSNR']:9.2f} {r['rightPSNR']:8.3f} {r['n']:4d}")
    P()

P("=" * 108)
P(f"MEANS over {len([c for c in CLIPS if c in D])} clips: {[c for c in CLIPS if c in D]}")
P("=" * 108)
have = [c for c in CLIPS if c in D]
mb = [D[c]["mamba_ll"]["lpips"] for c in have if "mamba_ll" in D[c]]
base_m = sum(mb) / len(mb) if mb else None
ms25 = [D[c]["mamba_s25_ll"]["lpips"] for c in have if "mamba_s25_ll" in D[c]]
s25_m = sum(ms25) / len(ms25) if ms25 else None
head_m = (base_m - s25_m) if (base_m and s25_m) else None
P(f"  {'row':26s} {'mean LPIPS':>10s} {'d vs mamba':>11s} {'frac of mamba headroom':>23s}  {'n':>3s}")
for lab, name in PRETTY:
    v = [D[c][lab]["lpips"] for c in have if lab in D[c]]
    if not v:
        continue
    m = sum(v) / len(v)
    dv = "" if base_m is None else f"{m - base_m:+.4f}"
    fr = ""
    if head_m and len(v) == len(mb) and lab not in ("origin_ll", "s25_ll", "student_ll", "mamba_ll"):
        fr = f"{100.0 * (base_m - m) / head_m:22.1f}%"
    P(f"  {name:26s} {m:10.4f} {dv:>11s} {fr:>23s}  {len(v):3d}")
P()
if head_m:
    P(f"  mamba-side headroom on this set: mamba {base_m:.4f} -> mamba+s25 {s25_m:.4f} "
      f"= {base_m - s25_m:+.4f}")
open(out_path, "w").write("\n".join(L) + "\n")
