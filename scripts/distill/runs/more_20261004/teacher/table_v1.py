#!/usr/bin/env python
"""Teacher-lane table: parses the ROW lines of SCORES_*.txt (score_clip_ll.py output), applies the PREREG.txt
rules (F1, F2, P1, P2) and prints per-clip LPIPS / sharp / sharp-vs-GT / rightPSNR with deltas vs s25.

usage: table_v1.py OUT.txt SCORES_a.txt [SCORES_b.txt ...]
Row identity = the render directory name (tag) mapped through LABELS below; the s25 reference = tag <clip>_s25_ll.
The deployed-origin 8-step context row is CITED from TABLE_HEADLINE_12CLIP.txt (not rescored here)."""
import json
import os
import re
import sys

CLIPS = ["0301", "0204", "0052", "0147"]
HEADLINE = "scripts/distill/runs/beyond_distil_mamba_scaled/TABLE_HEADLINE_12CLIP.txt"
ORIGIN8 = {"0301": 0.4351, "0204": 0.2053, "0052": 0.4467, "0147": 0.5269}       # TABLE_HEADLINE row 'origin (deployed 8 steps)'
S25_HEADLINE = {"0301": 0.3986, "0204": 0.1897, "0052": 0.4384, "0147": 0.5220}  # TABLE_HEADLINE row 'origin + s25'
LABELS = [("s25_ll", "s25  Euler N=25 (reference, 25x2)"),
          ("gate_euler25", "gate: my Euler N=25 (== s25 md5)"),
          ("H13", "H13  Heun N=13 (25 evals)"),
          ("D16", "D16  DPM++2M N=16 (16 evals)"),
          ("D25", "D25  DPM++2M N=25 (25 evals)"),
          ("C25b", "C25b Euler+churn S_churn=10 (25)"),
          ("C25", "C25  Euler+churn S_churn=5 (25)"),
          ("X1D50", "X1   DPM++2M N=50 (50 evals, converged)"),
          ("X1s50", "X1   Euler N=50 (50 evals, paired)"),
          ("X1s50nat", "X1   Euler N=50 (50 evals, unpaired)")]


def label_of(tag):
    clip, rest = tag.split("_", 1)
    for key, name in LABELS:
        if rest == key:
            return clip, key, name
    return clip, rest, rest


def main():
    out = sys.argv[1]
    rows = {}
    srcs = {}
    for f in sys.argv[2:]:
        for ln in open(f):
            if not ln.startswith("ROW "):
                continue
            kv = dict(re.findall(r"(\w+)=(\S+)", ln))
            clip, key, name = label_of(kv["tag"])
            rows[(key, clip)] = dict(lpips=float(kv["lpips"]), sharp=float(kv["sharp"]), gt=float(kv["gtSharp"]),
                                    rpsnr=float(kv["rightPSNR"]), n=int(kv["n"]), left=float(kv["leftPSNR"]),
                                    path=kv["path"])
            srcs[(key, clip)] = f
    keys = [k for k, _ in LABELS if any((k, c) in rows for c in CLIPS)]
    name = dict(LABELS)
    L = []
    P = L.append
    P("=" * 118)
    P("TEACHER LANE -- origin UNet, guidance 1.01, lossless FFV1, real-GT LPIPS (score_clip_ll.py SCORE_STEP=4)")
    P("regime clips 0301 0204 0052 0147; every candidate RNG-paired with s25 (pad to 25 draws/window)")
    P("=" * 118)
    P("")
    P(f"{'row':38s} " + " ".join(f"{c:>8s}" for c in CLIPS) + "   mean(avail)  n")
    gt = {c: rows[("s25_ll", c)]["gt"] for c in CLIPS if ("s25_ll", c) in rows}
    P(f"{'GT sharpness':38s} " + " ".join(f"{gt.get(c, float('nan')):8.4f}" for c in CLIPS))
    P(f"{'origin deployed 8 steps (CITED headline)':38s} " + " ".join(f"{ORIGIN8[c]:8.4f}" for c in CLIPS)
      + f"   {sum(ORIGIN8.values()) / 4:.4f}      4")
    for k in keys:
        v = [rows[(k, c)]["lpips"] if (k, c) in rows else None for c in CLIPS]
        av = [x for x in v if x is not None]
        P(f"{name[k][:38]:38s} " + " ".join(f"{x:8.4f}" if x is not None else f"{'-':>8s}" for x in v)
          + f"   {sum(av) / len(av):.4f}     {len(av):2d}")
    P("")
    P("--- delta LPIPS vs s25 (negative = better than s25) ---")
    for k in keys:
        if k == "s25_ll":
            continue
        d = [rows[(k, c)]["lpips"] - rows[("s25_ll", c)]["lpips"] if (k, c) in rows and ("s25_ll", c) in rows else None
             for c in CLIPS]
        av = [x for x in d if x is not None]
        P(f"{name[k][:38]:38s} " + " ".join(f"{x:+8.4f}" if x is not None else f"{'-':>8s}" for x in d)
          + (f"   {sum(av) / len(av):+.4f}     {len(av):2d}" if av else ""))
    P("")
    P("--- sharpness / GT sharpness (1.0 = matches the real right eye) ---")
    for k in keys:
        v = [rows[(k, c)]["sharp"] / rows[(k, c)]["gt"] if (k, c) in rows else None for c in CLIPS]
        P(f"{name[k][:38]:38s} " + " ".join(f"{x:8.3f}" if x is not None else f"{'-':>8s}" for x in v))
    P("")
    P("--- right-eye PSNR vs GT (dB; supplementary) ---")
    for k in keys:
        v = [rows[(k, c)]["rpsnr"] if (k, c) in rows else None for c in CLIPS]
        P(f"{name[k][:38]:38s} " + " ".join(f"{x:8.3f}" if x is not None else f"{'-':>8s}" for x in v))
    P("")
    P("=" * 118)
    P("PRE-REGISTERED RULES (PREREG.txt)")
    P("=" * 118)
    s25 = {c: rows[("s25_ll", c)] for c in CLIPS if ("s25_ll", c) in rows}
    # I6
    i6 = [(c, s25[c]["lpips"], S25_HEADLINE[c], abs(s25[c]["lpips"] - S25_HEADLINE[c]) <= 0.0001 + 1e-9) for c in s25]
    P("I6 scorer reproduction of the s25 headline row (+-0.0001): "
      + "  ".join(f"{c} {a:.4f} vs {b:.4f} {'OK' if ok else 'FAIL'}" for c, a, b, ok in i6))
    over = [c for c in CLIPS if c in s25 and s25[c]["sharp"] / s25[c]["gt"] > 1.0]
    P(f"P2 overshoot clips (s25 sharp/GT > 1, from this rescore): {over}")
    for k in keys:
        if k in ("s25_ll", "gate_euler25"):
            continue
        if k.startswith("X1"):
            P(f"  {name[k]}")
            P("      secondary anchor (PREREG X1 / ADDENDUM 2) -- NON-GATING, not a candidate; delta vs s25 reported above")
            continue
        res = []
        if ("0301" in s25) and (k, "0301") in rows:
            d0 = rows[(k, "0301")]["lpips"] - s25["0301"]["lpips"]
            res.append(f"F1 0301 delta {d0:+.4f} <= -0.003: {'PASS' if d0 <= -0.003 else 'FAIL (futility)'}")
        if ("0052" in s25) and (k, "0052") in rows:
            d2 = rows[(k, "0052")]["lpips"] - s25["0052"]["lpips"]
            r_c = rows[(k, "0052")]["sharp"] / rows[(k, "0052")]["gt"]
            r_s = s25["0052"]["sharp"] / s25["0052"]["gt"]
            ok = d2 <= 0.002 and r_c <= r_s + 0.02
            res.append(f"F2 0052 delta {d2:+.4f} (<= +0.002) sh/GT {r_c:.3f} vs s25 {r_s:.3f}+0.02: "
                       f"{'PASS' if ok else 'FAIL (futility)'}")
        have = all((k, c) in rows for c in CLIPS) and all(c in s25 for c in CLIPS)
        if have:
            m = sum(rows[(k, c)]["lpips"] for c in CLIPS) / 4 - sum(s25[c]["lpips"] for c in CLIPS) / 4
            p1 = m <= -0.005
            p2c = [(c, rows[(k, c)]["sharp"] / rows[(k, c)]["gt"], s25[c]["sharp"] / s25[c]["gt"]) for c in over]
            p2 = all(a <= b + 0.02 for _, a, b in p2c)
            mo_c = sum(max(0.0, rows[(k, c)]["sharp"] / rows[(k, c)]["gt"] - 1) for c in CLIPS) / 4
            mo_s = sum(max(0.0, s25[c]["sharp"] / s25[c]["gt"] - 1) for c in CLIPS) / 4
            res.append(f"P1 mean4 delta {m:+.4f} <= -0.005: {'PASS' if p1 else 'FAIL'}")
            res.append("P2 " + ", ".join(f"{c} {a:.3f} <= {b:.3f}+0.02" for c, a, b in p2c)
                       + f": {'PASS' if p2 else 'FAIL'}  (mean overshoot cand {mo_c:.3f} vs s25 {mo_s:.3f})")
            res.append(f"VERDICT {'PASS' if (p1 and p2) else 'FAIL'}")
        else:
            res.append(f"VERDICT: not on all 4 clips ({[c for c in CLIPS if (k, c) in rows]}) -> cannot PASS")
        P(f"  {name[k]}")
        for r in res:
            P(f"      {r}")
    P("")
    P("PROVENANCE (score file per row)")
    for (k, c), f in sorted(srcs.items()):
        P(f"  {k:14s} {c}  {f}  {rows[(k, c)]['path']}")
    txt = "\n".join(L) + "\n"
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    open(out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
