#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- R1 analysis (CPU only).
Compares this lane's own score_clip_ll.py ROW lines (576x1024, 12 clips) with every number the lanes reported,
runs the pair checks, and recomputes the lanes' pre-registered verdicts from THIS lane's numbers.
usage: analyze_R1_v1.py MY_SCORES.txt P3.json OUT.txt
"""
import json, os, re, sys, statistics as st

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
SP = "scripts/distill/runs/finalcheck_20261004/speed"
LANE_FILES = [f"{SP}/SCORES_STEP1.txt", f"{SP}/SCORES_STEP2_4clip.txt", f"{SP}/SCORES_STEP2_T5G100_4clip.txt",
              f"{SP}/SCORES_STEP2_EXT12.txt", f"{SP}/SCORES_POSTHOC_T5G100NAT.txt",
              "outputs/finalcheck_20261004/validate/v2_scores/V2_576_rescore.txt"]
HEAD = "scripts/distill/runs/beyond_distil_mamba_scaled/TABLE_HEADLINE_12CLIP.txt"


def rows(path):
    out = {}
    for line in open(path):
        if line.startswith("ROW "):
            d = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
            lab = d["tag"][len(d["clip"]) + 1:]
            out.setdefault((d["clip"], lab), []).append(dict(dy=int(d["dy"]), dx=int(d["dx"]), left=float(d["leftPSNR"]),
                                                             lp=float(d["lpips"]), sh=float(d["sharp"]), gs=float(d["gtSharp"]),
                                                             rp=float(d["rightPSNR"]), n=int(d["n"]), path=d["path"], src=path))
    return out


me_all = rows(sys.argv[1])
p3 = {(r["clip"], r["label"]): r for r in json.load(open(sys.argv[2]))}
outp = sys.argv[3]
assert not os.path.exists(outp), "refusing to overwrite"
me = {}
for k, v in me_all.items():
    assert len(v) == 1, f"duplicate ROW for {k}"
    me[k] = v[0]
L = []
P = L.append

# ------------------------------------------------------------------ 1. reproduction vs every lane ROW line
P("=" * 118)
P("R1 -- independent re-score, 576x1024, 12 clips, score_clip_ll.py UNCHANGED (SCORE_STEP=4); source " + sys.argv[1])
P("=" * 118)
P("\n[1] REPRODUCTION of every lane ROW line for the same render path (tolerance |dLPIPS|<=0.0001, |dsharp|<=0.0001)")
nmatch = ncmp = 0; maxd = 0.0; maxds = 0.0; bad = []
for f in LANE_FILES:
    for k, lst in rows(f).items():
        for r in lst:
            if k not in me:
                continue
            m = me[k]
            if os.path.normpath(m["path"]) != os.path.normpath(r["path"]):
                bad.append(f"path differs {k}: {r['path']} vs {m['path']}"); continue
            ncmp += 1
            d, ds = abs(m["lp"] - r["lp"]), abs(m["sh"] - r["sh"])
            maxd, maxds = max(maxd, d), max(maxds, ds)
            same_geo = (m["dy"], m["dx"], m["n"]) == (r["dy"], r["dx"], r["n"]) and abs(m["left"] - r["left"]) < 1e-3
            if d <= 1e-4 and ds <= 1e-4 and same_geo:
                nmatch += 1
            else:
                bad.append(f"MISMATCH {k} lane {r['lp']:.6f}/{r['sh']:.6f} ({r['dy']},{r['dx']},n{r['n']}) [{f}] vs mine "
                           f"{m['lp']:.6f}/{m['sh']:.6f} ({m['dy']},{m['dx']},n{m['n']})")
P(f"  lane ROW lines compared: {ncmp}; reproduced: {nmatch}; max |dLPIPS| {maxd:.6f}; max |dsharp| {maxds:.6f}")
for b in bad:
    P("  " + b)

# headline table rows (4 dp)
hd = open(HEAD).read().splitlines()
def hrow(name):
    for line in hd:
        if line.strip().startswith(name):
            vals = re.findall(r"[-+]?\d+\.\d+", line[len(line) - len(line.lstrip()) + len(name):])
            return [float(v) for v in vals[:12]]
    return None
HMAP = {"origin_ll": "origin (deployed 8 steps)", "mamba_ll": "mamba 5-slot (SHIPPED)", "s25_ll": "origin + s25 (2.4x cost)",
        "mstudent2_step800_deliv_ll": "THIS DELIVERABLE (A)"}
P("\n[1b] REPRODUCTION of TABLE_HEADLINE_12CLIP.txt per-clip LPIPS (printed at 4 dp; |mine - table| <= 0.0001)")
for lab, name in HMAP.items():
    tv = None
    # first occurrence of the name is the LPIPS block
    for line in hd:
        if line.strip().startswith(name):
            tv = [float(v) for v in re.findall(r"\d+\.\d{4}", line)[:12]]; break
    diffs = [abs(round(me[(c, lab)]["lp"], 4) - t) for c, t in zip(CLIPS, tv)]
    P(f"  {lab:28s} max |diff| {max(diffs):.4f} -> {'REPRODUCED' if max(diffs) <= 1e-4 + 1e-9 else 'NOT REPRODUCED'}")

# ------------------------------------------------------------------ 2. pair checks
P("\n[2] PAIR CHECKS per clip: identical (dy,dx), n, leftPSNR across ALL rows; identical writer shape (P3); P3 pass")
labels = sorted({k[1] for k in me})
allok = True
for c in CLIPS:
    ks = [k for k in me if k[0] == c]
    geo = {(me[k]["dy"], me[k]["dx"], me[k]["n"], round(me[k]["left"], 4)) for k in ks}
    shp = {p3[k]["writer_shape"] for k in ks}
    p3ok = all(p3[k]["ok"] for k in ks)
    ok = len(geo) == 1 and len(shp) == 1 and p3ok
    allok &= ok
    g = next(iter(geo))
    P(f"  {c}: rows {len(ks):2d}  offset ({g[0]},{g[1]}) n={g[2]} leftPSNR={g[3]}  shape {sorted(shp)}  P3 all pass={p3ok} -> {'OK' if ok else 'FAIL'}")
P(f"  PAIR CHECK: {'ALL OK' if allok else 'FAILURES ABOVE'}")

# ------------------------------------------------------------------ 3. tables from MY numbers
NAMES = [("origin_ll", "origin 8x2@1.01 (deployed)"), ("origin_g100_s8", "origin 8x1@1.00"),
         ("origin_g101_T6pad", "origin T6@1.01 (pad)"), ("origin_g101_T5pad", "origin T5@1.01 (pad)"),
         ("origin_g100_T5pad", "origin T5@1.00 (pad)"), ("mamba_ll", "shipped 5-slot Mamba 8x2@1.01"),
         ("s25_ll", "origin + 25 steps @1.01"), ("mstudent2_step800_deliv_ll", "DELIVERABLE 8x2@1.01"),
         ("deliv_g100_s8", "deliverable 8x1@1.00"), ("deliv_g101_T6pad", "deliverable T6@1.01 (pad)"),
         ("deliv_g101_T5pad", "deliverable T5@1.01 (pad)"), ("deliv_g100_T5pad", "deliverable T5@1.00 (pad)"),
         ("deliv_g100_T5nat", "deliverable T5@1.00 (UNPADDED)")]
REF = {"origin": "origin_ll", "deliv": "mstudent2_step800_deliv_ll", "mamba": "mamba_ll", "s25": "origin_ll"}
def model_ref(lab):
    if lab.startswith("deliv") or lab.startswith("mstudent2"): return "mstudent2_step800_deliv_ll"
    if lab.startswith("origin"): return "origin_ll"
    return None
def lp(c, lab): return me[(c, lab)]["lp"]
def shr(c, lab): return me[(c, lab)]["sh"] / me[(c, lab)]["gs"]
P("\n[3] PER CLIP LPIPS (this lane's numbers)")
P("  " + f"{'row':34s}" + "".join(f"{c:>8s}" for c in CLIPS) + f"{'MEAN12':>9s}")
for lab, nm in NAMES:
    P("  " + f"{nm:34s}" + "".join(f"{lp(c, lab):8.4f}" for c in CLIPS) + f"{st.mean(lp(c, lab) for c in CLIPS):9.4f}")
P("\n[4] MEANS (12 clips): LPIPS | delta vs deployed origin | clips better than deployed origin | worst clip vs origin |"
  " delta vs same model's deployed 8x2@1.01 | worst clip vs that | mean sharp/GT | mean per-clip sharp ratio vs same model's 8x2")
summary = {}
for lab, nm in NAMES:
    m = st.mean(lp(c, lab) for c in CLIPS)
    d = [lp(c, lab) - lp(c, "origin_ll") for c in CLIPS]
    better = sum(x < 0 for x in d)
    wc = max(range(12), key=lambda i: d[i])
    ref = model_ref(lab)
    if ref and ref != lab:
        dr = [lp(c, lab) - lp(c, ref) for c in CLIPS]; wr = max(range(12), key=lambda i: dr[i])
        srat = st.mean(me[(c, lab)]["sh"] / me[(c, ref)]["sh"] for c in CLIPS)
        drs = f"{st.mean(dr):+.4f} | {CLIPS[wr]} {dr[wr]:+.4f} | "
        srs = f"{srat:.4f}"
    else:
        dr = None; drs = "    -   |      -       | "; srs = "  -   "; srat = None
    shg = st.mean(shr(c, lab) for c in CLIPS)
    summary[lab] = dict(mean=m, d_origin=st.mean(d), better=better, worst_origin=(CLIPS[wc], d[wc]),
                        d_ref=(st.mean(dr) if dr else None), worst_ref=((CLIPS[wr], dr[wr]) if dr else None),
                        sh_gt=shg, sh_ref=srat)
    P(f"  {nm:34s} {m:.4f} | {st.mean(d):+.4f} | {better:2d}/12 | {CLIPS[wc]} {d[wc]:+.4f} | {drs}{shg:.3f} | {srs}")

# ------------------------------------------------------------------ 5. verdicts recomputed with the lanes' rules
P("\n[5] VERDICTS RECOMPUTED FROM THIS LANE'S NUMBERS (rules of speed/PREREG.txt and PREREG_ADDENDUM_posthoc.txt)")
def rule(lab, ref, sharp=False):
    dr = [lp(c, lab) - lp(c, ref) for c in CLIPS]
    a = st.mean(dr) <= 0.002; wi = max(range(12), key=lambda i: dr[i]); b = dr[wi] <= 0.005
    s = f"mean {st.mean(dr):+.4f} ({'ok' if a else 'FAIL'}), worst {CLIPS[wi]} {dr[wi]:+.4f} ({'ok' if b else 'FAIL'})"
    ok = a and b
    if sharp:
        r = st.mean(me[(c, lab)]["sh"] / me[(c, ref)]["sh"] for c in CLIPS); cc = 0.97 <= r <= 1.03
        s += f", sharp ratio {r:.4f} ({'ok' if cc else 'FAIL'})"; ok &= cc
    return ("PASS" if ok else "FAIL"), s
for lab, ref, sharp, tag in [("deliv_g100_s8", "mstudent2_step800_deliv_ll", True, "S1 deliverable guidance 1.00 [GATING]"),
                             ("origin_g100_s8", "origin_ll", True, "S1 origin guidance 1.00 [reported]"),
                             ("deliv_g101_T6pad", "mstudent2_step800_deliv_ll", False, "S2' deliverable T6@1.01 [GATING]"),
                             ("deliv_g101_T5pad", "mstudent2_step800_deliv_ll", False, "S2' deliverable T5@1.01 [GATING]"),
                             ("deliv_g100_T5pad", "mstudent2_step800_deliv_ll", False, "S2' deliverable T5@1.00 vs deployed [GATING]"),
                             ("deliv_g100_T5nat", "mstudent2_step800_deliv_ll", False, "post-hoc deliverable T5@1.00 UNPADDED vs deployed"),
                             ("origin_g101_T6pad", "origin_ll", False, "origin T6@1.01 [reported]"),
                             ("origin_g101_T5pad", "origin_ll", False, "origin T5@1.01 [reported]"),
                             ("origin_g100_T5pad", "origin_ll", False, "origin T5@1.00 [reported]")]:
    v, s = rule(lab, ref, sharp)
    P(f"  {tag:52s} {v}: {s}")
P("\n[6] SEPARABILITY (deliverable - origin at the SAME sampler setting; mean, clips better)")
for dl, ol, nm in [("mstudent2_step800_deliv_ll", "origin_ll", "8x2@1.01"), ("deliv_g100_s8", "origin_g100_s8", "8x1@1.00"),
                   ("deliv_g101_T6pad", "origin_g101_T6pad", "T6@1.01 pad"), ("deliv_g101_T5pad", "origin_g101_T5pad", "T5@1.01 pad"),
                   ("deliv_g100_T5pad", "origin_g100_T5pad", "T5@1.00 pad")]:
    d = [lp(c, dl) - lp(c, ol) for c in CLIPS]
    P(f"  {nm:14s} {st.mean(d):+.4f}  better on {sum(x < 0 for x in d)}/12  (per clip min {min(d):+.4f} max {max(d):+.4f})")
d = [lp(c, "deliv_g100_T5nat") - lp(c, "deliv_g100_T5pad") for c in CLIPS]
P(f"\n[7] seed-realisation effect, deliverable T5@1.00 unpadded - padded: mean {st.mean(d):+.4f}, per clip "
  + " ".join(f"{c}:{x:+.4f}" for c, x in zip(CLIPS, d)) + f"  (sd {st.pstdev(d):.4f})")
for lab in ("mstudent2_step800_deliv_ll",):
    dd = sorted((lp(c, lab) - lp(c, "origin_ll"), c) for c in CLIPS)
    P(f"    deliverable 8x2 gains vs origin, smallest three: " + ", ".join(f"{c} {x:+.4f}" for x, c in dd[-3:]))
json.dump(dict(summary=summary, rows={f"{k[0]}|{k[1]}": v for k, v in me.items()}), open(outp.replace(".txt", ".json"), "w"), indent=1)
open(outp, "w").write("\n".join(L) + "\n")
print("\n".join(L))
