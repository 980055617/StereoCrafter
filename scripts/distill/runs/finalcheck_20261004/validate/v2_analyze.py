"""V2 analysis: S1/S2/S3 scorer checks + the PRE-REGISTERED transfer rule (PREREG.txt), from the scorer outputs only.
usage: python v2_analyze.py OUT.txt     (reads outputs/finalcheck_20261004/validate/v2_scores/*)"""
import re, sys, glob, os
S = os.environ.get("V2S_DIR", "outputs/finalcheck_20261004/validate/v2_scores")      # env override only for dry-runs
OUT = sys.argv[1]
L = []; p = L.append
CLIPS = ["0170", "0204", "0042", "0052"]
RES = ["1024x1792", "1024x1920"]
EXP_OFF = {"0042": (-12, -12), "0052": (-12, -12), "0170": (-28, 0), "0204": (-28, 0)}
D576_TABLE = {"0042": -0.0061, "0052": -0.0072, "0170": -0.0071, "0204": -0.0186}   # TABLE_HEADLINE_12CLIP.txt
def rows(path):
    out = []
    for ln in open(path):
        if ln.startswith("ROW "):
            d = dict(kv.split("=", 1) for kv in ln.split()[1:])
            out.append(d)
    return out
def printed(path):   # the human-readable rows: tag offset leftPSNR LPIPS sharp [rPSNR n]
    out = {}
    for ln in open(path):
        m = re.match(r"^(\S+)\s+(\(-?\d+,-?\d+\))\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)", ln)
        if m: out.setdefault(m.group(1), []).append(m.groups()[1:])
        g = re.match(r"^--- (\d{4}) GT ---\s+(\d+\.\d+)\s+(\d+\.\d+)", ln)
        if g: out.setdefault(f"GT_{g.group(1)}", []).append(g.groups()[1:])
    return out
p("V2 HI-RES QUALITY -- deliverable vs origin, lossless FFV1, score_clip_ll.py SCORE_STEP=4")
p(f"sources: {S}/V2_hires_scores.txt, {S}/V2_576_rescore.txt, {S}/S1_*.txt, {S}/S2_*.txt")
p("")
# ---------------- S1 ----------------
p("=== S1 reproduction of fulldata/lpips/fullhd.txt '0204_origin (-28,0) 48.63 0.1882 0.0057' (old mp4v render) ===")
want = ("(-28,0)", "48.63", "0.1882", "0.0057")
s1_any = False
for f in sorted(glob.glob(f"{S}/S1_old0204_gt_*.txt")):
    pr = printed(f); got = pr.get("0204_origin", [None])[0]
    ok = got is not None and tuple(got) == want
    s1_any |= ok
    p(f"  GT dir {os.path.basename(f)[len('S1_old0204_gt_'):-4]:22s} -> {' '.join(got) if got else 'NO ROW'}   {'REPRODUCES' if ok else 'differs'}")
p(f"  S1: {'PASS (reproduced with the GT dir marked REPRODUCES)' if s1_any else 'FAIL -- no candidate GT dir reproduces the old row'}")
p("")
# ---------------- S2 ----------------
p("=== S2 equivalence: tracked scripts/distill/score_clip.py vs score_clip_ll.py on identical inputs ===")
a = printed(f"{S}/S2_tracked_score_clip.txt"); b = printed(f"{S}/S2_score_clip_ll.txt")
s2 = True
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k), b.get(k)
    same = va == vb
    s2 &= same
    p(f"  {k:34s} tracked={va} ll={vb} {'IDENTICAL' if same else 'DIFFERENT'}")
p(f"  S2: {'PASS' if s2 and a else 'FAIL'}")
p("")
# ---------------- main ----------------
# ---------------- H2 render gates (timing_gpu1.txt): only rc=0 gate=PASS renders may enter the table ----------------
GATES = {}
TL = os.environ.get("V2_TIMING", "outputs/finalcheck_20261004/validate/timing_gpu1.txt")
for ln in open(TL):
    if ln.startswith("RUN "):
        f = ln.split()
        kv = dict(x.split("=", 1) for x in f[3:] if "=" in x)
        GATES[(f[1], f[2])] = (kv.get("rc"), kv.get("gate"), kv.get("secs"))
p(f"=== H2 render gates ({TL}) ===")
for (c, lab), (rc, g, secs) in sorted(GATES.items()):
    p(f"  {c} {lab:24s} rc={rc} gate={g} secs={secs}")
p("")
R = rows(f"{S}/V2_hires_scores.txt")
for extra in sorted(glob.glob(f"{S}/V2_hires_scores_r*.txt")):     # re-scored retries, if any
    R += rows(extra)
tab = {}; excluded = []
for d in R:
    c = d["clip"]; t = d["tag"]
    for res in RES:
        for m in ("origin", "deliv"):
            mm = re.fullmatch(rf"{c}_{m}_ll_{res}(_r(\d+))?", t)
            if not mm: continue
            lab = t[len(c) + 1:]
            rc, g, _ = GATES.get((c, lab), (None, None, None))
            if rc != "0" or g != "PASS":
                excluded.append(f"{c} {lab} (rc={rc} gate={g})"); continue
            rank = int(mm.group(2) or 1)
            if (c, res, m) not in tab or rank > tab[(c, res, m)][0]:
                tab[(c, res, m)] = (rank, d)
tab = {k: v[1] for k, v in tab.items()}
if excluded:
    p("  EXCLUDED by H2 gate: " + "; ".join(excluded)); p("")
s3 = True
p("=== S3 geometry: recovered offset vs crop math, and origin/deliv registration identical ===")
for c in CLIPS:
    for res in RES:
        o, dl = tab.get((c, res, "origin")), tab.get((c, res, "deliv"))
        if not o or not dl:
            p(f"  {c} {res}: MISSING row(s)"); s3 = False; continue
        off_o = (int(o["dy"]), int(o["dx"])); off_d = (int(dl["dy"]), int(dl["dx"]))
        ok = off_o == EXP_OFF[c] and off_d == EXP_OFF[c] and o["leftPSNR"] == dl["leftPSNR"]
        s3 &= ok
        p(f"  {c} {res}: origin {off_o} leftPSNR {o['leftPSNR']} | deliv {off_d} leftPSNR {dl['leftPSNR']} | expected {EXP_OFF[c]} -> {'ok' if ok else 'MISMATCH'}")
p(f"  S3: {'PASS' if s3 else 'FAIL'}")
p("")
r576 = {}
if os.path.exists(f"{S}/V2_576_rescore.txt"):
    for d in rows(f"{S}/V2_576_rescore.txt"):
        c = d["clip"]
        if d["tag"] == f"{c}_origin_ll": r576[(c, "origin")] = d
        if d["tag"] == f"{c}_mstudent2_step800_deliv_ll": r576[(c, "deliv")] = d
verdict = {}
for res in RES:
    p(f"=== {res} (HxW), tile_num=1 -- per clip ===")
    p(f"  {'clip':6s} {'LPIPS origin':>13s} {'LPIPS deliv':>12s} {'delta':>9s} | {'sharp o':>8s} {'sharp d':>8s} {'GTsharp':>8s} {'o/GT':>6s} {'d/GT':>6s} | {'576 delta (table)':>17s} {'576 delta (rescore)':>19s}")
    deltas = []; lo = []; ld = []
    for c in CLIPS:
        o, dl = tab.get((c, res, "origin")), tab.get((c, res, "deliv"))
        if not o or not dl: p(f"  {c}: MISSING"); continue
        lpo, lpd = float(o["lpips"]), float(dl["lpips"]); dd = lpd - lpo
        so, sd, gs = float(o["sharp"]), float(dl["sharp"]), float(o["gtSharp"])
        re576 = ""
        if (c, "origin") in r576 and (c, "deliv") in r576:
            re576 = f"{float(r576[(c,'deliv')]['lpips']) - float(r576[(c,'origin')]['lpips']):+.4f}"
        deltas.append(dd); lo.append(lpo); ld.append(lpd)
        p(f"  {c:6s} {lpo:13.4f} {lpd:12.4f} {dd:+9.4f} | {so:8.4f} {sd:8.4f} {gs:8.4f} {so/gs:6.3f} {sd/gs:6.3f} | {D576_TABLE[c]:+17.4f} {re576:>19s}")
    if len(deltas) != 4:
        verdict[res] = None
        p(f"  only {len(deltas)}/4 clips passed the gates -> PRE-REGISTERED RULE NOT EVALUABLE at {res}")
    if len(deltas) == 4:
        mo, md = sum(lo) / 4, sum(ld) / 4; worst = max(deltas)
        tr = (md < mo) and (worst <= 0.005)
        verdict[res] = tr
        p(f"  4-clip mean: origin {mo:.4f}  deliv {md:.4f}  delta {md - mo:+.4f}  improved {sum(x < 0 for x in deltas)}/4  worst clip {worst:+.4f}"
          f"   (576x1024 same 4 clips, table: {sum(D576_TABLE.values())/4:+.4f})")
        p(f"  PRE-REGISTERED RULE (mean improves AND no clip worse than +0.005): {'TRANSFERS' if tr else 'DOES NOT TRANSFER'}")
    p("")
if r576:
    p("=== 576x1024 re-score of the existing renders (consistency with TABLE_HEADLINE_12CLIP.txt) ===")
    for c in CLIPS:
        if (c, "origin") in r576:
            p(f"  {c}: origin {float(r576[(c,'origin')]['lpips']):.4f}  deliv {float(r576[(c,'deliv')]['lpips']):.4f}")
    p("")
p("=== FINAL STATEMENT (per resolution) ===")
for res in RES:
    if verdict.get(res) is None:
        p(f"  {res}: NOT EVALUABLE (missing or gate-failed renders)")
    elif res in verdict:
        p(f"  {res}: the 576x1024 quality gain {'TRANSFERS' if verdict[res] else 'does NOT transfer'} -> the quality claim "
          f"{'COVERS' if verdict[res] else 'does NOT cover'} this speed-claim resolution"
          f"{'' if (s2 and s3) else '  [scorer checks S2/S3 not all passed -- read with care]'}")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
