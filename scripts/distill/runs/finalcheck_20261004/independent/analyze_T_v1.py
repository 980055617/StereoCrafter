#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- temporal analysis (CPU only).
T1-repro: tracked score_temporal.py JSONs (0259, 0225) vs validate's V1 JSON at 4 decimals.
T2-a    : T2 (validate's copy, re-run here) origin/deliv vs validate's V1 JSON at 4 decimals.
T2-b    : the recommended setting keeps the deliverable's temporal profile (PREREG.txt section T).
usage: analyze_T_v1.py OUT.txt T1_0259.json T1_0225.json T2.json
"""
import json, os, sys, statistics as st

os.chdir("/home/kawa/master_project/StereoCrafter")
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
VAL = json.load(open("outputs/finalcheck_20261004/validate/v1_temporal/temporal_576_12clip.json"))
outp = sys.argv[1]
assert not os.path.exists(outp), "refusing to overwrite"
t1 = {}
for p in sys.argv[2:4]:
    t1.update(json.load(open(p)))
t2 = json.load(open(sys.argv[4]))
L = []; P = L.append
KEYS = ("tLP", "warp", "seam", "nonseam")
SUF = {"origin": "origin_ll", "shipped": "mamba_ll", "deliv": "mstudent2_step800_deliv_ll", "s25": "s25_ll",
       "g100_s8": "deliv_g100_s8", "T5pad": "deliv_g100_T5pad", "T5nat": "deliv_g100_T5nat"}


def get(res, c, m):
    return res[c]["GT"] if m == "GT" else res[c][f"{c}_{SUF[m]}"]


def cmp4(a, b):
    return all(round(a[k], 4) == round(b[k], 4) for k in KEYS), max(abs(a[k] - b[k]) for k in KEYS)


P("=" * 118)
P("TEMPORAL -- independent lane.  T1 = TRACKED scripts/distill/score_temporal.py; T2 = validate's score_temporal_ll.py re-run")
P("=" * 118)
P("\n[T1-repro] tracked script vs validate's V1 JSON (outputs/finalcheck_20261004/validate/v1_temporal/temporal_576_12clip.json)")
allok = True
for c in ("0259", "0225"):
    for m in ("GT", "origin", "shipped", "deliv", "s25"):
        ok, mx = cmp4(get(t1, c, m), get(VAL, c, m)); allok &= ok
        P(f"  {c} {m:8s} equal at 4 dp: {ok}   max |diff| {mx:.2e}")
P(f"  T1-repro: {'PASS' if allok else 'FAIL'}")
P("\n[T1 extra rows, tracked script] per clip: tLP  tLP/GT  warp  warp/warp(deliv)  seam ratio")
for c in ("0259", "0225"):
    g = get(t1, c, "GT"); d = get(t1, c, "deliv")
    for m in ("origin", "deliv", "s25", "g100_s8", "T5pad", "T5nat"):
        r = get(t1, c, m)
        P(f"  {c} {m:8s} tLP {r['tLP']:.4f} ({r['tLP'] / g['tLP']:.3f}x GT)  warp {r['warp']:.4f} ({r['warp'] / d['warp']:.3f}x deliv)  "
          f"ratio {r['seam'] / r['nonseam']:.3f}")
P("\n[T2-a] re-run of validate's copy: origin and deliv vs validate's V1 JSON")
ok2 = True
for c in CLIPS:
    for m in ("GT", "origin", "deliv"):
        ok, mx = cmp4(get(t2, c, m), get(VAL, c, m)); ok2 &= ok
        if not ok:
            P(f"  {c} {m} NOT equal at 4 dp (max |diff| {mx:.2e})")
P(f"  T2-a: {'PASS (all 36 GT/origin/deliv entries equal at 4 dp)' if ok2 else 'FAIL'}")
# cross-check T1 extra rows vs T2 rows (tracked vs copy on the same renders)
okx = True
for c in ("0259", "0225"):
    for m in ("g100_s8", "T5pad", "T5nat"):
        ok, mx = cmp4(get(t1, c, m), get(t2, c, m)); okx &= ok
        P(f"  tracked vs copy on {c} {m}: equal at 4 dp {ok} (max |diff| {mx:.2e})")


def agg(res, m, src=None):
    w = [get(res, c, m)["warp"] for c in CLIPS]
    r = [get(res, c, m)["seam"] / get(res, c, m)["nonseam"] for c in CLIPS]
    t = [get(res, c, m)["tLP"] for c in CLIPS]
    tg = [get(res, c, "GT")["tLP"] for c in CLIPS]
    clo = st.mean(abs(a / b - 1) for a, b in zip(t, tg))
    return dict(warp=st.mean(w), ratio=st.mean(r), tlp=st.mean(t), clo=clo, w=w, r=r, t=t, tg=tg)


A = {m: agg(t2, m) for m in ("origin", "deliv", "g100_s8", "T5pad", "T5nat")}
A["s25"] = agg(VAL, "s25")
P("\n[T2] 12-clip means (warp, seam ratio = mean of per-clip seam/nonseam, tLP, closeness c = mean|tLP/tLP_GT - 1|)")
for m in ("origin", "deliv", "s25", "g100_s8", "T5pad", "T5nat"):
    a = A[m]
    P(f"  {m:8s} warp {a['warp']:.5f}  ratio {a['ratio']:.4f}  tLP {a['tlp']:.4f}  c {a['clo']:.4f}" + ("   [s25 from validate's JSON]" if m == "s25" else ""))
P("\n[T2-b] PRE-REGISTERED: recommended setting keeps the deliverable's temporal profile (vs deliverable 8x2@1.01)")
for m in ("T5pad", "T5nat", "g100_s8"):
    a, d, o = A[m], A["deliv"], A["origin"]
    i = a["warp"] <= 1.05 * d["warp"]; ii = a["ratio"] <= 1.10 * d["ratio"]
    f2 = sum(x > y for x, y in zip(a["w"], d["w"]))
    tag = "[GATING]" if m != "g100_s8" else "[fallback, reported]"
    P(f"  {m:8s} {tag:20s} (i) warp {a['warp']:.5f} vs 1.05 x {d['warp']:.5f} = {1.05 * d['warp']:.5f} -> {'ok' if i else 'FAIL'} "
      f"({a['warp'] / d['warp'] - 1:+.1%});  (ii) ratio {a['ratio']:.4f} vs 1.10 x {d['ratio']:.4f} -> {'ok' if ii else 'FAIL'} "
      f"({a['ratio'] / d['ratio'] - 1:+.1%})  => {'KEEPS' if (i and ii) else 'DOES NOT KEEP'}")
    P(f"           reported: warp > deliv on {f2}/12 clips; c {a['clo']:.4f} vs deliv {d['clo']:.4f} vs s25 {A['s25']['clo']:.4f}; "
      f"validate's flags vs origin: F1 {a['warp'] / o['warp'] - 1:+.1%} ({'FLAG' if a['warp'] > 1.05 * o['warp'] else 'ok'}), "
      f"F2 {sum(x > y for x, y in zip(a['w'], o['w']))}/12 ({'FLAG' if sum(x > y for x, y in zip(a['w'], o['w'])) >= 9 else 'ok'}), "
      f"F3 {a['ratio'] / o['ratio'] - 1:+.1%} ({'FLAG' if a['ratio'] > 1.10 * o['ratio'] else 'ok'})")
P("\n[T2] per clip warp ratio vs deliverable 8x2@1.01 / tLP ratio vs deliverable")
P("  " + f"{'':8s}" + "".join(f"{c:>8s}" for c in CLIPS))
for m in ("g100_s8", "T5pad", "T5nat"):
    P("  " + f"{m:8s}" + "".join(f"{x / y:8.3f}" for x, y in zip(A[m]["w"], A["deliv"]["w"])) + "   (warp)")
    P("  " + f"{m:8s}" + "".join(f"{x / y:8.3f}" for x, y in zip(A[m]["t"], A["deliv"]["t"])) + "   (tLP)")
open(outp, "w").write("\n".join(L) + "\n")
print("\n".join(L))
