#!/usr/bin/env python
"""finalcheck_20261004 / independent lane -- H3 timing (CPU only).
(a) 576x1024: recompute the speed lane's deployed-path per-window UNet time from its speed_log.json files
    (definition of bench TABLE_COMBINED_v1.txt X1 "meas": median over renders of [median over windows 1..n-2 of the
    CUDA-event UNet ms per window]) and compare with that column (tolerance 1 %).
(b) hi-res: the same statistic from this lane's renders (outputs/finalcheck_20261004/independent/hres/clips/*), plus
    whole-clip UNet seconds and window-0 excess; ratios vs the origin 8x2@1.01 render of the same resolution.
usage: analyze_timing_v2.py OUT.txt  (v2: GPU-0-only timing sets, ADDENDUM 4)
"""
import glob, json, os, statistics as st, sys

os.chdir("/home/kawa/master_project/StereoCrafter")
outp = sys.argv[1]
assert not os.path.exists(outp), "refusing to overwrite"
L = []; P = L.append


def stats(p):
    j = json.load(open(p))
    w = j["windows"]
    ms = [x["unet_ms"] for x in w]
    steady = st.median(ms[1:-1]) if len(ms) > 2 else st.median(ms)
    calls = sorted(set(x["unet_calls"] for x in w)); bs = sorted(set(b for x in w for b in x["batches"]))
    return dict(n=len(w), steady_ms=steady, total_s=sum(ms) / 1000, w0_ms=ms[0], calls=calls, bs=bs,
                per_fwd_ms=steady / calls[0] if len(calls) == 1 else None, res=j.get("res", "config"),
                inloop_s=j.get("call_s_sum"), hook_s=j.get("hook_total_s"))


BENCH_MEAS = {"origin_g101_s8": 7.627, "deliv_g101_s8": 7.246, "deliv_g100_s8": 4.012, "deliv_g101_T6pad": 5.432,
              "deliv_g101_T5pad": 4.526, "deliv_g100_T5pad": 2.507, "origin_g100_s8": 4.211, "origin_g101_T6pad": 5.721,
              "origin_g101_T5pad": 4.765, "origin_g100_T5pad": 2.631}
P("=" * 118)
P("H3 TIMING -- CUDA-event UNet time per 14-frame window on the DEPLOYED path (speed hook instrumentation)")
P("=" * 118)
P("\n(a) 576x1024, recomputed from outputs/finalcheck_20261004/speed/clips/<clip>_<label>/speed_log.json")
P(f"  {'label':20s} {'n':>3s} {'steady s/win':>13s} {'bench meas':>11s} {'diff':>7s} {'rel vs origin 8x2':>18s} {'s/forward':>10s} calls bs")
ref576 = None
res576 = {}
for lab in BENCH_MEAS:
    ps = sorted(glob.glob(f"outputs/finalcheck_20261004/speed/clips/*_{lab}/speed_log.json"))
    ps = [p for p in ps if os.path.basename(os.path.dirname(p)).split("_", 1)[1] == lab]
    S = [dict(stats(p), dir=os.path.basename(os.path.dirname(p))) for p in ps]
    med = st.median(s["steady_ms"] for s in S) / 1000
    res576[lab] = (med, S)
    if lab == "origin_g101_s8":
        ref576 = med
    d = med / BENCH_MEAS[lab] - 1
    P(f"  {lab:20s} {len(S):3d} {med:13.3f} {BENCH_MEAS[lab]:11.3f} {d:+7.2%} {med / ref576:18.4f} "
      f"{st.median(s['per_fwd_ms'] for s in S) / 1000:10.4f} {S[0]['calls']} {S[0]['bs']}")
P("  window-0 excess (w0 - steady) median over renders: " + ", ".join(
    f"{lab} {st.median((s['w0_ms'] - s['steady_ms']) / 1000 for s in res576[lab][1]):+.2f} s" for lab in ("origin_g101_s8", "deliv_g101_s8", "deliv_g100_T5pad", "origin_g100_T5pad")))

P("\n(b) hi-res, this lane's renders (outputs/finalcheck_20261004/independent/hres/clips)")
rows = []
for p in sorted(glob.glob("outputs/finalcheck_20261004/independent/hres/clips/*/speed_log.json")):
    d = os.path.basename(os.path.dirname(p)); s = stats(p); s["dir"] = d; rows.append(s)
# v2 (ADDENDUM 4): every timing number comes from GPU 0.  1024x1792: the GPU-0 trio + H4 (no suffix) and the three
# quality-only renders (shared GPU 0 with the R1 scorer, shown for information); 1024x1920: the GPU-0 set (suffix _g0).
# GPU-1 1024x1920 renders are listed separately as a thermal-drift note.
SETS = [("1024x1792", "", lambda d: not d.endswith("_g0")),
        ("1024x1920", "_g0", lambda d: d.endswith("_g0")),
        ("1024x1920 (GPU 1, drift note only)", "", lambda d: (not d.endswith("_g0")))]
for label, suf, sel in SETS:
    res = label.split()[0]
    R = [r for r in rows if r["res"] == res and sel(r["dir"])]
    if not R:
        P(f"  {label}: no renders"); continue
    ref = [r for r in R if r["dir"] == f"0204_origin_g101_s8_{res}{suf}"]
    refms = ref[0]["steady_ms"] if ref else None
    reftot = ref[0]["total_s"] if ref else None
    P(f"  --- {label} --- (reference: 0204 origin 8x2@1.01{suf}, steady {refms / 1000 if refms else float('nan'):.3f} s/window)")
    P(f"  {'render':34s} {'win':>4s} {'steady s/win':>13s} {'rel steady':>11s} {'whole-clip UNet s':>18s} {'rel whole':>10s} {'w0 excess s':>12s} {'s/forward':>10s} calls bs")
    for r in R:
        P(f"  {r['dir']:34s} {r['n']:4d} {r['steady_ms'] / 1000:13.3f} {(r['steady_ms'] / refms) if refms else float('nan'):11.4f} "
          f"{r['total_s']:18.2f} {(r['total_s'] / reftot) if reftot else float('nan'):10.4f} {(r['w0_ms'] - r['steady_ms']) / 1000:+12.2f} "
          f"{(r['per_fwd_ms'] or float('nan')) / 1000:10.4f} {r['calls']} {r['bs']}")
    t5 = [r for r in R if "_deliv_g100_T5pad_" in r["dir"]]
    for r in R:
        ms = [w for w in json.load(open(f"outputs/finalcheck_20261004/independent/hres/clips/{r['dir']}/speed_log.json"))["windows"]]
        u = [w["unet_ms"] / 1000 for w in ms[1:-1]]
        P(f"    drift {r['dir']}: windows 1..n-2 first {u[0]:.2f} s, last {u[-1]:.2f} s ({u[-1] / u[0] - 1:+.1%})")
    if t5 and refms:
        P(f"  deliverable T5@1.00 steady s/window over {len(t5)} clips: median {st.median(r['steady_ms'] for r in t5) / 1000:.3f} "
          f"(min {min(r['steady_ms'] for r in t5) / 1000:.3f}, max {max(r['steady_ms'] for r in t5) / 1000:.3f}) -> rel "
          f"{st.median(r['steady_ms'] for r in t5) / refms:.4f}")
open(outp, "w").write("\n".join(L) + "\n")
json.dump(dict(rows=rows, res576={k: [v[0], [s for s in v[1]]] for k, v in res576.items()}), open(outp.replace(".txt", ".json"), "w"), indent=1)
print("\n".join(L))
