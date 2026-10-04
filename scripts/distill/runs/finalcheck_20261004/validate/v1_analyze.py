"""V1 analysis: applies the PRE-REGISTERED rules in PREREG.txt (F1/F2/F3 + tLP reading) to a score_temporal JSON.
usage: python v1_analyze.py IN.json OUT.txt [label]
Only reads the JSON; every printed number is computed from it (source path is printed in the header)."""
import json, sys
IN, OUT = sys.argv[1], sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else "576x1024"
d = json.load(open(IN))
CLIPS = sorted(d.keys())
M = [("origin", "_origin_ll"), ("shipped", "_mamba_ll"), ("deliv", "_mstudent2_step800_deliv_ll"), ("s25", "_s25_ll")]
if LABEL not in ("576x1024", "partial-test"):        # hi-res renders of this lane: <clip>_<model>_ll_<HxW>
    M = [("origin", f"_origin_ll_{LABEL}"), ("deliv", f"_deliv_ll_{LABEL}")]
def tagfor(c, suf):
    for t in d[c]:
        if t.startswith(c) and t.endswith(suf):
            return t
    return None
rows = {}
for c in CLIPS:
    rows[c] = {"GT": d[c]["GT"]}
    for name, suf in M:
        t = tagfor(c, suf)
        if t is not None:
            rows[c][name] = d[c][t]
methods = [m for m, _ in M if all(m in rows[c] for c in CLIPS)]
L = []
p = L.append
p(f"V1 TEMPORAL CONSISTENCY -- {LABEL} -- source {IN}")
p(f"clips (n={len(CLIPS)}): {' '.join(CLIPS)}   methods: {', '.join(methods)}")
p("tLP = mean LPIPS(frame t, t+1); warp = flow-warped |I_t+1(x+F) - I_t(x)| with RAFT flow on the GT window;")
p("ratio = seam/nonseam mean |R_t+1 - R_t| (seams t=11k+2, k>=1).  Lower is better for warp and ratio.")
p("")
def tab(title, key, fmt="{:8.4f}", with_gt=True):
    p(f"--- {title} ---")
    hdr = f"  {'row':10s}" + "".join(f"{c:>9s}" for c in CLIPS) + f"{'MEAN':>10s}"
    p(hdr)
    names = (["GT"] if with_gt else []) + methods
    for m in names:
        vals = [rows[c][m][key] if key != "ratio" else rows[c][m]["seam"] / rows[c][m]["nonseam"] for c in CLIPS]
        p(f"  {m:10s}" + "".join(" " + fmt.format(v) for v in vals) + f"  {sum(vals)/len(vals):8.4f}")
    p("")
tab("tLP", "tLP")
tab("warp error", "warp")
tab("seam ratio (seam/nonseam)", "ratio")
p("--- tLP / tLP_GT (1.0 = flickers as much as the real right eye) ---")
p(f"  {'row':10s}" + "".join(f"{c:>9s}" for c in CLIPS) + f"{'MEAN':>10s}{'mean|r-1|':>11s}")
clo = {}
for m in methods:
    r = [rows[c][m]["tLP"] / rows[c]["GT"]["tLP"] for c in CLIPS]
    clo[m] = sum(abs(x - 1) for x in r) / len(r)
    p(f"  {m:10s}" + "".join(f" {x:8.3f}" for x in r) + f"  {sum(r)/len(r):8.3f}  {clo[m]:9.4f}")
p("")
p("--- warp / warp(origin) and ratio / ratio(origin), per clip ---")
for key in ("warp", "ratio"):
    for m in methods:
        if m == "origin": continue
        def g(c, mm):
            x = rows[c][mm]
            return x["warp"] if key == "warp" else x["seam"] / x["nonseam"]
        r = [g(c, m) / g(c, "origin") for c in CLIPS]
        p(f"  {key:5s} {m:8s}" + "".join(f" {x:8.3f}" for x in r))
p("")
def mean(m, key):
    v = [rows[c][m]["warp"] if key == "warp" else rows[c][m]["seam"] / rows[c][m]["nonseam"] for c in CLIPS]
    return sum(v) / len(v)
n = len(CLIPS)
p("=== PRE-REGISTERED FLAGS (PREREG.txt), deliverable vs origin ===")
nan = any(rows[c][m]["warp"] != rows[c][m]["warp"] for c in CLIPS for m in methods)
f2_need = 9 if n == 12 else (3 if n == 4 else None)
res = {}
for m in [x for x in methods if x != "origin"]:
    mw, ow = mean(m, "warp"), mean("origin", "warp")
    mr, orr = mean(m, "ratio"), mean("origin", "ratio")
    worse = sum(rows[c][m]["warp"] > rows[c]["origin"]["warp"] for c in CLIPS)
    F1 = mw > 1.05 * ow; F2 = (worse >= f2_need) if f2_need else None; F3 = mr > 1.10 * orr
    res[m] = (F1, F2, F3)
    tag = "PRE-REGISTERED" if m == "deliv" else "context only"
    p(f"  [{tag}] {m}: mean warp {mw:.5f} vs origin {ow:.5f} ({(mw/ow-1)*100:+.1f}%) -> F1 {'FLAG' if F1 else 'ok'} (>+5%)")
    p(f"  [{tag}] {m}: warp worse than origin on {worse}/{n} clips -> F2 {'FLAG' if F2 else 'ok'} (>= {f2_need}/{n})")
    p(f"  [{tag}] {m}: mean seam ratio {mr:.4f} vs origin {orr:.4f} ({(mr/orr-1)*100:+.1f}%) -> F3 {'FLAG' if F3 else 'ok'} (>+10%)")
if nan:
    p("  !! NaN warp present -> F1/F2 NOT EVALUATED")
F1, F2, F3 = res["deliv"]
flags = [k for k, v in (("F1", F1), ("F2", F2), ("F3", F3)) if v]
p(f"  OUTCOME (deliverable): {'TEMPORAL REGRESSION FLAGGED: ' + ', '.join(flags) if flags else 'PASS (no flag)'}")
p("")
p("=== PRE-REGISTERED tLP READING ===")
p(f"  closeness c_m = mean |tLP_m/tLP_GT - 1|:  " + "  ".join(f"{m} {clo[m]:.4f}" for m in methods))
if "s25" in clo:
    further = clo["deliv"] > clo["s25"]
    nfar = sum(abs(rows[c]["deliv"]["tLP"] / rows[c]["GT"]["tLP"] - 1) > abs(rows[c]["s25"]["tLP"] / rows[c]["GT"]["tLP"] - 1) for c in CLIPS)
    p(f"  deliverable further from GT's tLP than origin+25 steps: {'YES' if further else 'NO'} "
      f"(c_deliv {clo['deliv']:.4f} vs c_s25 {clo['s25']:.4f}); deliverable further on {nfar}/{n} clips")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
