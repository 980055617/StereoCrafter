"""more_20261004 / mamba_scope -- analysis of run_v1 (rules fixed in PREREG.txt, written before any number existed).
usage: analyze_v1.py RUNDIR > TABLE_v1.txt
Every number printed comes from RUNDIR/bench_results.txt (RESULT lines), RUNDIR/profiles/*.json (pass B), or the
finalcheck anchors quoted below (outputs/finalcheck_20261004/bench/run_v1, PREREG G2/G3).
"""
import json, os, re, sys, csv, statistics as st
from collections import defaultdict

RUN = sys.argv[1]
RES_ORDER = [(576, 1024), (1024, 1792), (1024, 1920)]
CFGS_A = ["origin", "mamba5", "mamba10", "all16"]
CFGS_B = ["origin", "mamba10", "all16"]
EXPECT = {  # G1: per_fwd_calls (origin_attn, mamba_core, plain_attn1), gates, total_replaced
    "origin": ((0, 0, 16), [], 0), "mamba5": ((0, 5, 11), [1.0], 5),
    "mamba10": ((0, 10, 6), [1.0], 10), "all16": ((0, 16, 0), [1.0], 16)}
REPLACED = {  # slot-prefix families replaced per config
    "origin": [], "mamba5": ["down_blocks.0.", "up_blocks.3."],
    "mamba10": ["down_blocks.0.", "up_blocks.3.", "down_blocks.1.", "up_blocks.2."], "all16": [""]}
# finalcheck anchors, BS=1 (outputs/finalcheck_20261004/bench/run_v1; TABLE_COMBINED_v1.txt / TABLE_DTYPE_v1.txt)
ANCH_T = {(576, 1024): (0.4857, 0.4587), (1024, 1792): (1.8792, 1.4953), (1024, 1920): (2.0603, 1.6159)}
ANCH_SHARE = {(576, 1024): 0.9445, (1024, 1792): 0.7957, (1024, 1920): 0.7843}
ANCH_MIB = {(576, 1024): (5075, 5081), (1024, 1792): (9570, 9575), (1024, 1920): (10040, 10047)}
ORIGIN_BS2 = {(576, 1024): 0.9612, (1024, 1792): 3.7127, (1024, 1920): 4.0591}  # finalcheck run_v1 BS=2 means (cross-run)
LEVEL = lambda n: ("L0" if n.startswith(("down_blocks.0.", "up_blocks.3.")) else
                   "L1" if n.startswith(("down_blocks.1.", "up_blocks.2.")) else
                   "L2" if n.startswith(("down_blocks.2.", "up_blocks.1.")) else "mid")
SHORT = lambda n: n.replace(".transformer_blocks.0.attn1", "").replace("down_blocks.", "d").replace("up_blocks.", "u") \
                   .replace(".attentions.", ".a").replace("mid_block.a0", "mid")

lines = open(os.path.join(RUN, "bench_results.txt")).read().splitlines()
resA, resB, meta = [], [], {}
for ln in lines:
    if ln.startswith("PASSA RESULT "): resA.append(json.loads(ln[len("PASSA RESULT "):]))
    elif ln.startswith("PASSB RESULT "): resB.append(json.loads(ln[len("PASSB RESULT "):]))
    elif ln.startswith("RUNMETA "):
        kv = dict(x.split("=", 1) for x in ln[len("RUNMETA "):].split() if "=" in x)
        meta[(kv["pass"], kv["tag"])] = kv
fails = [ln for ln in lines if ln.startswith("FAILED")]

def parse_tag(tag):
    m = re.match(r"(\w+?)_bs1_h(\d+)w(\d+)_r(\d+)(_retry)?$", tag)
    return m.group(1), (int(m.group(2)), int(m.group(3))), int(m.group(4))

def hr(v): return (max(v) - min(v)) / 2 if len(v) > 1 else 0.0
out = []
P = out.append
P("=" * 150)
P("more_20261004 / mamba_scope -- does replacing the 5 LEVEL-1 spatial attn1 slots with Mamba buy >= 10 % extra UNet time at 1024x1792?")
P(f"run dir {RUN};  rules: scripts/distill/runs/more_20261004/mamba_scope/PREREG.txt;  BS=1, fp16, 14 frames, synthetic, no weights")
for f in ("versions.txt", "script_md5.txt", "passB_copy_diff.txt"):
    p = os.path.join(RUN, f)
    if os.path.exists(p):
        P(f"  {f}:"); [P("     " + x) for x in open(p).read().splitlines()]
P("=" * 150)
P(f"  pass A processes with RESULT: {len(resA)} (expected 36);  pass B: {len(resB)} (expected 18);  FAILED lines: {len(fails)}")
for f in fails: P("    " + f)

# ---------------- G1 instrumentation ----------------
g1_bad = []
for P_, rows in (("A", resA), ("B", resB)):
    for r in rows:
        cfg, res, rep = parse_tag(r["label"])
        exp_calls, exp_g, exp_rep = EXPECT[cfg]
        calls = r["per_fwd_calls"]; got = (calls["origin_attn"], calls["mamba_core"], calls["plain_attn1"])
        tok = (res[0] // 8) * (res[1] // 8)
        tokens = r.get("tokens")
        rm = meta.get((P_, r["label"]), {})
        ok = got == exp_calls and r["gates"] == exp_g and r["batch"] == 1 and tokens == tok and \
             rm.get("rc") == "0" and rm.get("total_replaced") == str(exp_rep)
        if P_ == "B":
            sl = r["slots"]
            ok = ok and len(sl) == 16 and all(v["calls"] == 10 for v in sl.values())
            for n, v in sl.items():
                rep_ = any(n.startswith(p) for p in REPLACED[cfg])
                want = "GatedResidualMambaSelfAttention" if rep_ else "Attention"
                if v["cls"] != want: ok = False
        if not ok: g1_bad.append((P_, r["label"], got, r["gates"], rm.get("total_replaced"), rm.get("rc")))
P(f"  G1 instrumentation: {'PASS' if not g1_bad else 'FAIL'}" + ("" if not g1_bad else f"  {g1_bad}"))

# ---------------- pass A aggregation ----------------
A = defaultdict(dict)   # A[(cfg,res)][rep] = row
for r in resA:
    cfg, res, rep = parse_tag(r["label"]); A[(cfg, res)][rep] = r
def tvals(cfg, res): return [A[(cfg, res)][k]["sec"] for k in sorted(A[(cfg, res)])]
def mean(cfg, res): v = tvals(cfg, res); return sum(v) / len(v)
def mib(cfg, res): v = [A[(cfg, res)][k]["peak_MiB"] for k in sorted(A[(cfg, res)])]; return sum(v) / len(v)
def paired(num, den, res):  # per-rep paired ratios t(num)/t(den)
    reps = sorted(set(A[(num, res)]) & set(A[(den, res)]))
    return [A[(num, res)][k]["sec"] / A[(den, res)][k]["sec"] for k in reps]

# G2 / G3 / G5
g2, g3 = [], []
for res in RES_ORDER:
    for i, cfg in enumerate(("origin", "mamba5")):
        m = mib(cfg, res); a = ANCH_MIB[res][i]
        g2.append((res, cfg, m, a, abs(m - a) <= 4))
        t = mean(cfg, res); at = ANCH_T[res][i]
        g3.append((res, cfg, t, at, (t / at - 1) * 100, abs(t / at - 1) <= 0.015))
    share = mean("mamba5", res) / mean("origin", res)
    g3.append((res, "share", share, ANCH_SHARE[res], (share - ANCH_SHARE[res]) * 100, abs(share - ANCH_SHARE[res]) <= 0.010))
P(f"  G2 VRAM anchor (+-4 MiB vs finalcheck BS=1): {'PASS' if all(x[-1] for x in g2) else 'FAIL'}")
for res, cfg, m, a, ok in g2: P(f"     {res[0]}x{res[1]:<5d} {cfg:7s} peak {m:8.1f} MiB  anchor {a}  {'ok' if ok else 'OUT'}")
g3_fail_res = sorted({x[0] for x in g3 if not x[-1]})
P(f"  G3 timing anchor (+-1.5 % means, +-1.0 pt share vs finalcheck BS=1): {'PASS' if not g3_fail_res else 'FAIL at ' + str(g3_fail_res)}")
for res, cfg, t, at, d, ok in g3:
    if cfg == "share": P(f"     {res[0]}x{res[1]:<5d} share   {t:.4f} vs {at:.4f}  ({d:+.2f} pt)  {'ok' if ok else 'OUT'}")
    else: P(f"     {res[0]}x{res[1]:<5d} {cfg:7s} {t:.4f} s vs {at:.4f}  ({d:+.2f} %)  {'ok' if ok else 'OUT'}")

# G4 exclusivity
chk = open(os.path.join(RUN, "exclusivity_checks.txt")).read().splitlines() if os.path.exists(os.path.join(RUN, "exclusivity_checks.txt")) else []
uuid0 = None
try:
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    uu = {l.split(",")[1].strip(): int(l.split(",")[0]) for l in q.strip().splitlines()}
except Exception:
    uu = {}
viol = []; cur = None; gpu1_apps = set()
for l in chk:
    if l.startswith(("PRE ", "POST ")): cur = l; continue
    if l.startswith("compute-apps:") or not l.strip() or re.match(r"^\d, ", l): continue
    parts = [x.strip() for x in l.split(",")]
    if len(parts) >= 3 and parts[0] in uu:
        idx = uu[parts[0]]
        if idx == 0 and cur and cur.startswith("PRE") and "snapd-desktop-integration" not in parts[2] and "Xorg" not in parts[2] and "gnome-shell" not in parts[2]:
            viol.append((cur, parts[2], parts[3] if len(parts) > 3 else ""))
        if idx == 1: gpu1_apps.add(parts[2].split("/")[-1])
P(f"  G4 GPU-0 exclusivity before every process: {'PASS' if not viol else 'FAIL'}" + ("" if not viol else f"  {viol[:5]}"))
mon = os.path.join(RUN, "gpu_monitor.csv")
if os.path.exists(mon):
    g1u, g1m, n1 = [], [], 0
    for row in csv.reader(open(mon)):
        if len(row) < 4: continue
        try:
            idx = int(row[1]); u = float(row[2].strip().split()[0]); m = float(row[3].strip().split()[0])
        except Exception: continue
        if idx == 1: g1u.append(u); g1m.append(m)
    if g1u:
        busy = sum(1 for u in g1u if u > 0)
        P(f"  GPU 1 (other lanes) during the run: samples {len(g1u)}, util>0 in {busy} ({100*busy/len(g1u):.0f} %), "
          f"max util {max(g1u):.0f} %, max mem {max(g1m):.0f} MiB; compute apps seen on GPU 1: {sorted(gpu1_apps) or 'none'}")

# ---------------- pass A table ----------------
P("")
P("=" * 150)
P("PASS A -- whole-UNet forward, scripts/distill/bench2.py UNMODIFIED, BS=1 (n=3 processes; +- = half-range; reps-outer)")
P("=" * 150)
P(f"  {'res(HxW)':11s} {'config':8s} {'n':>2s} {'s/fwd':>8s} {'+-':>7s} {'prec%':>6s} {'vs origin':>10s} {'vs mamba5':>10s} {'peakMiB':>8s} {'dMiB vs m5':>10s}")
for res in RES_ORDER:
    to = mean("origin", res); t5 = mean("mamba5", res); m5 = mib("mamba5", res)
    for cfg in CFGS_A:
        v = tvals(cfg, res); t = mean(cfg, res)
        prec = 100 * hr(v) / t
        P(f"  {res[0]}x{res[1]:<6d} {cfg:8s} {len(v):2d} {t:8.4f} {hr(v):7.4f} {prec:6.2f}{'!' if prec > 1.0 else ' '}"
          f"{100*(t/to-1):+9.2f}% {100*(t/t5-1):+9.2f}% {mib(cfg, res):8.0f} {mib(cfg, res)-m5:+10.0f}")
    P("")

P("=" * 150)
P("THE NUMBER -- extra UNet saving of level-0+1 (mamba10) over the shipped level-0 model (mamba5), BS=1")
P("  E = 1 - t(mamba10)/t(mamba5)  (ratio of n=3 means; +- = half-range of the per-rep paired ratios);  E_pts = (t5 - t10)/t(origin)")
P("=" * 150)
summary = {}
for res in RES_ORDER:
    to, t5, t10, t16 = (mean(c, res) for c in CFGS_A)
    pr = paired("mamba10", "mamba5", res); pr16 = paired("all16", "mamba5", res)
    E = 1 - t10 / t5; E16 = 1 - t16 / t5
    summary[res] = dict(to=to, t5=t5, t10=t10, t16=t16, E=E, E16=E16)
    P(f"  {res[0]}x{res[1]:<5d} t(origin) {to:.4f}  t(mamba5) {t5:.4f}  t(mamba10) {t10:.4f}  t(all16) {t16:.4f}")
    P(f"     E(mamba10) = {100*E:+.2f} %  +- {100*hr(pr):.2f}   (per-rep {', '.join(f'{100*(1-x):+.2f}' for x in pr)})   "
      f"E_pts = {100*(t5-t10)/to:+.2f} pt of origin   mamba10 vs origin {100*(t10/to-1):+.2f} %  (mamba5 {100*(t5/to-1):+.2f} %)")
    P(f"     E(all16, informational) = {100*E16:+.2f} %  +- {100*hr(pr16):.2f}   all16 vs origin {100*(t16/to-1):+.2f} %")
    tb = ORIGIN_BS2[res]
    P(f"     per-window UNet, T5 @1.00 (5 x BS1) / deployed origin (8 x BS2 = 8 x {tb} s, finalcheck run_v1, CROSS-RUN): "
      f"mamba5 {5*t5/(8*tb):.4f}  mamba10 {5*t10/(8*tb):.4f}  all16 {5*t16/(8*tb):.4f}  origin-T5@1.00 {5*to/(8*tb):.4f}")
P("")

# ---------------- informational: GPU-1 (other lanes) activity overlapping each pass-A process ----------------
import datetime as _dt
g1_samples = []
if os.path.exists(mon):
    for row in csv.reader(open(mon)):
        if len(row) < 3: continue
        try:
            ts = _dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            if int(row[1]) == 1: g1_samples.append((ts, float(row[2].strip().split()[0])))
        except Exception: continue
def g1_busy(rm):
    s, e = float(rm["start"]), float(rm["end"])
    v = [u for ts, u in g1_samples if s <= ts <= e]
    return (sum(1 for u in v if u > 0) / len(v)) if v else float("nan")
P("=" * 150)
P("INFORMATIONAL (post-hoc): fraction of GPU-1 monitor samples with util > 0 during each pass-A process (other lanes' jobs;")
P("  GPU 0 itself was exclusive, G4).  Listed per rep next to the s/fwd it may have perturbed (CPU/host contention).")
P("=" * 150)
for res in RES_ORDER:
    P(f"  {res[0]}x{res[1]}")
    for cfg in CFGS_A:
        cells = []
        for rep in sorted(A[(cfg, res)]):
            r = A[(cfg, res)][rep]; rm = meta.get(("A", r["label"]), {})
            cells.append(f"r{rep} {r['sec']:.4f} s (GPU1 busy {100*g1_busy(rm):3.0f} %)" if rm else f"r{rep} {r['sec']:.4f}")
        P(f"     {cfg:8s} " + "   ".join(cells))
P("")

# ---------------- informational: one-time cost per process (not gating; not in PREREG's metric list) ----------------
P("=" * 150)
P("INFORMATIONAL (post-hoc, non-gating): one-time cost per process = wall-clock (RUNMETA end-start) - 13 x steady-state s/fwd")
P("  = Python import + model build/load + warm-up excess (incl. the Mamba2 kernel build/autotune); median over reps; delta vs origin")
P("=" * 150)
for res in RES_ORDER:
    oc = {}
    for cfg in CFGS_A:
        v = []
        for rep, r in A[(cfg, res)].items():
            rm = meta.get(("A", r["label"]))
            if rm: v.append(float(rm["end"]) - float(rm["start"]) - 13 * r["sec"])
        oc[cfg] = st.median(v) if v else float("nan")
    P(f"  {res[0]}x{res[1]:<5d} " + "   ".join(f"{c} {oc[c]:6.2f} s ({oc[c]-oc['origin']:+.2f})" for c in CFGS_A))
P("")

# ---------------- pass B ----------------
B = defaultdict(lambda: defaultdict(list))   # B[(cfg,res)][slot] = [avgMs per rep]
Bclean = defaultdict(list); Bhook = defaultdict(list)
for r in resB:
    cfg, res, rep = parse_tag(r["label"])
    for n, v in r["slots"].items(): B[(cfg, res)][n].append(v["avgMs"])
    Bclean[(cfg, res)].append(r["sec_clean"]); Bhook[(cfg, res)].append(r["sec_hooked"])
slots_order = sorted(next(iter(B.values())).keys(), key=lambda n: ({"L0": 0, "L1": 1, "L2": 2, "mid": 3}[LEVEL(n)], n)) if B else []
P("=" * 150)
P("PASS B -- per-slot inclusive attn1 module time (ms per UNet forward, CUDA events), BS=1, mean over reps (+- half-range)")
P("  replaced slot = the whole GatedResidualMambaSelfAttention adapter; un-replaced = diffusers Attention (SDPA)")
P("=" * 150)
lvl_sum = {}
for res in RES_ORDER:
    if not all(B.get((c, res)) for c in CFGS_B):
        P(f"--- {res[0]}x{res[1]}: pass B incomplete ({', '.join(c for c in CFGS_B if not B.get((c, res)))} missing) -- skipped"); continue
    P(f"--- {res[0]}x{res[1]}  (tokens/frame at level 0/1/2/mid: {(res[0]//8)*(res[1]//8)} / {(res[0]//16)*(res[1]//16)} / "
      f"{(res[0]//32)*(res[1]//32)} / {(res[0]//64)*(res[1]//64)})")
    P(f"  {'slot':10s} {'lvl':4s} {'origin attn':>12s} {'+-':>6s} {'mamba10':>9s} {'+-':>6s} {'all16':>9s} {'+-':>6s} {'speedup m10':>11s} {'saving m10':>10s}")
    for n in slots_order:
        o = B[("origin", res)][n]; m = B[("mamba10", res)][n]; a = B[("all16", res)][n]
        om, mm, am = st.mean(o), st.mean(m), st.mean(a)
        P(f"  {SHORT(n):10s} {LEVEL(n):4s} {om:12.3f} {hr(o):6.3f} {mm:9.3f} {hr(m):6.3f} {am:9.3f} {hr(a):6.3f} {om/mm:10.2f}x {om-mm:10.3f}")
    for cfg in CFGS_B:
        sums = defaultdict(float)
        for n in slots_order: sums[LEVEL(n)] += st.mean(B[(cfg, res)][n])
        lvl_sum[(cfg, res)] = dict(sums)
    tc = st.mean(Bclean[("origin", res)])
    P(f"  per-level sums (ms):  " + "   ".join(
        f"{cfg}: " + " / ".join(f"{lv} {lvl_sum[(cfg, res)][lv]:.2f}" for lv in ("L0", "L1", "L2", "mid")) for cfg in CFGS_B))
    P(f"  origin attn1 share of origin's clean forward ({tc:.4f} s, pass-B phase 1): " + " / ".join(
        f"{lv} {100*lvl_sum[('origin', res)][lv]/1000/tc:.2f} %" for lv in ("L0", "L1", "L2", "mid")) +
        f"  (all 16: {100*sum(lvl_sum[('origin', res)].values())/1000/tc:.2f} %)")
    P(f"  hook overhead: origin {100*(st.mean(Bhook[('origin',res)])/tc-1):+.2f} %, mamba10 "
      f"{100*(st.mean(Bhook[('mamba10',res)])/st.mean(Bclean[('mamba10',res)])-1):+.2f} %, all16 "
      f"{100*(st.mean(Bhook[('all16',res)])/st.mean(Bclean[('all16',res)])-1):+.2f} %;  pass-B clean vs pass-A mean: "
      + ", ".join(f"{c} {100*(st.mean(Bclean[(c,res)])/mean(c,res)-1):+.2f} %" for c in CFGS_B))
    P("")

P("=" * 150)
P("CEILING -- the extra saving a ZERO-COST level-1 mixer would buy:  C = sum(level-1 origin attn1 ms) / t(mamba5)")
P("=" * 150)
ceil = {}
for res in RES_ORDER:
    if (("origin", res)) not in lvl_sum: continue
    s = summary[res]; L1o = lvl_sum[("origin", res)]["L1"] / 1000; L1m = lvl_sum[("mamba10", res)]["L1"] / 1000
    C = L1o / s["t5"]; ceil[res] = C
    attrib = (L1o - L1m); total = s["t5"] - s["t10"]
    P(f"  {res[0]}x{res[1]:<5d} level-1 attention {1000*L1o:7.2f} ms  -> C = {100*C:5.2f} % of t(mamba5)  (C_pts = {100*L1o/s['to']:.2f} pt of origin)")
    P(f"            level-1 Mamba (dim 640) {1000*L1m:7.2f} ms -> per-slot speedup {L1o/L1m:.2f}x;  slot-level saving {1000*attrib:7.2f} ms vs "
      f"pass-A total saving t5-t10 {1000*total:7.2f} ms ({100*total/attrib if attrib else float('nan'):.0f} % of the slot-level saving)")
    L2o = lvl_sum[("origin", res)]["L2"] / 1000 + lvl_sum[("origin", res)]["mid"] / 1000
    L2m = lvl_sum[("all16", res)]["L2"] / 1000 + lvl_sum[("all16", res)]["mid"] / 1000
    P(f"            (informational) level-2+mid attention {1000*L2o:7.2f} ms vs Mamba {1000*L2m:7.2f} ms -> speedup {L2o/L2m:.2f}x")
P("")

P("=" * 150)
P("VERDICT (PREREG rule: PASS iff E >= 10 % at 1024x1792 AND a plausible distillation path; C < 10 % -> closed on speed)")
P("=" * 150)
k = (1024, 1792)
if k in summary:
    E = summary[k]["E"]; C = ceil.get(k)
    speed_ok = E >= 0.10
    P(f"  E(1024x1792) = {100*E:+.2f} %  -> speed criterion {'MET' if speed_ok else 'NOT MET'} (bar 10 %)")
    if C is not None:
        P(f"  C(1024x1792) = {100*C:.2f} %  -> {'a zero-cost level-1 mixer could reach the bar' if C >= 0.10 else 'NO level-1 mixer of any cost can reach the bar: CLOSED on speed'}")
    if g3_fail_res: P(f"  NOTE: G3 failed at {g3_fail_res} -> those rows are flagged 'possibly contaminated' (paired ratios still shown)")
print("\n".join(out))
