#!/usr/bin/env python
"""finalcheck_20261004 / bench lane -- analysis v1 (CPU only: run with CUDA_VISIBLE_DEVICES="").

usage: analyze_v1.py RUN_DIR [TAG]
  RUN_DIR  outputs/finalcheck_20261004/bench/run_v1  (bench_results.txt, gpu_monitor.csv, exclusivity_checks.txt, logs/)
  TAG      suffix of the written tables (default v1)
writes  scripts/distill/runs/finalcheck_20261004/bench/TABLE_BENCH_<TAG>.txt     (Q1 + gates B1..B5)
        scripts/distill/runs/finalcheck_20261004/bench/TABLE_COMBINED_<TAG>.txt  (Q2 + cross-check X1 + wall-clock)
Every rule applied here was fixed in PREREG.txt (same directory) before any bench process ran.
Inputs read (never written): the bench run dir; the speed lane's verdict tables (scripts/.../speed/TABLE_*.txt), its driver
log outputs/finalcheck_20261004/speed/timing_gpu0.txt and every render's speed_log.json.
"""
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
LANE = "scripts/distill/runs/finalcheck_20261004/bench"
TABLE_DIR = os.environ.get("TABLE_DIR", LANE)   # dry runs on partial data write elsewhere
SPEED = "scripts/distill/runs/finalcheck_20261004/speed"
SPEED_LOG = "outputs/finalcheck_20261004/speed/timing_gpu0.txt"
RUN = sys.argv[1].rstrip("/")
TAG = sys.argv[2] if len(sys.argv) > 2 else "v1"

RES = [(576, 1024), (1024, 1792), (1024, 1920)]
RN = {r: f"{r[0]}x{r[1]}" for r in RES}
MODELS = ["origin", "mamba5"]
REPS = [1, 2, 3]
# 2026-10-01 lane, scripts/distill/runs/slotbudget/TABLES_v1.txt (BS=2): (mean sec, peak MiB)
REF1001 = {("origin", RES[0]): (0.9641, 7396), ("mamba5", RES[0]): (0.9119, 7404),
           ("origin", RES[1]): (3.7251, 16790), ("mamba5", RES[1]): (2.9592, 16798),
           ("origin", RES[2]): (4.0745, 17783), ("mamba5", RES[2]): (3.1835, 17788)}
REF1001_DELTA = {RES[0]: -5.41, RES[1]: -20.56, RES[2]: -21.87}
EXPECT = {"origin": ({"origin_attn": 0, "mamba_core": 0, "plain_attn1": 16}, [], "0"),
          "mamba5": ({"origin_attn": 0, "mamba_core": 5, "plain_attn1": 11}, [1.0], "5")}
TAGRE = re.compile(r"^(origin|mamba5)_bs(\d)_h(\d+)w(\d+)_r(\d+)(_retry)?$")

OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def mhr(v):
    return st.mean(v), ((max(v) - min(v)) / 2 if len(v) > 1 else 0.0)


# ------------------------------------------------------------------------------------------------ bench results
results, meta = {}, {}
for line in open(f"{RUN}/bench_results.txt"):
    line = line.rstrip("\n")
    if line.startswith("RESULT "):
        j = json.loads(line[7:])
        results[j["label"]] = j
    elif line.startswith("RUNMETA "):
        d = dict(kv.split("=", 1) for kv in line[8:].split())
        meta[d["tag"]] = d
failed_lines = [l.rstrip() for l in open(f"{RUN}/bench_results.txt") if l.startswith("FAILED")]

# gpu monitor (2 s sampler over the whole bench)
mon = []
for line in open(f"{RUN}/gpu_monitor.csv"):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 7:
        continue
    try:
        t = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        mon.append(dict(t=t, gpu=int(p[1]), util=float(p[2].split()[0]), mem=float(p[3].split()[0]),
                        temp=float(p[4]), sm=float(p[5].split()[0])))
    except (ValueError, IndexError):
        continue

# compute-apps snapshot before every process
pre_apps = {}
cur = None
for line in open(f"{RUN}/exclusivity_checks.txt"):
    line = line.rstrip("\n")
    m = re.match(r"^(PRE|POST) (\S+) ", line)
    if m:
        cur = (m.group(1), m.group(2))
        if cur[0] == "PRE":
            pre_apps[cur[1]] = []
        continue
    if line == "compute-apps:" or cur is None or cur[0] != "PRE":
        continue
    if line.startswith("GPU-"):
        pre_apps[cur[1]].append(line)

# choose the process per (model, BS, res, rep): base tag if it succeeded, else its retry
proc = {}
for tag, d in meta.items():
    m = TAGRE.match(tag)
    if not m:
        continue
    model, bs, h, w, rep, retry = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)
    ok = d.get("rc") == "0" and d.get("has_result") == "1" and tag in results
    key = (model, bs, (h, w), rep)
    if ok and (key not in proc or (proc[key]["retry"] and not retry)):
        proc[key] = dict(tag=tag, retry=bool(retry), res=results[tag], meta=d)

P("=" * 150)
P(f"BENCH LANE (finalcheck_20261004) -- scripts/distill/bench2.py UNMODIFIED, synthetic UNet-only forward, 14 frames, fp16, no checkpoint")
P(f"run dir {RUN};  criteria: {LANE}/PREREG.txt (written before any bench process)")
P("each value = mean seconds per UNet forward over 10 timed forwards after 3 untimed warm-ups (one process); "
  "n=3 processes per configuration, reps-outer/configs-inner")
P("=" * 150)
for f in ("versions.txt", "bench2_md5.txt"):
    if os.path.exists(f"{RUN}/{f}"):
        P(f"  {f}: {open(f'{RUN}/{f}').read().strip()}")
env = open(f"{RUN}/env_at_start.txt").read().splitlines() if os.path.exists(f"{RUN}/env_at_start.txt") else []
leak = [e for e in env if e.startswith(("MAMBA_", "PYTORCH_CUDA_ALLOC_CONF"))]
P(f"  env at start: CUDA_VISIBLE_DEVICES={[e for e in env if e.startswith('CUDA_VISIBLE_DEVICES=')]}; "
  f"MAMBA_*/PYTORCH_CUDA_ALLOC_CONF present: {leak if leak else 'none'}")
P()

# ------------------------------------------------------------------------------------------------ per process
P("--- PER PROCESS (B1 instrumentation + B4 exclusivity) ---")
P(f"  {'tag':30s} {'sec':>7s} {'peakMiB':>8s} {'batch':>5s} {'tokens':>6s}  calls(oa/mc/pa) gates  repl  "
  f"GPU1 max util/mem during   GPU0 temp  GPU0 SM MHz   PRE apps   B1")
b1_fail, b4_fail = [], []
b4_rep_viol = defaultdict(list)
for key in sorted(proc, key=lambda k: (k[3], -k[1], MODELS.index(k[0]), RES.index(k[2]))):
    model, bs, res, rep = key
    pr = proc[key]
    r, d = pr["res"], pr["meta"]
    calls, gates, repl = EXPECT[model]
    reasons = []
    if r["batch"] != bs:
        reasons.append(f"batch {r['batch']}!={bs}")
    if r["tokens"] != (res[0] // 8) * (res[1] // 8):
        reasons.append(f"tokens {r['tokens']}")
    if r["per_fwd_calls"] != calls:
        reasons.append(f"calls {r['per_fwd_calls']}")
    if r["gates"] != gates:
        reasons.append(f"gates {r['gates']}")
    if d.get("total_replaced") != repl:
        reasons.append(f"total_replaced {d.get('total_replaced')}")
    if r["label"] != pr["tag"]:
        reasons.append("label mismatch")
    t0, t1 = float(d["start"]), float(d["end"])
    s1 = [s for s in mon if s["gpu"] == 1 and t0 <= s["t"] <= t1]
    s0 = [s for s in mon if s["gpu"] == 0 and t0 <= s["t"] <= t1]
    u1 = max((s["util"] for s in s1), default=float("nan"))
    m1 = max((s["mem"] for s in s1), default=float("nan"))
    apps = pre_apps.get(pr["tag"], ["<no PRE snapshot>"])
    foreign = [a for a in apps if "snapd-desktop-integration" not in a]
    v4 = []
    if not s1:
        v4.append("no GPU1 samples")
    elif u1 > 0 or m1 > 100:
        v4.append(f"GPU1 util {u1:.0f}% mem {m1:.0f}")
    if foreign:
        v4.append(f"foreign apps {foreign}")
    if v4:
        b4_fail.append((pr["tag"], v4))
        b4_rep_viol[rep].append(pr["tag"])
    if reasons:
        b1_fail.append((pr["tag"], reasons))
    tr = f"{min(x['temp'] for x in s0):.0f}-{max(x['temp'] for x in s0):.0f}" if s0 else "-"
    smr = f"{min(x['sm'] for x in s0):.0f}-{max(x['sm'] for x in s0):.0f}" if s0 else "-"
    c = r["per_fwd_calls"]
    P(f"  {pr['tag']:30s} {r['sec']:7.4f} {r['peak_MiB']:8d} {r['batch']:5d} {r['tokens']:6d}  "
      f"{c['origin_attn']}/{c['mamba_core']}/{c['plain_attn1']:<2d}          {str(r['gates']):6s} {d.get('total_replaced'):>4s}  "
      f"{u1:5.0f}% / {m1:5.0f} MiB (n={len(s1):3d})   {tr:>7s}   {smr:>11s}   {len(apps)} {'ok' if not foreign else 'FOREIGN'}"
      f"   {'PASS' if not reasons else 'FAIL ' + '; '.join(reasons)}")
expected_keys = [(m, b, r, p) for p in REPS for b in (2, 1) for m in MODELS for r in RES]
missing = [k for k in expected_keys if k not in proc]
P(f"  processes used: {len(proc)} of {len(expected_keys)} expected; missing: {missing if missing else 'none'}")
P(f"  retries used: {[p['tag'] for p in proc.values() if p['retry']] or 'none'};  FAILED lines in the driver log: "
  f"{failed_lines or 'none'}")
all1 = [s for s in mon if s["gpu"] == 1]
P(f"  whole-bench GPU1 sampler: n={len(all1)} samples, max util {max((s['util'] for s in all1), default=float('nan')):.0f} %, "
  f"max mem {max((s['mem'] for s in all1), default=float('nan')):.0f} MiB")
P()

# ------------------------------------------------------------------------------------------------ per configuration
T = defaultdict(dict)       # (model, bs, res) -> {rep: sec}
PK = defaultdict(dict)      # (model, bs, res) -> {rep: peak}
for (model, bs, res, rep), pr in proc.items():
    T[(model, bs, res)][rep] = pr["res"]["sec"]
    PK[(model, bs, res)][rep] = pr["res"]["peak_MiB"]

P("--- PER CONFIGURATION: mean +- half-range (max-min)/2 over n processes ---")
P(f"  {'res (HxW)':11s} {'config':7s} {'BS':>2s} {'n':>2s} {'sec/forward':>11s} {'+-half':>8s} {'half%':>6s} "
  f"{'vs origin same BS':>17s} {'peak MiB':>9s} {'dVRAM':>7s}   {'BS1/BS2 (same model)':>20s}")
B5 = []
for res in RES:
    for bs in (2, 1):
        for model in MODELS:
            v = list(T[(model, bs, res)].values())
            if not v:
                P(f"  {RN[res]:11s} {model:7s} {bs:2d}  - NOT MEASURED")
                continue
            m, h = mhr(v)
            o = st.mean(T[("origin", bs, res)].values())
            pk = st.mean(PK[(model, bs, res)].values())
            po = st.mean(PK[("origin", bs, res)].values())
            ratio = ""
            if bs == 1 and T[(model, 2, res)]:
                ratio = f"{m / st.mean(T[(model, 2, res)].values()):.4f}"
            if 100 * h / m > 1.0:
                B5.append(f"{RN[res]} {model} BS{bs}: half-range {100 * h / m:.2f} %")
            P(f"  {RN[res]:11s} {model:7s} {bs:2d} {len(v):2d} {m:11.4f} {h:8.4f} {100 * h / m:6.2f} "
              f"{100 * (m - o) / o:+16.2f}% {pk:9.0f} {100 * (pk - po) / po:+6.2f}%   {ratio:>20s}")
P()

# ------------------------------------------------------------------------------------------------ gates
P("--- GATES (PREREG.txt) ---")
P(f"  B1 instrumentation (calls, gates, total_replaced, batch, tokens, rc, RESULT): "
  f"{'PASS' if not b1_fail and not missing else 'FAIL ' + str(b1_fail) + ' missing=' + str(missing)}")
b2 = []
for (model, res), (sec, pk) in REF1001.items():
    for rep, p in PK[(model, 2, res)].items():
        if abs(p - pk) > 4:
            b2.append(f"{model} {RN[res]} r{rep}: {p} vs {pk}")
P(f"  B2 environment identity (BS=2 peak_MiB within +-4 MiB of 2026-10-01): {'PASS' if not b2 else 'FAIL ' + str(b2)}")
b3 = []
for res in RES:
    for model in MODELS:
        ref = REF1001[(model, res)][0]
        v = list(T[(model, 2, res)].values())
        if not v:
            b3.append(f"{model} {RN[res]} missing")
            continue
        m = st.mean(v)
        dev = 100 * (m - ref) / ref
        ok = abs(dev) <= 1.5
        P(f"     B3 {RN[res]:10s} {model:7s} BS2 mean {m:.4f} vs 2026-10-01 {ref:.4f}: {dev:+.2f} % "
          f"(tol +-1.5 %) {'ok' if ok else 'OUT'}")
        if not ok:
            b3.append(f"{model} {RN[res]} {dev:+.2f}%")
    mo, mm = st.mean(T[("origin", 2, res)].values()), st.mean(T[("mamba5", 2, res)].values())
    dl = 100 * (mm - mo) / mo
    okd = abs(dl - REF1001_DELTA[res]) <= 1.0
    P(f"     B3 {RN[res]:10s} Mamba delta {dl:+.2f} % vs 2026-10-01 {REF1001_DELTA[res]:+.2f} %: "
      f"{dl - REF1001_DELTA[res]:+.2f} pt (tol +-1.0 pt) {'ok' if okd else 'OUT'}")
    if not okd:
        b3.append(f"delta {RN[res]} {dl:+.2f}%")
P(f"  B3 timing reproduction: {'PASS' if not b3 else 'FAIL ' + str(b3)}")
P(f"  B4 exclusivity (GPU1 0 % util and <=100 MiB in every sample during every process; no foreign compute app before any "
  f"process): {'PASS' if not b4_fail else 'FAIL ' + str(b4_fail)}")
P(f"  B5 precision (flag half-range > 1.0 % of the mean; reported, not a gate): {B5 if B5 else 'no configuration flagged'}")
open(f"{TABLE_DIR}/TABLE_BENCH_{TAG}.txt", "w").write("\n".join(OUT) + "\n")
BENCH_OUT = OUT[:]
OUT.clear()

# ================================================================================================ COMBINATION (Q2)
VRE = re.compile(r"^\s+\[(GATING|reported)\]\s+(.*?)\s+\((\S+) vs (\S+), n=(\d+), better on (\d+)/(\d+)\): (PASS|FAIL)")
ARE = re.compile(r"^\s+\(a\) mean delta ([+-]\d\.\d+) <= \+0\.002: (PASS|FAIL)")
BRE = re.compile(r"^\s+\(b\) worst clip (\d{4}) ([+-]\d\.\d+) <= \+0\.005: (PASS|FAIL)")
verd = {}
for f in ("TABLE_STEP1_GUIDANCE.txt", "TABLE_STEP2_EXT12.txt", "TABLE_POSTHOC_T5G100_UNPADDED.txt"):
    last = None
    for line in open(f"{SPEED}/{f}"):
        m = VRE.match(line)
        if m:
            last = (m.group(3), m.group(4))
            verd[last] = dict(kind=m.group(1), name=m.group(2), verdict=m.group(8), n=int(m.group(5)), file=f)
            continue
        m = ARE.match(line)
        if m and last:
            verd[last]["mean"] = m.group(1)
            continue
        m = BRE.match(line)
        if m and last:
            verd[last]["worst"] = f"{m.group(1)} {m.group(2)}"
            last = None
# the deliverable's own headline row (TABLE_STEP1_GUIDANCE.txt, MEANS section)
deliv_head = None
for line in open(f"{SPEED}/TABLE_STEP1_GUIDANCE.txt"):
    m = re.match(r"^\s+deliverable @1\.01 \(8x2\)\s+(0\.\d{4})\s+([+-]0\.\d{4})\s+(\d+/\d+)", line)
    if m:
        deliv_head = m.groups()

D, O = "mstudent2_step800_deliv_ll", "origin_ll"
# key, display, model, steps, BS, speed-lane label, verdict key, block, expected verdict (PREREG)
ROWS = [
    ("o82", "deployed origin 8x2 @1.01", "origin", 8, 2, "origin_g101_s8", None, "ref", None),
    ("d82", "deliverable 8x2 @1.01", "mamba5", 8, 2, "deliv_g101_s8", "HEAD", "deliv", None),
    ("d81", "deliverable 8x1 @1.00", "mamba5", 8, 1, "deliv_g100_s8", ("deliv_g100_s8", D), "deliv", "PASS"),
    ("d62", "deliverable T6 @1.01 (6x2)", "mamba5", 6, 2, "deliv_g101_T6pad", ("deliv_g101_T6pad", D), "deliv", "PASS"),
    ("d52", "deliverable T5 @1.01 (5x2)", "mamba5", 5, 2, "deliv_g101_T5pad", ("deliv_g101_T5pad", D), "deliv", "PASS"),
    ("d51", "deliverable T5 @1.00 (5x1)", "mamba5", 5, 1, "deliv_g100_T5pad", ("deliv_g100_T5pad", D), "deliv", "PASS"),
    ("o81", "origin 8x1 @1.00", "origin", 8, 1, "origin_g100_s8", ("origin_g100_s8", O), "origin", "PASS"),
    ("o62", "origin T6 @1.01 (6x2)", "origin", 6, 2, "origin_g101_T6pad", ("origin_g101_T6pad", O), "origin", "PASS"),
    ("o52", "origin T5 @1.01 (5x2)", "origin", 5, 2, "origin_g101_T5pad", ("origin_g101_T5pad", O), "origin", "PASS"),
    ("o51", "origin T5 @1.00 (5x1)", "origin", 5, 1, "origin_g100_T5pad", ("origin_g100_T5pad", O), "excl", "FAIL"),
]


def vtext(row):
    vk = row[6]
    if vk is None:
        return "reference (deployed)"
    if vk == "HEAD":
        return (f"the deliverable: {deliv_head[0]}, {deliv_head[1]} vs origin, {deliv_head[2]} clips" if deliv_head
                else "the deliverable (headline row not parsed)")
    v = verd.get(vk)
    if not v:
        return "VERDICT NOT FOUND"
    return f"{v['kind']} {v['verdict']} ({v.get('mean', '?')}, worst {v.get('worst', '?')}; {v['file']})"


P("=" * 170)
P("Q2 -- EFFECTIVE UNet TIME PER 14-FRAME DENOISING WINDOW, relative to DEPLOYED ORIGIN (8 steps x batch 2), per resolution")
P("per-window = steps x t(model, BS)   [BS = 2 for guidance 1.01, 1 for guidance 1.00];   rel = per-window / (8 x t(origin, BS=2))")
P("rel = ratio of the n=3 means; +- = half-range of the 3 per-rep paired ratios (every rep holds every configuration)")
P("factorisation  rel = step share (steps/8) x guidance share t(origin,BS)/t(origin,2) x Mamba share t(model,BS)/t(origin,BS)")
P("=" * 170)
P("verdicts parsed from the speed lane (inclusion rule fixed in PREREG.txt):")
mism = []
for row in ROWS:
    v = verd.get(row[6]) if isinstance(row[6], tuple) else None
    exp = row[8]
    flag = ""
    if exp and v and v["verdict"] != exp:
        flag = f"   <-- MISMATCH with the PREREG expectation {exp}; the parsed verdict is followed"
        mism.append(row[1])
    P(f"  {row[1]:30s} {vtext(row)}{flag}")
un = verd.get(("deliv_g100_T5nat", D))
if un:
    P(f"  note: deliverable T5 @1.00 UNPADDED (the literal shipped sampler; same 5x1 UNet work as the padded row): "
      f"{un['kind']} {un['verdict']} ({un.get('mean')}, worst {un.get('worst')}; {un['file']})")
P("  not tested: T6 @1.00 (the speed lane did not render it) -> no row")
P()


def rel_stats(model, steps, bs, res):
    """ratio of means and the per-rep paired ratios"""
    o2 = T[("origin", 2, res)]
    c = T[(model, bs, res)]
    reps = sorted(set(o2) & set(c))
    pr = [steps * c[r] / (8 * o2[r]) for r in reps]
    point = steps * st.mean(c.values()) / (8 * st.mean(o2.values()))
    return point, ((max(pr) - min(pr)) / 2 if len(pr) > 1 else 0.0), pr


def share_stats(num_key, den_key):
    a, b = T[num_key], T[den_key]
    reps = sorted(set(a) & set(b))
    pr = [a[r] / b[r] for r in reps]
    return st.mean(a.values()) / st.mean(b.values()), ((max(pr) - min(pr)) / 2 if len(pr) > 1 else 0.0)


COMB = {}
for res in RES:
    o2 = st.mean(T[("origin", 2, res)].values())
    P(f"--- {RN[res]} (HxW)   deployed origin per-window UNet = 8 x {o2:.4f} = {8 * o2:.3f} s ---"
      + ("" if res == RES[0] else "   [lever rows: SPEED-ONLY at this resolution -- their quality was verified at 576x1024 only]"))
    P(f"  {'configuration':30s} {'elem/win':>8s} {'s/forward':>9s} {'per-window s':>12s} {'rel':>7s} {'+-':>6s}  "
      f"{'step':>5s} x {'guid.':>6s} {'+-':>6s} x {'Mamba':>6s} {'+-':>6s}   quality status")
    for row in ROWS:
        key, disp, model, steps, bs, slabel, vk, block, exp = row
        if not T[(model, bs, res)]:
            P(f"  {disp:30s}  NOT MEASURED")
            continue
        t = st.mean(T[(model, bs, res)].values())
        rel, relh, prs = rel_stats(model, steps, bs, res)
        g, gh = share_stats(("origin", bs, res), ("origin", 2, res))
        if model == "origin":
            ms, msh = 1.0, 0.0
        else:
            ms, msh = share_stats((model, bs, res), ("origin", bs, res))
        COMB[(key, res)] = dict(rel=rel, relh=relh, per_window=steps * t, t=t, g=g, ms=ms, prs=prs)
        if block == "excl":
            continue
        if block == "ref":
            status = "reference"
        elif key == "d82":
            status = "deliverable; quality covered here" if res == RES[0] else \
                "deliverable; hi-res quality TRANSFERS (validate/V2_HIRES_TABLE.txt, 4 clips)"
        else:
            status = ("PASS @576x1024" if res == RES[0] else "SPEED-ONLY (quality not tested at this res)")
            if block == "origin":
                status = "origin lever, reported " + status
        P(f"  {disp:30s} {steps * bs:8d} {t:9.4f} {steps * t:12.3f} {rel:7.4f} {relh:6.4f}  {steps / 8:5.3f} x {g:6.4f} "
          f"{gh:6.4f} x {ms:6.4f} {msh:6.4f}   {status}")
    ex = COMB.get(("o51", res))
    if ex:
        P(f"  EXCLUDED  origin T5 @1.00 (5x1): FAILED its criterion (+0.0021, worst 0301 +0.0065) -> not claimable "
          f"[speed-only value, shown for the separability argument: rel {ex['rel']:.4f}]")
    P()

# Mamba share summary (architecture only, same sampler setting)
P("--- MAMBA SHARE per forward (t(mamba5,BS)/t(origin,BS), same batch), steady state ---")
for res in RES:
    s2, h2 = share_stats(("mamba5", 2, res), ("origin", 2, res))
    s1, h1 = share_stats(("mamba5", 1, res), ("origin", 1, res))
    P(f"  {RN[res]:11s} BS2 {s2:.4f} +- {h2:.4f} ({100 * (s2 - 1):+.2f} %)    BS1 {s1:.4f} +- {h1:.4f} ({100 * (s1 - 1):+.2f} %)")
P("  (bench2.py excludes warm-up.  On the deployed path the deliverable pays a one-time first-window warm-up of "
  "+4.8..+5.2 s per process at 576x1024 vs origin +0.1 s, speed/TABLE_TIMING_final.txt FIRST-WINDOW section;"
  " not measured at hi-res by any lane.)")
P()

# ------------------------------------------------------------------------------------------------ speed lane data
RUNRE = re.compile(r"^RUN (\S+) (\S+) model=(\S+) guid=(\S+) sigmas=(\S+) pad=(\S+) rc=(\d+) secs=(\S+) md5=(\S*) "
                   r"load1=(\S+) gpus=(\S+) (\S+) dir=(\S+)")
SR = defaultdict(list)
for line in open(SPEED_LOG):
    m = RUNRE.match(line.strip())
    if not m:
        continue
    clip, label, model, guid, sig, pad, rc, secs, md5, load1, gpus, hhmm, d = m.groups()
    sp = os.path.join(d, "speed_log.json")
    if int(rc) != 0 or not os.path.exists(sp):
        continue
    j = json.load(open(sp))
    W = j["windows"]
    g1 = re.search(r"1,(\d+)%", gpus)
    SR[label].append(dict(clip=clip, process_s=float(secs), inloop_s=j["call_s_sum"], unet_s=j["unet_ms_sum"] / 1000,
                          n_windows=j["n_windows"], calls_pw=j["unet_calls_per_window"], batch=j["batch_sizes"],
                          steady_unet=st.median([w["unet_ms"] for w in W[1:-1]]) / 1000,
                          steady_call=st.median([w["call_s"] for w in W[1:-1]]),
                          gpu1_busy=(int(g1.group(1)) > 0) if g1 else None, load1=float(load1), at=hhmm))


def med(label, f):
    return st.median([f(r) for r in SR[label]])


P("=" * 170)
P("X1 CROSS-CHECK at 576x1024: bench-predicted rel vs the speed lane's MEASURED steady-state UNet s/window on the deployed bf16 path")
P("measured = median over renders of (median over windows 1..n-2 of CUDA-event UNet ms), from every rc=0 render's speed_log.json,")
P("           divided by the same for origin_g101_s8;  PASS iff |pred/meas - 1| <= 0.04 for every included row")
P("=" * 170)
ref_meas = med("origin_g101_s8", lambda r: r["steady_unet"])
P("  meas s/forward = deployed bf16 pipeline (CUDA events); bench s/forward = synthetic fp16 bench2.py; dep/bench = their ratio")
P(f"  {'configuration':30s} {'speed label':18s} {'n':>3s} {'meas UNet s/win':>15s} {'meas s/forward':>14s} "
  f"{'bench s/forward':>15s} {'dep/bench':>9s} {'meas rel':>8s} {'pred rel':>8s} {'pred/meas':>9s}  X1")
x1_fail = []
for row in ROWS:
    key, disp, model, steps, bs, slabel, vk, block, exp = row
    if not SR.get(slabel) or (key, RES[0]) not in COMB:
        continue
    mu = med(slabel, lambda r: r["steady_unet"])
    cpw = sorted({x for r in SR[slabel] for x in r["calls_pw"]})
    per_fwd = mu / cpw[0]
    mrel = mu / ref_meas
    prel = COMB[(key, RES[0])]["rel"]
    ratio = prel / mrel
    bench_t = COMB[(key, RES[0])]["t"]
    ok = abs(ratio - 1) <= 0.04
    tagx = "excluded row, info only" if block == "excl" else ("ok" if ok else "OUT")
    if block != "excl" and not ok:
        x1_fail.append(f"{disp}: pred/meas {ratio:.3f}")
    P(f"  {disp:30s} {slabel:18s} {len(SR[slabel]):3d} {mu:15.3f} {per_fwd:14.4f} {bench_t:15.4f} {per_fwd / bench_t:9.3f} "
      f"{mrel:8.4f} {prel:8.4f} {ratio:9.3f}  {tagx}")
P(f"  X1: {'PASS' if not x1_fail else 'FAIL -> ' + '; '.join(x1_fail)}")
if x1_fail:
    P("  consequence (pre-registered): the hi-res rows are 'fp16 synthetic only; transfer to the deployed path not shown'")
P()

P("=" * 170)
P("WALL-CLOCK from the speed lane (576x1024 ONLY; GPU 0, one render at a time, but NOT machine-exclusive: GPU 1 ran other lanes)")
P("process = python start->exit (load + encode + sampling + VAE decode + FFV1);  in-loop = CUDA-synced pipeline __call__ sum;"
  "  UNet = CUDA-event sum (whole clip, includes the deliverable's ~5 s first-window warm-up)")
P("=" * 170)
P(f"  {'configuration':30s} {'speed label':18s} {'n':>3s} {'process':>8s} {'in-loop':>8s} {'UNet':>7s}   "
  f"{'median ratio vs deployed origin 8x2 (n=2)':>42s}   paired vs origin_g101_s8 same clip: process / in-loop / UNet   GPU1 util>0 in the driver's pre-render snapshot")
oref = "origin_g101_s8"
mp, mi, mun = med(oref, lambda r: r["process_s"]), med(oref, lambda r: r["inloop_s"]), med(oref, lambda r: r["unet_s"])
byclip = {r["clip"]: r for r in SR[oref]}
for row in ROWS:
    key, disp, model, steps, bs, slabel, vk, block, exp = row
    if not SR.get(slabel):
        continue
    p_, i_, u_ = med(slabel, lambda r: r["process_s"]), med(slabel, lambda r: r["inloop_s"]), med(slabel, lambda r: r["unet_s"])
    paired = []
    for r in sorted(SR[slabel], key=lambda r: r["clip"]):
        if r["clip"] in byclip and slabel != oref:
            b = byclip[r["clip"]]
            paired.append(f"{r['clip']} {r['process_s'] / b['process_s']:.3f}/{r['inloop_s'] / b['inloop_s']:.3f}/"
                          f"{r['unet_s'] / b['unet_s']:.3f}")
    busy = [r["gpu1_busy"] for r in SR[slabel] if r["gpu1_busy"] is not None]
    excl = " [EXCLUDED row]" if block == "excl" else ""
    P(f"  {disp:30s} {slabel:18s} {len(SR[slabel]):3d} {p_:8.1f} {i_:8.1f} {u_:7.1f}   "
      f"{p_ / mp:12.3f} / {i_ / mi:5.3f} / {u_ / mun:5.3f}             {'  '.join(paired) if paired else '-':60s}  "
      f"{sum(busy)}/{len(busy)}{excl}")
P("  hi-res (1024x1792, 1024x1920) wall-clock: NOT MEASURED for any lever configuration; not extrapolated.")
P()
P("NOTES")
P("  * Every quality verdict above is the speed lane's (12 clips, lossless FFV1, 576x1024); this lane measures speed only.")
P("  * The speed lane's recommendation (deliverable T5 @1.00) is conditional on the flicker / hi-res / blind checks being re-run at")
P("    T5 @1.00 (they were run at 8x2 @1.01) -- unchanged by this lane.")
if mism:
    P(f"  * VERDICT MISMATCHES vs PREREG expectation: {mism}")
open(f"{TABLE_DIR}/TABLE_COMBINED_{TAG}.txt", "w").write("\n".join(OUT) + "\n")
print(f"\nwrote {TABLE_DIR}/TABLE_BENCH_{TAG}.txt and {TABLE_DIR}/TABLE_COMBINED_{TAG}.txt")
