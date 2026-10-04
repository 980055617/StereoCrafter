#!/usr/bin/env python
"""finalcheck_20261004 / bench lane -- analysis of the DIAGNOSTIC dtype arm (PREREG_ADDENDUM_dtype.txt). CPU only.

usage: analyze_dtype_v1.py DTYPE_RUN_DIR FP16_RUN_DIR [TAG]
  DTYPE_RUN_DIR  outputs/finalcheck_20261004/bench/dtype_v1   (bench2_dtype_diag.py, DT=bf16 + fp16 copy-identity control)
  FP16_RUN_DIR   outputs/finalcheck_20261004/bench/run_v1     (the primary, unmodified bench2.py run)
writes scripts/distill/runs/finalcheck_20261004/bench/TABLE_DTYPE_<TAG>.txt
All of it is POST-HOC and NON-GATING; the primary answers stay those of TABLE_BENCH_v1 / TABLE_COMBINED_v1.
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
TABLE_DIR = os.environ.get("TABLE_DIR", LANE)
SPEED_LOG = "outputs/finalcheck_20261004/speed/timing_gpu0.txt"
RUN, FP16RUN = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
TAG = sys.argv[3] if len(sys.argv) > 3 else "v1"
RES = [(576, 1024), (1024, 1792), (1024, 1920)]
RN = {r: f"{r[0]}x{r[1]}" for r in RES}
MODELS = ["origin", "mamba5"]
EXPECT = {"origin": ({"origin_attn": 0, "mamba_core": 0, "plain_attn1": 16}, [], "0"),
          "mamba5": ({"origin_attn": 0, "mamba_core": 5, "plain_attn1": 11}, [1.0], "5")}
OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def load_run(run, tagre):
    results, meta = {}, {}
    for line in open(f"{run}/bench_results.txt"):
        line = line.rstrip("\n")
        if line.startswith("RESULT "):
            j = json.loads(line[7:])
            results[j["label"]] = j
        elif line.startswith("RUNMETA "):
            d = dict(kv.split("=", 1) for kv in line[8:].split())
            meta[d["tag"]] = d
    proc = {}
    for tag, d in meta.items():
        m = tagre.match(tag)
        if not m:
            continue
        g = m.groupdict()
        key = (g["model"], g.get("dt") or "fp16", int(g["bs"]), (int(g["h"]), int(g["w"])), int(g["rep"]))
        ok = d.get("rc") == "0" and d.get("has_result") == "1" and tag in results
        retry = bool(g.get("retry"))
        if ok and (key not in proc or (proc[key]["retry"] and not retry)):
            proc[key] = dict(tag=tag, retry=retry, res=results[tag], meta=d)
    return proc


DTRE = re.compile(r"^(?P<model>origin|mamba5)_(?P<dt>fp16|bf16)_bs(?P<bs>\d)_h(?P<h>\d+)w(?P<w>\d+)_r(?P<rep>\d+)(?P<retry>_retry)?$")
V1RE = re.compile(r"^(?P<model>origin|mamba5)_bs(?P<bs>\d)_h(?P<h>\d+)w(?P<w>\d+)_r(?P<rep>\d+)(?P<retry>_retry)?$")
proc = load_run(RUN, DTRE)
p16 = load_run(FP16RUN, V1RE)

# exclusivity (same rule as PREREG B4)
mon = []
for line in open(f"{RUN}/gpu_monitor.csv"):
    p = [x.strip() for x in line.split(",")]
    try:
        mon.append(dict(t=datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), gpu=int(p[1]),
                        util=float(p[2].split()[0]), mem=float(p[3].split()[0])))
    except (ValueError, IndexError):
        continue
pre_apps, cur = {}, None
for line in open(f"{RUN}/exclusivity_checks.txt"):
    line = line.rstrip("\n")
    m = re.match(r"^(PRE|POST) (\S+) ", line)
    if m:
        cur = (m.group(1), m.group(2))
        if cur[0] == "PRE":
            pre_apps[cur[1]] = []
        continue
    if cur and cur[0] == "PRE" and line.startswith("GPU-"):
        pre_apps[cur[1]].append(line)

P("=" * 160)
P("BENCH LANE -- DIAGNOSTIC dtype arm (POST-HOC, NON-GATING; PREREG_ADDENDUM_dtype.txt)")
P(f"run dir {RUN}; fp16 reference = the primary run {FP16RUN} (unmodified bench2.py)")
for f in ("versions.txt", "bench2_md5.txt", "diag_copy_diff.txt"):
    if os.path.exists(f"{RUN}/{f}"):
        P(f"  {f}:")
        for l in open(f"{RUN}/{f}").read().strip().splitlines():
            P(f"     {l}")
P("=" * 160)
b1, b4 = [], []
for key, pr in sorted(proc.items(), key=lambda kv: kv[1]["meta"]["start"]):
    model = key[0]
    r, d = pr["res"], pr["meta"]
    calls, gates, repl = EXPECT[model]
    res = key[3]
    bad = []
    if r["batch"] != key[2]:
        bad.append("batch")
    if r["tokens"] != (res[0] // 8) * (res[1] // 8):
        bad.append("tokens")
    if r["per_fwd_calls"] != calls or r["gates"] != gates or d.get("total_replaced") != repl:
        bad.append(f"calls/gates/repl {r['per_fwd_calls']} {r['gates']} {d.get('total_replaced')}")
    if bad:
        b1.append((pr["tag"], bad))
    t0, t1 = float(d["start"]), float(d["end"])
    s1 = [s for s in mon if s["gpu"] == 1 and t0 <= s["t"] <= t1]
    foreign = [a for a in pre_apps.get(pr["tag"], []) if "snapd-desktop-integration" not in a]
    if not s1 or max(s["util"] for s in s1) > 0 or max(s["mem"] for s in s1) > 100 or foreign:
        b4.append(pr["tag"])
exp_keys = [(m, "bf16", b, r, p) for p in (1, 2, 3) for b in (2, 1) for m in MODELS for r in RES] + \
           [(m, "fp16", 1, RES[0], p) for p in (1, 2, 3) for m in MODELS]
missing = [k for k in exp_keys if k not in proc]
P(f"  processes: {len(proc)} of {len(exp_keys)} expected; missing {missing or 'none'}; retries "
  f"{[p['tag'] for p in proc.values() if p['retry']] or 'none'}")
P(f"  B1 instrumentation: {'PASS' if not b1 and not missing else 'FAIL ' + str(b1)}")
P(f"  B4 exclusivity: {'PASS' if not b4 else 'FAIL ' + str(b4)}   (GPU1 sampler n="
  f"{len([s for s in mon if s['gpu'] == 1])}, max util {max((s['util'] for s in mon if s['gpu'] == 1), default=float('nan')):.0f} %)")

T = defaultdict(dict)
PK = defaultdict(dict)
for (model, dt, bs, res, rep), pr in proc.items():
    T[(model, dt, bs, res)][rep] = pr["res"]["sec"]
    PK[(model, dt, bs, res)][rep] = pr["res"]["peak_MiB"]
F = defaultdict(dict)
for (model, dt, bs, res, rep), pr in p16.items():
    F[(model, bs, res)][rep] = pr["res"]["sec"]


def mhr(v):
    v = list(v)
    return st.mean(v), (max(v) - min(v)) / 2 if len(v) > 1 else 0.0


P()
P("--- C0 COPY-IDENTITY: fp16 control (bench2_dtype_diag.py, DT=fp16) vs the primary run (unmodified bench2.py), BS=1 576x1024 ---")
c0 = []
for model in MODELS:
    a = T[(model, "fp16", 1, RES[0])]
    b = F[(model, 1, RES[0])]
    if not a or not b:
        c0.append(f"{model} missing")
        continue
    ma, ha = mhr(a.values())
    mb, hb = mhr(b.values())
    dev = 100 * (ma - mb) / mb
    ok = abs(dev) <= 1.5
    if not ok:
        c0.append(f"{model} {dev:+.2f}%")
    P(f"  {model:7s} diag-copy fp16 {ma:.4f} +- {ha:.4f}   primary fp16 {mb:.4f} +- {hb:.4f}   {dev:+.2f} % (tol +-1.5 %) "
      f"{'ok' if ok else 'OUT'}")
P(f"  C0: {'PASS' if not c0 else 'FAIL ' + str(c0)}")
P()
P("--- D1 bf16 per forward (mean +- half-range, n) vs the primary fp16 run ---")
P(f"  {'res':11s} {'config':7s} {'BS':>2s} {'n':>2s} {'bf16 s':>8s} {'+-':>7s} {'peak MiB':>8s} {'fp16 s':>8s} {'bf16/fp16':>9s}  "
  f"{'bf16 BS1/BS2':>12s} {'fp16 BS1/BS2':>12s}")
for res in RES:
    for bs in (2, 1):
        for model in MODELS:
            v = T[(model, "bf16", bs, res)]
            if not v:
                P(f"  {RN[res]:11s} {model:7s} {bs:2d}  NOT MEASURED")
                continue
            m, h = mhr(v.values())
            f16 = st.mean(F[(model, bs, res)].values())
            r12 = r12f = ""
            if bs == 1:
                r12 = f"{m / st.mean(T[(model, 'bf16', 2, res)].values()):.4f}"
                r12f = f"{f16 / st.mean(F[(model, 2, res)].values()):.4f}"
            P(f"  {RN[res]:11s} {model:7s} {bs:2d} {len(v):2d} {m:8.4f} {h:7.4f} {st.mean(PK[(model, 'bf16', bs, res)].values()):8.0f} "
              f"{f16:8.4f} {m / f16:9.4f}  {r12:>12s} {r12f:>12s}")
P()
P("--- D3 MAMBA SHARE t(mamba5,BS)/t(origin,BS): bf16 vs fp16 ---")
for res in RES:
    row = []
    for bs in (2, 1):
        b = st.mean(T[("mamba5", "bf16", bs, res)].values()) / st.mean(T[("origin", "bf16", bs, res)].values())
        f = st.mean(F[("mamba5", bs, res)].values()) / st.mean(F[("origin", bs, res)].values())
        row.append(f"BS{bs} bf16 {b:.4f} ({100 * (b - 1):+.2f} %) fp16 {f:.4f} ({100 * (f - 1):+.2f} %)")
    P(f"  {RN[res]:11s} " + "   |   ".join(row))
P()

# ---------------------------------------------------------------- D2: X1 with bf16 predictions
RUNRE = re.compile(r"^RUN (\S+) (\S+) model=(\S+) guid=(\S+) sigmas=(\S+) pad=(\S+) rc=(\d+) secs=(\S+) md5=(\S*) "
                   r"load1=(\S+) gpus=(\S+) (\S+) dir=(\S+)")
SR = defaultdict(list)
for line in open(SPEED_LOG):
    m = RUNRE.match(line.strip())
    if not m or int(m.group(7)) != 0:
        continue
    sp = os.path.join(m.group(13), "speed_log.json")
    if not os.path.exists(sp):
        continue
    j = json.load(open(sp))
    SR[m.group(2)].append(dict(steady_unet=st.median([w["unet_ms"] for w in j["windows"][1:-1]]) / 1000,
                               cpw=j["unet_calls_per_window"]))
ROWS = [("deployed origin 8x2 @1.01", "origin", 8, 2, "origin_g101_s8", "incl"),
        ("deliverable 8x2 @1.01", "mamba5", 8, 2, "deliv_g101_s8", "incl"),
        ("deliverable 8x1 @1.00", "mamba5", 8, 1, "deliv_g100_s8", "incl"),
        ("deliverable T6 @1.01 (6x2)", "mamba5", 6, 2, "deliv_g101_T6pad", "incl"),
        ("deliverable T5 @1.01 (5x2)", "mamba5", 5, 2, "deliv_g101_T5pad", "incl"),
        ("deliverable T5 @1.00 (5x1)", "mamba5", 5, 1, "deliv_g100_T5pad", "incl"),
        ("origin 8x1 @1.00", "origin", 8, 1, "origin_g100_s8", "incl"),
        ("origin T6 @1.01 (6x2)", "origin", 6, 2, "origin_g101_T6pad", "incl"),
        ("origin T5 @1.01 (5x2)", "origin", 5, 2, "origin_g101_T5pad", "incl"),
        ("origin T5 @1.00 (5x1)", "origin", 5, 1, "origin_g100_T5pad", "excl")]


def med(label):
    return st.median([r["steady_unet"] for r in SR[label]])


def rel(TT, model, steps, bs, res, dt=None):
    k = (lambda m_, b_: (m_, dt, b_, res)) if dt else (lambda m_, b_: (m_, b_, res))
    o2 = TT[k("origin", 2)]
    c = TT[k(model, bs)]
    reps = sorted(set(o2) & set(c))
    prs = [steps * c[r] / (8 * o2[r]) for r in reps]
    return steps * st.mean(c.values()) / (8 * st.mean(o2.values())), ((max(prs) - min(prs)) / 2 if len(prs) > 1 else 0.0)


P("=" * 160)
P("D2  X1-bf16 at 576x1024: |pred/meas - 1| <= 0.04 for every included row (same rule as PREREG X1, predictions from the bf16 bench)")
P("=" * 160)
P(f"  {'configuration':30s} {'meas rel':>8s} {'fp16 pred':>9s} {'fp16 p/m':>8s} {'bf16 pred':>9s} {'bf16 p/m':>8s}  X1-bf16")
ref = med("origin_g101_s8")
fails = []
for disp, model, steps, bs, lab, blk in ROWS:
    mrel = med(lab) / ref
    pf, _ = rel(F, model, steps, bs, RES[0])
    pb, _ = rel(T, model, steps, bs, RES[0], dt="bf16")
    ok = abs(pb / mrel - 1) <= 0.04
    if blk == "incl" and not ok:
        fails.append(disp)
    P(f"  {disp:30s} {mrel:8.4f} {pf:9.4f} {pf / mrel:8.3f} {pb:9.4f} {pb / mrel:8.3f}  "
      f"{'excluded row, info only' if blk == 'excl' else ('ok' if ok else 'OUT')}")
X1B = not fails
P(f"  X1-bf16: {'PASS -> the dtype explains the X1 gap; bf16 values below = DEPLOYED-DTYPE ESTIMATE' if X1B else 'FAIL -> ' + '; '.join(fails)}")
P()
P("=" * 160)
P("PER-WINDOW UNet TIME relative to deployed origin 8x2 @1.01 at the same resolution: published-methodology fp16 vs bf16 (deployed dtype)")
P("+- = half-range of the 3 per-rep paired ratios.  Lever rows at 1024x1792 / 1024x1920 are SPEED-ONLY (quality verified at 576x1024 only).")
P("=" * 160)
for res in RES:
    o16 = st.mean(F[("origin", 2, res)].values())
    ob = st.mean(T[("origin", "bf16", 2, res)].values())
    P(f"--- {RN[res]}  deployed origin per-window UNet: fp16 bench {8 * o16:.3f} s, bf16 bench {8 * ob:.3f} s ---")
    for disp, model, steps, bs, lab, blk in ROWS:
        pf, hf = rel(F, model, steps, bs, res)
        pb, hb = rel(T, model, steps, bs, res, dt="bf16")
        tb = st.mean(T[(model, "bf16", bs, res)].values())
        P(f"  {disp:30s} fp16 rel {pf:.4f} +- {hf:.4f}   bf16 rel {pb:.4f} +- {hb:.4f}   bf16 per-window {steps * tb:7.3f} s"
          + ("   [EXCLUDED: failed its quality criterion]" if blk == "excl" else ""))
    P()
open(f"{TABLE_DIR}/TABLE_DTYPE_{TAG}.txt", "w").write("\n".join(OUT) + "\n")
print(f"\nwrote {TABLE_DIR}/TABLE_DTYPE_{TAG}.txt")
