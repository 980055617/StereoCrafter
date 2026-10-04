#!/usr/bin/env python
"""finalcheck_20261004 / bench lane -- analysis of the DIAGNOSTIC "sync" arm (PREREG_ADDENDUM_sync.txt). CPU only.

usage: analyze_sync_v1.py SYNC_RUN_DIR FP16_RUN_DIR [TAG]
  SYNC_RUN_DIR  outputs/finalcheck_20261004/bench/sync_v1   (bench2_sync_diag.py: host sync after every timed forward)
  FP16_RUN_DIR  outputs/finalcheck_20261004/bench/run_v1    (the primary, unmodified bench2.py run)
writes scripts/distill/runs/finalcheck_20261004/bench/TABLE_SYNC_<TAG>.txt
POST-HOC and NON-GATING; the primary answers stay those of TABLE_BENCH_v1 / TABLE_COMBINED_v1.
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
TAGRE = re.compile(r"^(?P<model>origin|mamba5)_bs(?P<bs>\d)_h(?P<h>\d+)w(?P<w>\d+)_r(?P<rep>\d+)(?P<retry>_retry)?$")
OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


def load_run(run):
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
        m = TAGRE.match(tag)
        if not m:
            continue
        g = m.groupdict()
        key = (g["model"], int(g["bs"]), (int(g["h"]), int(g["w"])), int(g["rep"]))
        ok = d.get("rc") == "0" and d.get("has_result") == "1" and tag in results
        retry = bool(g.get("retry"))
        if ok and (key not in proc or (proc[key]["retry"] and not retry)):
            proc[key] = dict(tag=tag, retry=retry, res=results[tag], meta=d)
    return proc


proc, p16 = load_run(RUN), load_run(FP16RUN)
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
P("BENCH LANE -- DIAGNOSTIC 'sync' arm (POST-HOC, NON-GATING; PREREG_ADDENDUM_sync.txt): host sync after every timed forward")
P(f"run dir {RUN}; unsynced reference = the primary run {FP16RUN} (unmodified bench2.py)")
for f in ("versions.txt", "bench2_md5.txt", "diag_copy_diff.txt"):
    if os.path.exists(f"{RUN}/{f}"):
        P(f"  {f}:")
        for l in open(f"{RUN}/{f}").read().strip().splitlines():
            P(f"     {l}")
P("=" * 160)
b1, b4 = [], []
for key, pr in proc.items():
    r, d = pr["res"], pr["meta"]
    calls, gates, repl = EXPECT[key[0]]
    res = key[2]
    if (r["batch"] != key[1] or r["tokens"] != (res[0] // 8) * (res[1] // 8) or r["per_fwd_calls"] != calls
            or r["gates"] != gates or d.get("total_replaced") != repl):
        b1.append(pr["tag"])
    t0, t1 = float(d["start"]), float(d["end"])
    s1 = [s for s in mon if s["gpu"] == 1 and t0 <= s["t"] <= t1]
    foreign = [a for a in pre_apps.get(pr["tag"], []) if "snapd-desktop-integration" not in a]
    if not s1 or max(s["util"] for s in s1) > 0 or max(s["mem"] for s in s1) > 100 or foreign:
        b4.append(pr["tag"])
exp_keys = [(m, b, r, p) for p in (1, 2, 3) for b in (2, 1) for m in MODELS for r in RES]
missing = [k for k in exp_keys if k not in proc]
P(f"  processes: {len(proc)} of {len(exp_keys)}; missing {missing or 'none'}; retries "
  f"{[p['tag'] for p in proc.values() if p['retry']] or 'none'}")
P(f"  B1 instrumentation: {'PASS' if not b1 and not missing else 'FAIL ' + str(b1)}")
P(f"  B4 exclusivity: {'PASS' if not b4 else 'FAIL ' + str(b4)}   (GPU1 sampler n={len([s for s in mon if s['gpu'] == 1])}, "
  f"max util {max((s['util'] for s in mon if s['gpu'] == 1), default=float('nan')):.0f} %)")

S, F = defaultdict(dict), defaultdict(dict)
for (model, bs, res, rep), pr in proc.items():
    S[(model, bs, res)][rep] = pr["res"]["sec"]
for (model, bs, res, rep), pr in p16.items():
    F[(model, bs, res)][rep] = pr["res"]["sec"]


def mhr(v):
    v = list(v)
    return st.mean(v), (max(v) - min(v)) / 2 if len(v) > 1 else 0.0


P()
P("--- per forward: synced vs unsynced (mean +- half-range, n=3) ---")
P(f"  {'res':11s} {'config':7s} {'BS':>2s} {'synced s':>9s} {'+-':>7s} {'unsynced s':>10s} {'synced-unsynced ms':>18s} {'ratio':>7s}")
for res in RES:
    for bs in (2, 1):
        for model in MODELS:
            if not S[(model, bs, res)]:
                P(f"  {RN[res]:11s} {model:7s} {bs:2d}  NOT MEASURED")
                continue
            ms_, hs = mhr(S[(model, bs, res)].values())
            mu = st.mean(F[(model, bs, res)].values())
            P(f"  {RN[res]:11s} {model:7s} {bs:2d} {ms_:9.4f} {hs:7.4f} {mu:10.4f} {1000 * (ms_ - mu):18.1f} {ms_ / mu:7.4f}")
P()

# deployed measurements (speed lane)
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
    su = st.median([w["unet_ms"] for w in j["windows"][1:-1]]) / 1000
    SR[m.group(2)].append(dict(steady_unet=su, per_fwd=su / j["unet_calls_per_window"][0]))


def med(label, f="steady_unet"):
    return st.median([r[f] for r in SR[label]])


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


def rel(TT, model, steps, bs, res):
    o2, c = TT[("origin", 2, res)], TT[(model, bs, res)]
    reps = sorted(set(o2) & set(c))
    prs = [steps * c[r] / (8 * o2[r]) for r in reps]
    return steps * st.mean(c.values()) / (8 * st.mean(o2.values())), ((max(prs) - min(prs)) / 2 if len(prs) > 1 else 0.0)


P("=" * 160)
P("X1-sync at 576x1024: |pred/meas - 1| <= 0.04 for EVERY included row (BS=2 rows are explicit requirements)")
P("=" * 160)
P(f"  {'configuration':30s} {'meas rel':>8s} {'unsync pred':>11s} {'p/m':>6s} {'sync pred':>9s} {'p/m':>6s}  X1-sync")
ref = med("origin_g101_s8")
fails = []
for disp, model, steps, bs, lab, blk in ROWS:
    mrel = med(lab) / ref
    pu, _ = rel(F, model, steps, bs, RES[0])
    ps, _ = rel(S, model, steps, bs, RES[0])
    ok = abs(ps / mrel - 1) <= 0.04
    if blk == "incl" and not ok:
        fails.append(f"{disp} {ps / mrel:.3f}")
    P(f"  {disp:30s} {mrel:8.4f} {pu:11.4f} {pu / mrel:6.3f} {ps:9.4f} {ps / mrel:6.3f}  "
      f"{'excluded row, info only' if blk == 'excl' else ('ok' if ok else 'OUT')}")
XS = not fails
P(f"  X1-sync: {'PASS -> the per-step host sync explains the X1 gap' if XS else 'FAIL -> ' + '; '.join(fails)}")
P()
P("  absolute per forward, 576x1024 (information): deployed (speed lane, bf16, CUDA events) vs synced bench vs unsynced bench")
for name, model, bs, lab in (("origin BS2", "origin", 2, "origin_g101_s8"), ("mamba5 BS2", "mamba5", 2, "deliv_g101_s8"),
                             ("origin BS1", "origin", 1, "origin_g100_s8"), ("mamba5 BS1", "mamba5", 1, "deliv_g100_s8")):
    d = med(lab, "per_fwd")
    s_ = st.mean(S[(model, bs, RES[0])].values())
    u_ = st.mean(F[(model, bs, RES[0])].values())
    P(f"    {name:11s} deployed {d:.4f}   synced {s_:.4f} (x{s_ / d:.3f})   unsynced {u_:.4f} (x{u_ / d:.3f})")
P()
P("=" * 160)
P("PER-WINDOW UNet TIME relative to deployed origin 8x2 @1.01: published methodology (unsynced fp16) vs "
  + ("DEPLOYED-LOOP ESTIMATE (sync bench)" if XS else "sync bench (mechanism NOT confirmed: not an estimate)"))
P("+- = half-range of the 3 per-rep paired ratios.  Lever rows at 1024x1792 / 1024x1920: SPEED-ONLY (quality verified at 576x1024 only).")
P("=" * 160)
for res in RES:
    P(f"--- {RN[res]}   deployed origin per-window UNet: unsynced {8 * st.mean(F[('origin', 2, res)].values()):.3f} s, "
      f"synced {8 * st.mean(S[('origin', 2, res)].values()):.3f} s ---")
    for disp, model, steps, bs, lab, blk in ROWS:
        pu, hu = rel(F, model, steps, bs, res)
        ps, hs = rel(S, model, steps, bs, res)
        extra = ""
        if res == RES[0]:
            extra = f"   measured (deployed) {med(lab) / ref:.4f}"
        P(f"  {disp:30s} unsynced {pu:.4f} +- {hu:.4f}   synced {ps:.4f} +- {hs:.4f}{extra}"
          + ("   [EXCLUDED: failed its quality criterion]" if blk == "excl" else ""))
    P()
P("Mamba share t(mamba5,BS)/t(origin,BS), synced vs unsynced:")
for res in RES:
    row = []
    for bs in (2, 1):
        s_ = st.mean(S[("mamba5", bs, res)].values()) / st.mean(S[("origin", bs, res)].values())
        u_ = st.mean(F[("mamba5", bs, res)].values()) / st.mean(F[("origin", bs, res)].values())
        row.append(f"BS{bs} synced {s_:.4f} ({100 * (s_ - 1):+.2f} %) unsynced {u_:.4f} ({100 * (u_ - 1):+.2f} %)")
    P(f"  {RN[res]:11s} " + "   |   ".join(row))
open(f"{TABLE_DIR}/TABLE_SYNC_{TAG}.txt", "w").write("\n".join(OUT) + "\n")
print(f"\nwrote {TABLE_DIR}/TABLE_SYNC_{TAG}.txt")
