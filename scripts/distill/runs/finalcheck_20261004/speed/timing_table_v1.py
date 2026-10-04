#!/usr/bin/env python
"""Wall-clock table for the speed lane, built ONLY from the driver log + each render's speed_log.json.

usage: timing_table_v1.py OUT.txt TIMING_LOG [TIMING_LOG ...]
  process_s  driver wall-clock, python start -> exit (model load + CLIP/VAE encode + sampling + VAE decode + FFV1)
  inloop_s   sum over windows of the CUDA-synchronised pipeline __call__ (CLIP+VAE encode + sampling loop)
  unet_s     sum of CUDA-event time around every UNet forward
Configs = render label minus the clip id.  Only rc=0 runs whose speed_log.json exists are used.
"""
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
OUT = sys.argv[1]
RUNRE = re.compile(r"^RUN (\S+) (\S+) model=(\S+) guid=(\S+) sigmas=(\S+) pad=(\S+) rc=(\d+) secs=(\S+) md5=(\S*) "
                   r"load1=(\S+) gpus=(\S+) (\S+) dir=(\S+)")
runs = []
for f in sys.argv[2:]:
    for line in open(f):
        m = RUNRE.match(line.strip())
        if not m:
            continue
        clip, label, model, guid, sig, pad, rc, secs, md5, load1, gpus, hhmm, d = m.groups()
        rec = dict(clip=clip, label=label, model=model, guid=float(guid), sig=sig, pad=pad, rc=int(rc),
                   process_s=float(secs), md5=md5, load1=float(load1), gpus=gpus, at=hhmm, dir=d, log=f)
        sp = os.path.join(d, "speed_log.json")
        if os.path.exists(sp):
            j = json.load(open(sp))
            rec.update(inloop_s=j["call_s_sum"], unet_s=j["unet_ms_sum"] / 1000.0, n_windows=j["n_windows"],
                       calls=j["unet_calls_total"], calls_per_window=j["unet_calls_per_window"],
                       batch=j["batch_sizes"], load_s=j["load_s"], hook_total_s=j["hook_total_s"],
                       win_s=[w["call_s"] for w in j["windows"]],
                       # steady state: windows 1..n-2 (drops window 0's warm-up and the shorter last window)
                       steady_win_s=st.median([w["call_s"] for w in j["windows"][1:-1]]),
                       steady_unet_ms=st.median([w["unet_ms"] for w in j["windows"][1:-1]]))
        runs.append(rec)

L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


def cfg_of(r):
    return r["label"]


ok = [r for r in runs if r["rc"] == 0 and "inloop_s" in r]
by = defaultdict(list)
for r in ok:
    by[cfg_of(r)].append(r)

P("=" * 150)
P("WALL-CLOCK per configuration (GPU 0 = RTX 4090, one render at a time; medians over the renders listed)")
P("process_s = python start->exit;  inloop_s = sum of CUDA-synced pipeline __call__ (encode + sampling);  "
  "unet_s = CUDA-event UNet time")
P("steady/win, unet/win = median over windows 1..n-2 of one render (drops window 0's warm-up and the shorter last "
  "window), then median over renders")
P("=" * 150)
P(f"  {'config':28s}{'n':>3s}{'UNet calls/win':>15s}{'batch':>7s}{'elem/win':>9s}{'process_s':>11s}"
  f"{'inloop_s':>10s}{'unet_s':>9s}{'inloop/win':>11s}{'steady/win':>11s}{'unet/win':>9s}{'windows':>8s}  clips")
for cfg in sorted(by, key=lambda c: (-st.median([r["unet_s"] for r in by[c]]), c)):
    rs = by[cfg]
    cpw = sorted({x for r in rs for x in r["calls_per_window"]})
    bs = sorted({x for r in rs for x in r["batch"]})
    elem = [c * b for c in cpw for b in bs] if len(cpw) == 1 and len(bs) == 1 else ["?"]
    P(f"  {cfg:28s}{len(rs):3d}{str(cpw):>15s}{str(bs):>7s}{str(elem[0]):>9s}"
      f"{st.median([r['process_s'] for r in rs]):11.1f}{st.median([r['inloop_s'] for r in rs]):10.1f}"
      f"{st.median([r['unet_s'] for r in rs]):9.1f}"
      f"{st.median([r['inloop_s'] / r['n_windows'] for r in rs]):11.2f}"
      f"{st.median([r['steady_win_s'] for r in rs]):11.2f}{st.median([r['steady_unet_ms'] for r in rs]) / 1000:9.2f}"
      f"{st.median([r['n_windows'] for r in rs]):8.0f}  {' '.join(sorted(r['clip'] for r in rs))}")
P()

# paired same-clip ratios against each model's deployed 8-step @1.01 render in this session
REF = {"deliv": "deliv_g101_s8", "origin": "origin_g101_s8"}
P("=" * 150)
P("PAIRED same-session ratios vs the same model's deployed 8-step @1.01 render of the SAME clip (this session)")
P("=" * 150)
idx = {(r["clip"], r["label"]): r for r in ok}
for r in sorted(ok, key=lambda r: (r["model"], r["label"], r["clip"])):
    mdl = "deliv" if r["label"].startswith("deliv") else ("origin" if r["label"].startswith("origin") else None)
    ref = idx.get((r["clip"], REF.get(mdl, "")))
    if ref is None or ref is r:
        continue
    P(f"  {r['clip']} {r['label']:28s} vs {ref['label']:16s} process {r['process_s']:6.1f}/{ref['process_s']:6.1f}"
      f" = {r['process_s'] / ref['process_s']:.3f}   inloop {r['inloop_s']:6.1f}/{ref['inloop_s']:6.1f}"
      f" = {r['inloop_s'] / ref['inloop_s']:.3f}   unet {r['unet_s']:6.1f}/{ref['unet_s']:6.1f}"
      f" = {r['unet_s'] / ref['unet_s']:.3f}")
P()
P("=" * 150)
P("FIRST-WINDOW WARM-UP and STEADY STATE (deliverable vs origin at the same sampler setting; medians over renders)")
P("window0 = CUDA-synced __call__ seconds of window 0; steady = median of windows 1..n-2; excess = window0 - steady")
P("=" * 150)
pairs = sorted({cfg.split("_", 1)[1] for cfg in by if cfg.startswith(("deliv_", "origin_"))})
for suf in pairs:
    dl, og = by.get("deliv_" + suf), by.get("origin_" + suf)
    if not dl or not og:
        continue
    def med(rs, f):
        return st.median([f(r) for r in rs])
    w0d, w0o = med(dl, lambda r: r["win_s"][0]), med(og, lambda r: r["win_s"][0])
    sd, so = med(dl, lambda r: r["steady_win_s"]), med(og, lambda r: r["steady_win_s"])
    ud, uo = med(dl, lambda r: r["steady_unet_ms"]) / 1000, med(og, lambda r: r["steady_unet_ms"]) / 1000
    tid, tio = med(dl, lambda r: r["inloop_s"]), med(og, lambda r: r["inloop_s"])
    P(f"  {suf:14s} deliv n={len(dl):2d} / origin n={len(og):2d}:  window0 {w0d:5.2f} vs {w0o:5.2f} s (excess "
      f"{w0d - sd:+.2f} vs {w0o - so:+.2f})   steady/win {sd:5.2f} vs {so:5.2f} s ({(sd / so - 1) * 100:+.1f}%)   "
      f"UNet/win {ud:5.2f} vs {uo:5.2f} s ({(ud / uo - 1) * 100:+.1f}%)   whole-clip inloop {tid:6.1f} vs {tio:6.1f} s "
      f"({(tid / tio - 1) * 100:+.1f}%)")
P()
P("=" * 150)
P("EVERY RUN (from the driver log; failures included)")
P("=" * 150)
for r in runs:
    extra = (f"inloop={r['inloop_s']:.1f} unet={r['unet_s']:.1f} calls={r['calls']} per_win={r['calls_per_window']} "
             f"batch={r['batch']} load={r['load_s']:.1f}") if "inloop_s" in r else "(no speed_log.json)"
    P(f"  {r['at']} {r['clip']} {r['label']:28s} rc={r['rc']} process={r['process_s']:.1f} {extra} "
      f"load1={r['load1']} gpus={r['gpus']} md5={r['md5']}")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"\nwrote {OUT}")
