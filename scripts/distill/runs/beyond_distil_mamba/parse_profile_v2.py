#!/usr/bin/env python
"""Summarise utils/module_timing.py attn1 profiles, EXCLUDING the one-time Triton JIT warmup.

WHY v2.  The raw totals are warmup-dominated and give the wrong sign.  The FIRST call into a Mamba2
block JIT-compiles its Triton kernels: measured 4980 ms on the first call at 576x1024 versus 10.8 ms
on every call after it, while the origin attention is a flat ~21 ms from call 1.  Summing raw totals
therefore reports Mamba as 2x SLOWER, which contradicts the published -5.3 / -20.5 / -21.7 %.  Dropping
the first call per module -- a once-per-process, once-per-shape cost that a deployed run amortises over
hundreds of calls -- recovers the steady state.  BOTH numbers are printed so the warmup is visible
rather than quietly removed.

usage: parse_profile_v2.py <out.txt> <profile root>
"""
import glob, json, os, sys

out_path, root = sys.argv[1], sys.argv[2]
SLOTS = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1",
         "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
         "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
         "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
         "up_blocks.3.attentions.2.transformer_blocks.0.attn1"]
DOWN0, UP3 = SLOTS[:2], SLOTS[2:]

rows = {}
for p in sorted(glob.glob(os.path.join(root, "*", "module_profile_*.json"))):
    j = json.load(open(p))
    lab = os.path.basename(p)[len("module_profile_"):-len(".json")]
    res, cfg = lab.rsplit("_", 1)
    ss, raw, warm = {}, {}, 0.0
    for m in j["modules"]:
        t = m["timingsMs"]
        raw[m["name"]] = sum(t)
        ss[m["name"]] = sum(t[1:])                      # steady state: drop the JIT warmup call
        warm = max(warm, t[0] if t else 0.0)
    calls = max((m["calls"] for m in j["modules"]), default=0)
    rows[(res, cfg)] = dict(raw=sum(raw.values()), ss=sum(ss.values()), warm=warm, calls=calls, path=p,
                            ss_down0=sum(ss.get(s, 0.0) for s in DOWN0),
                            ss_up3=sum(ss.get(s, 0.0) for s in UP3),
                            cls={s: m["class"] for m in j["modules"] for s in SLOTS if m["name"] == s})

L = []


def P(s=""):
    L.append(s); print(s, flush=True)


P("=" * 110)
P("attn1 UNet-MODULE TIME -- utils/module_timing.py, include '*.attn1' (32 modules: 16 spatial +")
P("16 temporal; only the 5 SPATIAL level-0 slots are ever replaced), clip 0301, 3 chunks of 14 frames,")
P("8 steps, guidance 1.01, 24 calls per module.  Same instrument as the published -5.3/-20.5/-21.7 %.")
P("STEADY STATE = the first call of every module dropped, because the first call into a Mamba2 block")
P("JIT-compiles its Triton kernels (measured below); the raw column keeps that cost visible.")
P("=" * 110)
P(f"  {'res':10s} {'config':7s} {'steady ms':>10s} {'vs origin':>10s} {'raw ms':>9s} {'max 1st call':>13s} "
  f"{'ss down0':>9s} {'ss up3':>9s}")
for res in sorted({r for r, _ in rows}, key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1])):
    base = rows.get((res, "none"), {}).get("ss")
    for cfg in ("none", "down0", "all5"):
        r = rows.get((res, cfg))
        if not r:
            continue
        rel = "" if not base else f"{100.0*(r['ss']-base)/base:+9.1f}%"
        P(f"  {res:10s} {cfg:7s} {r['ss']:10.1f} {rel:>10s} {r['raw']:9.1f} {r['warm']:13.1f} "
          f"{r['ss_down0']:9.1f} {r['ss_up3']:9.1f}")
    b = rows.get((res, "none")); d = rows.get((res, "down0")); a = rows.get((res, "all5"))
    if b and d and a:
        gain_a = 100.0 * (a["ss"] - b["ss"]) / b["ss"]
        gain_b = 100.0 * (d["ss"] - b["ss"]) / b["ss"]
        P(f"    -> option A (5-slot) {gain_a:+.2f} % of origin's attn1 time; option B (2-slot) "
          f"{gain_b:+.2f} %.")
        P(f"       what B GIVES UP = the three up_blocks.3 slots: {d['ss']-a['ss']:+.1f} ms "
          f"= {100.0*(d['ss']-a['ss'])/b['ss']:+.2f} % of origin's attn1 total, i.e. "
          f"{100.0*(d['ss']-a['ss'])/max(abs(a['ss']-b['ss']),1e-9):.0f} % of option A's whole attn1 saving.")
P()
for k, v in sorted(rows.items()):
    P(f"  {k}  {v['path']}")
open(out_path, "w").write("\n".join(L) + "\n")
