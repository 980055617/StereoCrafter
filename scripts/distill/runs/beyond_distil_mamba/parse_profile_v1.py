#!/usr/bin/env python
"""Summarise utils/module_timing.py attn1 profiles: total attn1 ms, and the 5 replaced slots' share.

usage: parse_profile_v1.py <out.txt> <profile root>
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
    by = {m["name"]: m for m in j["modules"]}
    tot = sum(m["totalMs"] for m in j["modules"])
    calls = max((m["calls"] for m in j["modules"]), default=0)
    rows[(res, cfg)] = dict(total=tot, calls=calls, path=p,
                            down0=sum(by[s]["totalMs"] for s in DOWN0 if s in by),
                            up3=sum(by[s]["totalMs"] for s in UP3 if s in by),
                            cls={s: by[s]["class"] for s in SLOTS if s in by})

L = []


def P(s=""):
    L.append(s); print(s, flush=True)


P("=" * 104)
P("attn1 UNet-MODULE TIME (utils/module_timing.py, include '*.attn1', clip 0301, 3 chunks of 14 frames,")
P("8 steps, guidance 1.01).  This is the instrument that produced the published -5.3/-20.5/-21.7 %.")
P("=" * 104)
P(f"  {'res':10s} {'config':8s} {'all attn1 ms':>13s} {'vs origin':>10s} {'down0 ms':>9s} "
  f"{'up3 ms':>9s} {'calls':>6s}")
for res in sorted({r for r, _ in rows}):
    base = rows.get((res, "none"), {}).get("total")
    for cfg in ("none", "down0", "all5"):
        r = rows.get((res, cfg))
        if not r:
            continue
        rel = "" if not base else f"{100.0*(r['total']-base)/base:+9.1f}%"
        P(f"  {res:10s} {cfg:8s} {r['total']:13.1f} {rel:>10s} {r['down0']:9.1f} {r['up3']:9.1f} "
          f"{r['calls']:6d}")
    b = rows.get((res, "none")); d = rows.get((res, "down0")); a = rows.get((res, "all5"))
    if b and d and a:
        P(f"    -> the three up_blocks.3 slots contribute {d['total']-a['total']:+.1f} ms of attn1 time "
          f"({100.0*(d['total']-a['total'])/b['total']:+.2f} % of origin's attn1 total):")
        P(f"       option B (2-slot) attn1 {d['total']:.1f} ms vs option A (5-slot) {a['total']:.1f} ms "
          f"vs origin {b['total']:.1f} ms")
P()
for k, v in sorted(rows.items()):
    P(f"  {k}  {v['path']}")
open(out_path, "w").write("\n".join(L) + "\n")
