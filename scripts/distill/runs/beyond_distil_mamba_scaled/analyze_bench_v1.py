#!/usr/bin/env python
"""Turn bench_deliv_v1.txt's RESULT lines into the speed/VRAM half of the headline table.

Same reduction as scripts/distill/runs/slotbudget/analyze_v1.py pass A: mean +- half-range over the
per-repeat processes, delta% against the origin row at the same resolution, peak VRAM and its delta.
Also prints the per-forward execution counts, which are the guard against the failure mode that once
hid in this project: a gate=0 build running BOTH the Mamba block and the reference attention.  For a
correct 5-slot build they must read mamba_core=5, plain_attn1=11, origin_attn=0, gates=[1.0].

usage: analyze_bench_v1.py <bench_out.txt> [<published_ref_note>]
"""
import json
import re
import sys
from collections import defaultdict

RES = ["h576w1024", "h1024w1792", "h1024w1920"]
CFG = ["origin", "mamba5ck", "mamba5deliv"]
PRETTY = {"origin": "origin (no replacement)",
          "mamba5ck": "mamba 5-slot SHIPPED (ckpt loaded)",
          "mamba5deliv": "THIS DELIVERABLE (ckpt loaded)"}
KEY = re.compile(r"^(origin|mamba5ck|mamba5deliv)_(h\d+w\d+)_r(\d+)$")

A = defaultdict(list)
for line in open(sys.argv[1]):
    if not line.startswith("RESULT "):
        continue
    r = json.loads(line[7:])
    m = KEY.match(r["label"])
    if m:
        A[(m.group(1), m.group(2))].append(r)


def half(v):
    return (max(v) - min(v)) / 2 if len(v) > 1 else 0.0


L = []


def P(s=""):
    L.append(s)
    print(s, flush=True)


P("=" * 122)
P("SPEED + PEAK VRAM -- scripts/distill/bench2.py UNMODIFIED, the provenance of the published")
P("-5.3 / -20.5 / -21.7 %.  batch 2 (guidance 1.01 keeps the CFG-doubled batch), 14 frames, fp16,")
P("3 untimed warmup forwards then 10 timed forwards / 10 with cuda.synchronize bracketing, peak reset")
P("after warmup, ONE PROCESS PER REPEAT, reps-outer/configs-inner, exclusive GPU.  Resolutions are HxW.")
P("Unlike the published bench, the two Mamba rows LOAD REAL WEIGHTS (CKPT=...), so the deliverable's")
P("speed row is a primary measurement of the shipped file and not an inherited claim.")
P("=" * 122)
P(f"  {'res(HxW)':12s}{'config':36s}{'n':>3s}{'sec/fwd':>10s}{'+-half':>9s}{'delta%':>9s}"
  f"{'peakMiB':>9s}{'dVRAM%':>8s}  checks")
for res in RES:
    o = A.get(("origin", res))
    if not o:
        continue
    ot = sum(x["sec"] for x in o) / len(o)
    ov = sum(x["peak_MiB"] for x in o) / len(o)
    for cfg in CFG:
        rs = A.get((cfg, res))
        if not rs:
            continue
        t = [x["sec"] for x in rs]
        v = [x["peak_MiB"] for x in rs]
        mt, mv = sum(t) / len(t), sum(v) / len(v)
        P(f"  {res:12s}{PRETTY[cfg]:36s}{len(rs):>3d}{mt:10.4f}{half(t):9.4f}"
          f"{100.0 * (mt - ot) / ot:+9.2f}{mv:9.0f}{100.0 * (mv - ov) / ov:+8.2f}  "
          f"calls={rs[0]['per_fwd_calls']} gates={rs[0]['gates']}")
    P()
P("=" * 122)
P("DELIVERABLE vs SHIPPED 5-slot Mamba -- they are the SAME ARCHITECTURE (same module classes, same")
P("shapes, only up_blocks.3 weight VALUES differ), so any gap here is measurement spread, not cost.")
P("=" * 122)
P(f"  {'res(HxW)':12s}{'shipped':>10s}{'deliverable':>13s}{'diff':>10s}{'diff%':>8s}{'spread(ship)':>14s}"
  f"{'spread(deliv)':>15s}{'vs origin%':>12s}")
for res in RES:
    a, b, o = A.get(("mamba5ck", res)), A.get(("mamba5deliv", res)), A.get(("origin", res))
    if not (a and b and o):
        continue
    ta = sum(x["sec"] for x in a) / len(a)
    tb = sum(x["sec"] for x in b) / len(b)
    to = sum(x["sec"] for x in o) / len(o)
    P(f"  {res:12s}{ta:10.4f}{tb:13.4f}{tb - ta:+10.4f}{100.0 * (tb - ta) / ta:+8.2f}"
      f"{half([x['sec'] for x in a]):14.4f}{half([x['sec'] for x in b]):15.4f}"
      f"{100.0 * (tb - to) / to:+12.2f}")
P()
if len(sys.argv) > 2:
    P(sys.argv[2])
out = sys.argv[1].rsplit(".", 1)[0] + "_TABLE.txt"
open(out, "w").write("\n".join(L) + "\n")
print(f"\nwrote {out}")
