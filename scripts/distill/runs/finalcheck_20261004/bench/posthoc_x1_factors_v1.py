#!/usr/bin/env python
"""finalcheck_20261004 / bench lane -- POST-HOC, NON-GATING information about the X1 failure.  CPU only.
Written after X1 (PREREG.txt) failed on the BS=1 rows; it cannot change any verdict.

usage: posthoc_x1_factors_v1.py FP16_RUN_DIR [TAG]
  (1) FACTOR SPLIT at 576x1024: which factor of rel = (steps/8) x guidance x Mamba fails to transfer?
      guidance factor = t(model, BS1)/t(model, BS2);  Mamba factor = t(mamba5, BS)/t(origin, BS)
      deployed = per-forward steady-state UNet seconds of the speed lane (speed_log.json, CUDA events, bf16)
      bench    = the primary fp16 bench2.py run
  (2) MACHINE-LOAD CHECK: steady UNet s/window of speed-lane renders whose pre-render driver snapshot showed GPU 1 idle
      vs busy (same UNet work), e.g. deliv_g100_T5nat (all GPU1-idle) vs deliv_g100_T5pad (mostly GPU1-busy).
writes scripts/distill/runs/finalcheck_20261004/bench/POSTHOC_X1_FACTORS_<TAG>.txt
"""
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
LANE = "scripts/distill/runs/finalcheck_20261004/bench"
SPEED_LOG = "outputs/finalcheck_20261004/speed/timing_gpu0.txt"
RUN = sys.argv[1].rstrip("/")
TAG = sys.argv[2] if len(sys.argv) > 2 else "v1"
OUT = []


def P(s=""):
    OUT.append(s)
    print(s, flush=True)


# primary fp16 bench (576x1024 only needed)
F = defaultdict(list)
for line in open(f"{RUN}/bench_results.txt"):
    if line.startswith("RESULT "):
        j = json.loads(line[7:])
        m = re.match(r"^(origin|mamba5)_bs(\d)_h576w1024_r\d+$", j["label"])
        if m:
            F[(m.group(1), int(m.group(2)))].append(j["sec"])
fb = {k: st.mean(v) for k, v in F.items()}

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
    g1 = re.search(r"1,(\d+)%", m.group(11))
    cpw = j["unet_calls_per_window"][0]
    su = st.median([w["unet_ms"] for w in j["windows"][1:-1]]) / 1000
    SR[m.group(2)].append(dict(clip=m.group(1), steady_unet=su, per_fwd=su / cpw, busy=int(g1.group(1)) > 0 if g1 else None,
                               load1=float(m.group(10))))


def dep(label):
    return st.median([r["per_fwd"] for r in SR[label]])


P("=" * 140)
P("POST-HOC (non-gating): what fails to transfer in X1, and is it machine load?   576x1024")
P("deployed = speed lane steady-state UNet seconds per forward (bf16 pipeline, CUDA events); bench = primary fp16 bench2.py run")
P("=" * 140)
P("(1) FACTOR SPLIT")
d_o2, d_o1 = dep("origin_g101_s8"), dep("origin_g100_s8")
d_m2, d_m1 = dep("deliv_g101_s8"), dep("deliv_g100_s8")
b_o2, b_o1, b_m2, b_m1 = fb[("origin", 2)], fb[("origin", 1)], fb[("mamba5", 2)], fb[("mamba5", 1)]
P(f"  per forward (s)          deployed   bench    dep/bench")
for name, d, b in (("origin BS2", d_o2, b_o2), ("origin BS1", d_o1, b_o1), ("deliverable/mamba5 BS2", d_m2, b_m2),
                   ("deliverable/mamba5 BS1", d_m1, b_m1)):
    P(f"  {name:24s} {d:8.4f} {b:8.4f}   {d / b:8.3f}")
P(f"  guidance factor t(BS1)/t(BS2): origin  deployed {d_o1 / d_o2:.4f}  bench {b_o1 / b_o2:.4f}  bench/deployed {(b_o1 / b_o2) / (d_o1 / d_o2):.3f}")
P(f"                                 mamba5  deployed {d_m1 / d_m2:.4f}  bench {b_m1 / b_m2:.4f}  bench/deployed {(b_m1 / b_m2) / (d_m1 / d_m2):.3f}")
P(f"  Mamba factor t(mamba5)/t(origin): BS2 deployed {d_m2 / d_o2:.4f}  bench {b_m2 / b_o2:.4f}  bench/deployed {(b_m2 / b_o2) / (d_m2 / d_o2):.3f}")
P(f"                                    BS1 deployed {d_m1 / d_o1:.4f}  bench {b_m1 / b_o1:.4f}  bench/deployed {(b_m1 / b_o1) / (d_m1 / d_o1):.3f}")
P("  step factor: exact by construction (per-forward time does not depend on the step count; deployed per-forward of the")
P("  T6/T5 renders: " + ", ".join(f"{lab} {dep(lab):.4f}" for lab in ("deliv_g101_T6pad", "deliv_g101_T5pad", "deliv_g100_T5pad",
                                                                    "origin_g101_T6pad", "origin_g101_T5pad", "origin_g100_T5pad")) + ")")
P()
P("(2) MACHINE-LOAD CHECK (GPU 1 state in the speed driver's pre-render snapshot; same UNet work within a label)")
P(f"  {'label':20s} {'n idle':>6s} {'idle med s/win':>14s} {'n busy':>6s} {'busy med s/win':>14s}")
for lab in ("deliv_g100_T5nat", "deliv_g100_T5pad", "deliv_g100_s8", "origin_g100_s8", "origin_g100_T5pad",
            "deliv_g101_T5pad", "origin_g101_T5pad", "deliv_g101_T6pad", "origin_g101_T6pad"):
    idle = [r["steady_unet"] for r in SR[lab] if r["busy"] is False]
    busy = [r["steady_unet"] for r in SR[lab] if r["busy"] is True]
    P(f"  {lab:20s} {len(idle):6d} {st.median(idle) if idle else float('nan'):14.4f} {len(busy):6d} "
      f"{st.median(busy) if busy else float('nan'):14.4f}")
P("  deliv_g100_T5nat and deliv_g100_T5pad do identical UNet work (5 calls x batch 1 per window); the first ran with GPU 1 idle,")
P("  the second mostly with GPU 1 busy.")
open(f"{LANE}/POSTHOC_X1_FACTORS_{TAG}.txt", "w").write("\n".join(OUT) + "\n")
print(f"\nwrote {LANE}/POSTHOC_X1_FACTORS_{TAG}.txt")
