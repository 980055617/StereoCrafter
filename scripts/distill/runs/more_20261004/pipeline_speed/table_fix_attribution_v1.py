#!/usr/bin/env python
"""Per-fix attribution from paired renders: exclusive-stage deltas (fix render minus baseline render), mapped to the fix
that acts on each stage.  usage: table_fix_attribution_v1.py <out_txt (new)> <label> <base_dir>[,<base_dir2>] <fix_dir>[,<fix_dir2>]
(several dirs per side -> mean of the per-render exclusive stage times)."""
import io, contextlib, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_stages_v2 as A

STAGE_FIX = [  # (fix, stages it acts on)
    ("F5 pre-filled Triton autotune choices (first UNet call)", ["__w0_excess__"]),
    ("F6 skip decoding discarded overlap chunks", ["vae_decode"]),
    ("F7 direct GPU post-processing (no PIL round trip)", ["tensor2vid_pil", "tensor2vid_DIRECT"]),
    ("F8 skip the unused CPU noise_aug randn", ["noise_aug_randn_cpu", "noise_aug_SKIPPED"]),
    ("F9 crop-first reader", ["read_video", "center_crop", "pre_window_misc"]),
]


def mean_ex(dirs):
    exs, ws, refs = [], [], []
    for d in dirs:
        with contextlib.redirect_stdout(io.StringIO()):
            r = A.main([d])[d]["runs"][0]
        exs.append(r["exclusive"]); ws.append(r["windows"]); refs.append(r["ref"])
    names = {n for e in exs for n in e}
    m = {n: statistics.mean(e.get(n, 0.0) for e in exs) for n in names}
    # first UNet forward minus the second one of the same window (robust also for 2-window diagnosis runs)
    m["__w0_excess__"] = statistics.mean((w["first_call_ms"] - w["second_call_ms"]) / 1000.0 for w in ws)
    return m, statistics.mean(refs), refs


out, label, bdirs, fdirs = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
assert not os.path.exists(out)
B, bref, brefs = mean_ex(bdirs)
F, fref, frefs = mean_ex(fdirs)
lines = [f"=== {label}", f"  baseline renders {bdirs} end-to-end {['%.2f' % x for x in brefs]} mean {bref:.2f} s",
         f"  fix renders      {fdirs} end-to-end {['%.2f' % x for x in frefs]} mean {fref:.2f} s",
         f"  total change {fref - bref:+.2f} s ({100 * (fref - bref) / bref:+.1f} %)", "",
         f"  {'fix':58s} {'base s':>8s} {'fix s':>8s} {'delta s':>8s}"]
acc = 0.0
for fix, stages in STAGE_FIX:
    b = sum(B.get(s, 0.0) for s in stages)
    f = sum(F.get(s, 0.0) for s in stages)
    acc += f - b
    lines.append(f"  {fix:58s} {b:8.2f} {f:8.2f} {f - b:+8.2f}")
other = (fref - bref) - acc
lines.append(f"  {'everything else (run-to-run variation, CPU contention)':58s} {'':8s} {'':8s} {other:+8.2f}")
lines.append("")
lines.append("  all exclusive stages (base -> fix):")
for n in sorted(set(B) | set(F), key=lambda n: -max(B.get(n, 0), F.get(n, 0))):
    if n.startswith("__"):
        continue
    lines.append(f"    {n:30s} {B.get(n, 0):8.2f} {F.get(n, 0):8.2f} {F.get(n, 0) - B.get(n, 0):+8.2f}")
txt = "\n".join(lines)
open(out, "w").write(txt + "\n")
print(txt)
