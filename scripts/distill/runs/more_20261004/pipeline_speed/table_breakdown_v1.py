#!/usr/bin/env python
"""Side-by-side end-to-end breakdown table for several renders (uses analyze_stages_v2's attribution).
usage: table_breakdown_v1.py <out_txt (new)> LABEL=<clip_dir> [LABEL=<clip_dir> ...]
For a SK_REPEAT>1 render, LABEL may end in '#1' to select rep 1 (per-clip time inside a persistent process)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_stages_v2 as A

out = sys.argv[1]
assert not os.path.exists(out), f"refusing to overwrite {out}"
cols = []
for spec in sys.argv[2:]:
    label, d = spec.split("=", 1)
    rep = 0
    if "#" in label:
        label, r = label.split("#")
        rep = int(r)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = A.main([d])[d.rstrip("/")]
    run = [r for r in res["runs"] if r["rep"] == rep][0]
    cols.append((label, d, run, res))

lines = []
w = 13
hdr = f"{'stage group':34s}" + "".join(f"{c[0][:w]:>{w}s}" for c in cols)
lines.append(hdr)
lines.append("-" * len(hdr))
for g in A.ORDER:
    row = f"{g:34s}"
    for label, d, run, res in cols:
        v = run["groups"].get(g)
        row += f"{(f'{v:.2f}' if v is not None else '-'):>{w}s}"
    lines.append(row)
others = sorted({g for c in cols for g in c[2]["groups"] if g not in A.ORDER})
for g in others:
    lines.append(f"{g:34s}" + "".join(f"{c[2]['groups'].get(g, 0):>{w}.2f}" for c in cols))
lines.append("-" * len(hdr))
lines.append(f"{'END-TO-END (reference seconds)':34s}" + "".join(f"{c[2]['ref']:>{w}.2f}" for c in cols))
lines.append(f"{'  of which UNet GPU (CUDA events)':34s}" + "".join(f"{c[2]['exclusive'].get('unet_gpu', 0):>{w}.2f}" for c in cols))
lines.append(f"{'  of which window-0 excess':34s}" + "".join(f"{(c[2]['windows']['w0_excess'] or 0):>{w}.2f}" for c in cols))
lines.append(f"{'  non-UNet share of end-to-end':34s}" + "".join(
    f"{100 * (1 - c[2]['exclusive'].get('unet_gpu', 0) / c[2]['ref']):>{w - 1}.1f}%" for c in cols))
lines.append(f"{'unattributed (s)':34s}" + "".join(f"{c[2]['unattributed']:>{w}.2f}" for c in cols))
lines.append(f"{'md5 (pre-encode sbs)':34s}" + "".join(f"{(c[2]['md5'] or '-')[:8]:>{w}s}" for c in cols))
lines.append("")
for label, d, run, res in cols:
    lines.append(f"{label}: {d}  rep={run['rep']}  process_s={res['process_s']}  windows={run['windows']['n']}")
lines.append("")
lines.append("Exclusive stage seconds (all named stages):")
names = sorted({n for c in cols for n in c[2]["exclusive"]}, key=lambda n: -max(c[2]["exclusive"].get(n, 0) for c in cols))
for n in names:
    lines.append(f"  {n:30s}" + "".join(f"{c[2]['exclusive'].get(n, 0):>{w}.2f}" for c in cols))
txt = "\n".join(lines)
open(out, "w").write(txt + "\n")
print(txt)
