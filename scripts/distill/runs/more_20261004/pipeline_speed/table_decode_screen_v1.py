#!/usr/bin/env python
"""Tabulate decode-screen JSONs (decode_bench_v1/v2).  usage: table_decode_screen_v1.py <out_txt (new)> <screen_dir> [...]"""
import glob, json, os, sys
out = sys.argv[1]
assert not os.path.exists(out)
lines = []
for d in sys.argv[2:]:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        R = json.load(open(f))
        if not R.get("passes"):
            rows.append((os.path.basename(f), None, None, None, R.get("error"), None)); continue
        last, p0 = R["passes"][-1], R["passes"][0]
        rows.append((os.path.basename(f)[:-5], last["decode_s_sum"], p0["decode_s_sum"], last["md5_right_u8"], R.get("error"),
                     R.get("peak_mem_gb")))
    base = [r for r in rows if r[0].startswith("c2_cl0_b0_s0_bf16") and r[0].count("_") in (4, 6, 7) and "c3d1" not in r[0]
            and "compdefault" not in r[0] and "rep2" not in r[0]]
    b = base[0] if base else None
    lines.append(f"=== {d}  (windows {json.load(open(glob.glob(os.path.join(d, '*.json'))[0]))['windows']}; decode seconds summed over "
                 f"those windows, last pass; pass0 includes warm-up/benchmarking/compilation)")
    lines.append(f"  {'variant':36s} {'decode_s':>9s} {'vs base':>8s} {'pass0_s':>8s} {'peak GB':>8s}  {'md5 (right-eye u8)':>34s} identity")
    for name, s, s0, md5, err, mem in rows:
        if s is None:
            lines.append(f"  {name:36s} FAILED {err}"); continue
        rel = s / b[1] if b else float('nan')
        same = "IDENTICAL" if b and md5 == b[3] else "differs"
        lines.append(f"  {name:36s} {s:9.3f} {rel:8.3f} {s0:8.3f} {mem:8.2f}  {md5:>34s} {same}")
    lines.append("")
txt = "\n".join(lines)
open(out, "w").write(txt + "\n")
print(txt)
