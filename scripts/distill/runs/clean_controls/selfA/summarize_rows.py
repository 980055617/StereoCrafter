"""Generic readout of score_clip_ll.py ROW lines from one or more score files: every row's LPIPS / sharp and its delta against
the clip's `<clip>_origin_nohook_ll` reference row (this lane's lossless origin render).  usage: python summarize_rows.py <scores.txt> ..."""
import sys, re
rows = []
for f in sys.argv[1:]:
    for line in open(f):
        if line.startswith("ROW "):
            kv = dict(re.findall(r"(\w+)=(\S+)", line)); kv["file"] = f.split("/")[-1]; rows.append(kv)
ref = {r["clip"]: r for r in rows if r["tag"] == f"{r['clip']}_origin_nohook_ll"}
seen = set()
print(f"{'clip':5s} {'tag':34s} {'LPIPS':>7s} {'dLPIPS':>8s} {'sharp':>7s} {'dsharp%':>8s} {'rPSNR':>7s} {'offset':>9s}  file")
for r in rows:
    key = (r["clip"], r["tag"])
    if key in seen: continue
    seen.add(key); R = ref.get(r["clip"])
    L, S = float(r["lpips"]), float(r["sharp"])
    dL = L - float(R["lpips"]) if R else float("nan"); dS = (S / float(R["sharp"]) - 1) * 100 if R else float("nan")
    print(f"{r['clip']:5s} {r['tag'][:34]:34s} {L:7.4f} {dL:+8.4f} {S:7.4f} {dS:+8.2f} {float(r['rightPSNR']):7.3f} {f'({r[chr(100)+chr(121)]},{r[chr(100)+chr(120)]})':>9s}  {r['file']}")
