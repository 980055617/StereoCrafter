#!/usr/bin/env python
"""Monitoring only (never used for selection): UNREG dev LPIPS / sharpness per rendered checkpoint vs origin_s8, from the
score_clip_ll.py ROW lines in score_dev_*.txt.  usage: python monitor_unreg_v1.py"""
import glob, os, re
L = os.path.dirname(os.path.abspath(__file__))
def rows(p):
    out = {}
    for line in open(p):
        if line.startswith("ROW "):
            kv = dict(t.split("=", 1) for t in line.split()[1:]); out[kv["clip"]] = (float(kv["lpips"]), float(kv["sharp"]))
    return out
o = rows(os.path.join(L, "score_dev_origin_s8.txt"))
print(f"{'run/step':22s} {'n':>2s} {'UNREG mean':>10s} {'d vs origin':>11s} {'impr':>5s} {'sharp ratio':>11s}   per-clip d")
for p in sorted(glob.glob(os.path.join(L, "score_dev_*_s*.txt")), key=lambda x: (x.split("score_dev_")[1].rsplit("_s", 1)[0], int(re.search(r"_s(\d+)\.txt$", x).group(1)) if re.search(r"_s(\d+)\.txt$", x) else 0)):
    if p.endswith("origin_s8.txt"): continue
    r = rows(p); cl = [c for c in r if c in o]
    if not cl: continue
    d = [r[c][0] - o[c][0] for c in cl]
    sr = sum(r[c][1] for c in cl) / sum(o[c][1] for c in cl)
    print(f"{os.path.basename(p)[10:-4]:22s} {len(cl):2d} {sum(r[c][0] for c in cl)/len(cl):10.4f} {sum(d)/len(d):+11.4f} {sum(x<0 for x in d):2d}/{len(d)} {sr:11.3f}   " + " ".join(f"{c}:{x:+.4f}" for c, x in zip(cl, d)))
