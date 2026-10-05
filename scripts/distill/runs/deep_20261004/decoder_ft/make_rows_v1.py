#!/usr/bin/env python
"""Build a rows json (PUBLISHED_ROWS.json format: {"labels": [...], "cells": {clip: {label: {...}}}}) for
score_registered_scalegt_v1.py from score_clip_ll.py "ROW clip=... tag=... dy= dx= ... lpips= ... path=..." lines.
label = tag with the leading "<clip>_" removed.  Existing rows (e.g. the published origin_ll / s25_ll rows of the 12 test
clips) can be merged with --base <published_rows.json>:<label>,<label> (their values are copied verbatim).
usage: python make_rows_v1.py <out_json> <labels comma list> <score_log> [<score_log> ...] [--base path:lab1,lab2]
Every (clip, label) in the requested label list must be found exactly once (else it exits non-zero).
"""
import json, os, sys
args = [a for a in sys.argv[1:] if not a.startswith("--base")]
base = [a.split("=", 1)[1] if "=" in a else None for a in sys.argv[1:] if a.startswith("--base")]
OUT, LABELS, logs = args[0], args[1].split(","), args[2:]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
cells = {}
for lg in logs:
    for line in open(lg):
        if not line.startswith("ROW "): continue
        kv = dict(t.split("=", 1) for t in line.split()[1:])
        clip, tag = kv["clip"], kv["tag"]
        lab = tag[len(clip) + 1:] if tag.startswith(clip + "_") else tag
        if lab not in LABELS: continue
        c = cells.setdefault(clip, {})
        row = dict(dy=int(kv["dy"]), dx=int(kv["dx"]), leftPSNR=float(kv["leftPSNR"]), lpips=float(kv["lpips"]),
                   sharp=float(kv["sharp"]), gtSharp=float(kv["gtSharp"]), rightPSNR=float(kv["rightPSNR"]), n=int(kv["n"]),
                   path=kv["path"], sources=[lg])
        if lab in c and abs(c[lab]["lpips"] - row["lpips"]) > 1e-6:
            sys.exit(f"conflicting rows for {clip} {lab}: {c[lab]['lpips']} vs {row['lpips']}")
        c[lab] = row
for b in base:
    path, labs = b.split(":")
    pub = json.load(open(path))
    for clip, cc in pub["cells"].items():
        for lab in labs.split(","):
            if lab in cc and lab in LABELS:
                cells.setdefault(clip, {})[lab] = dict(cc[lab], sources=[path])
missing = [(c, l) for c in cells for l in LABELS if l not in cells[c]]
if missing:
    sys.exit(f"missing rows: {missing}")
json.dump(dict(labels=LABELS, cells=cells, clips=sorted(cells)), open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(cells)} clips x {len(LABELS)} labels")
