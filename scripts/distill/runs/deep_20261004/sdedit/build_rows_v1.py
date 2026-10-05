#!/usr/bin/env python
"""Build the ROWS json that score_registered_v1.py (unchanged) reads, from score_clip_ll.py ROW lines.
usage: build_rows_v1.py SCORES.txt OUT_ROWS.json
labels = every label present on EVERY clip of the file, origin_ll first (it fixes the published window inside the
scorer), mstudent2_step800_deliv_ll second (the scorer's inspection panel reads both), the rest in file order.
cells[clip][label] = {dy, dx, leftPSNR, lpips, sharp, gtSharp, rightPSNR, n, path} -- "lpips" is the UNREG value the
registered scorer must reproduce (its G0-style check line).
"""
import json
import os
import sys
from collections import OrderedDict

SC, OUT = sys.argv[1], sys.argv[2]
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}")
cells = OrderedDict()
order = []
for line in open(SC):
    if not line.startswith("ROW "):
        continue
    kv = dict(tok.split("=", 1) for tok in line.split()[1:])
    clip, tag = kv["clip"], kv["tag"]
    assert tag.startswith(clip + "_"), (clip, tag)
    lab = tag[len(clip) + 1:]
    rec = dict(dy=int(kv["dy"]), dx=int(kv["dx"]), leftPSNR=float(kv["leftPSNR"]), lpips=float(kv["lpips"]),
               sharp=float(kv["sharp"]), gtSharp=float(kv["gtSharp"]), rightPSNR=float(kv["rightPSNR"]),
               n=int(kv["n"]), path=kv["path"])
    c = cells.setdefault(clip, OrderedDict())
    if lab in c:
        assert c[lab] == rec, f"duplicate ROW with different values: {clip} {lab}"
    c[lab] = rec
    if lab not in order:
        order.append(lab)
common = [lab for lab in order if all(lab in cells[c] for c in cells)]
head = [lab for lab in ("origin_ll", "mstudent2_step800_deliv_ll") if lab in common]
assert head == ["origin_ll", "mstudent2_step800_deliv_ll"], f"both deployed rows are required, got {head}"
labels = head + [lab for lab in common if lab not in head]
dropped = sorted(set(order) - set(common))
json.dump(dict(clips=list(cells), labels=labels, cells=cells, dropped_not_on_every_clip=dropped,
               source=os.path.abspath(SC)), open(OUT, "w"), indent=1)
print(f"rows: clips {list(cells)} labels {labels} dropped {dropped}")
