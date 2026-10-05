#!/usr/bin/env python
"""Build the per-clip rows file for score_registered_extm_r2.py (PREREG_r2.txt).

usage: python build_rows_r2.py <clip> <label> <unreg_rows_txt> <out_json>
  - origin_ll and mstudent2_step800_deliv_ll cells are copied from more_20261004/eval_robustness/PUBLISHED_ROWS.json
  - the external model's cell comes from the LAST 'ROW clip=<clip> tag=<clip>_<label>' line of <unreg_rows_txt>
    (output of fulldata_v2/beyond4/score_clip_ll.py, run verbatim)
  - gate: origin_ll re-scored in the same score_clip_ll.py call must reproduce its published lpips (|d| < 1e-5)
"""
import json
import os
import sys

REPO = "/home/kawa/master_project/StereoCrafter"
PUB = f"{REPO}/scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json"
clip, label, txt, out = sys.argv[1:5]
assert not os.path.exists(out), f"refusing to overwrite {out}"
pub = json.load(open(PUB))
rows = {}
for line in open(txt):
    if not line.startswith("ROW "):
        continue
    kv = dict(t.split("=", 1) for t in line.split()[1:])
    if kv["clip"] == clip:
        rows[kv["tag"]] = kv
o_tag, m_tag = f"{clip}_origin_ll", f"{clip}_{label}"
assert o_tag in rows and m_tag in rows, (sorted(rows), o_tag, m_tag)
po = pub["cells"][clip]["origin_ll"]
d = float(rows[o_tag]["lpips"]) - po["lpips"]
assert abs(d) < 1e-5, f"origin_ll re-score {rows[o_tag]['lpips']} != published {po['lpips']} (d {d:+.2e})"
m = rows[m_tag]
cell = dict(dy=int(m["dy"]), dx=int(m["dx"]), leftPSNR=float(m["leftPSNR"]), lpips=float(m["lpips"]),
            sharp=float(m["sharp"]), gtSharp=float(m["gtSharp"]), rightPSNR=float(m["rightPSNR"]), n=int(m["n"]),
            path=m["path"], sources=[txt])
labels = ["origin_ll", "mstudent2_step800_deliv_ll", label]
res = dict(clips=[clip], labels=labels,
           cells={clip: {"origin_ll": po, "mstudent2_step800_deliv_ll": pub["cells"][clip]["mstudent2_step800_deliv_ll"],
                         label: cell}},
           origin_rescore_delta=d, built_from=[PUB, txt])
json.dump(res, open(out, "w"), indent=1)
print(f"ROWS {clip} {label}: dy {cell['dy']} dx {cell['dx']} (origin {po['dy']},{po['dx']}) lpips {cell['lpips']:.6f} "
      f"(origin pub {po['lpips']:.6f}, rescore d {d:+.1e}) -> {out}")
