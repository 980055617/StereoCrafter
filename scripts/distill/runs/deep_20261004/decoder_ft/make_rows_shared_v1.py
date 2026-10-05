#!/usr/bin/env python
"""deep_20261004 / decoder_ft lane -- DEV rows json without running score_clip_ll on every row.

Every dev row of a clip (stock, each decoder checkpoint, each unsharp variant, both models) carries the SAME left half
(copied from the capture render), so the left-eye alignment (dy, dx) of score_clip_ll is the same for all of them.
score_clip_ll is therefore run only on the two stock rows per clip; every other label gets the stock row's dy/dx/leftPSNR
and lpips = NaN (that field is only printed by the registered scorer, never used).  The registered scorer
(score_registered_df_v1.py) recomputes UNREG for every label with score_clip_ll's verbatim math (gate G0 of eval_robustness
and scale_gt: |d| <= 5e-7) and runs its own left-eye search per label; its UNREG is the dev UNREG of record.
The analysis asserts afterwards that every label of a clip has the same md5 of the left half and the same (dy, dx).
usage: python make_rows_shared_v1.py <out_json> <root> <labels comma list> <score_log> [<score_log> ...]
"""
import json
import os
import sys

OUT, ROOT, LABELS, logs = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
stock = {}
for lg in logs:
    for line in open(lg):
        if not line.startswith("ROW "):
            continue
        kv = dict(t.split("=", 1) for t in line.split()[1:])
        clip, tag = kv["clip"], kv["tag"]
        lab = tag[len(clip) + 1:]
        stock.setdefault(clip, {})[lab] = dict(dy=int(kv["dy"]), dx=int(kv["dx"]), leftPSNR=float(kv["leftPSNR"]),
                                               lpips=float(kv["lpips"]), sharp=float(kv["sharp"]),
                                               gtSharp=float(kv["gtSharp"]), rightPSNR=float(kv["rightPSNR"]),
                                               n=int(kv["n"]), path=kv["path"], sources=[lg])
cells = {}
for clip, rows in stock.items():
    a, b = rows["origin_cap__stock"], rows["deliv_cap__stock"]
    assert (a["dy"], a["dx"]) == (b["dy"], b["dx"]), (clip, a, b)
    cells[clip] = {}
    for lab in LABELS:
        if lab in rows:
            cells[clip][lab] = rows[lab]
        else:
            p = f"{ROOT}/{clip}_{lab}/{clip}_inpainting_results_sbs.mkv"
            assert os.path.exists(p), p
            cells[clip][lab] = dict(dy=a["dy"], dx=a["dx"], leftPSNR=a["leftPSNR"], lpips=float("nan"), n=a["n"], path=p,
                                    sources=["shared left half: dy/dx from origin_cap__stock ROW"])
json.dump(dict(labels=LABELS, cells=cells, clips=sorted(cells)), open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(cells)} clips x {len(LABELS)} labels")
