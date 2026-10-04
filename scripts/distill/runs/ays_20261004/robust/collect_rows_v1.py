#!/usr/bin/env python
"""[ays_20261004/robust COPY of scripts/distill/runs/more_20261004/eval_robustness/collect_published_v1.py; changes ONLY:
 the labels come from argv instead of a constant, and the excluded tree is scripts/distill/runs/ays_20261004/ (the
 concurrent lanes) instead of the eval_robustness dir.]

Collect the PUBLISHED score_clip_ll.py ROW lines for the given rows x 12 test clips.
Every ROW line under scripts/distill/runs/ (excluding ays_20261004/) is parsed; for each (clip, label) all ROW lines
must agree on lpips / offset / n / path, otherwise the script fails loudly.  The resolved render path of a cell is
the path printed in its ROW line, i.e. the file that produced the published number.
usage: python collect_rows_v1.py <out.json> <label> [<label> ...]     (the first label must be origin_ll)
"""
import glob
import json
import os
import re
import sys

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
LABELS = sys.argv[2:]
assert LABELS and LABELS[0] == "origin_ll", LABELS
ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(\S+) dx=(\S+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                 r"gtSharp=(\S+) rightPSNR=(\S+) n=(\S+) path=(\S+)")
EXCL = "scripts/distill/runs/ays_20261004"
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"

out = {c: {} for c in CLIPS}
files = sorted(set(glob.glob("scripts/distill/runs/**/*.txt", recursive=True)
                   + glob.glob("scripts/distill/runs/**/*.log", recursive=True)))
for f in files:
    if f.startswith(EXCL):
        continue
    try:
        lines = open(f, errors="replace").read().splitlines()
    except OSError:
        continue
    for line in lines:
        m = ROW.match(line.strip())
        if not m:
            continue
        clip, tag = m.group(1), m.group(2)
        if clip not in out:
            continue
        lab = tag[len(clip) + 1:] if tag.startswith(clip + "_") else tag
        if lab not in LABELS:
            continue
        rec = dict(dy=int(m.group(3)), dx=int(m.group(4)), leftPSNR=float(m.group(5)), lpips=float(m.group(6)),
                   sharp=float(m.group(7)), gtSharp=float(m.group(8)), rightPSNR=float(m.group(9)),
                   n=int(m.group(10)), path=m.group(11))
        if lab in out[clip]:
            old = out[clip][lab]
            for k in ("dy", "dx", "lpips", "n", "path", "sharp"):
                if old[k] != rec[k]:
                    sys.exit(f"CONFLICT {clip} {lab} {k}: {old[k]} ({old['sources'][0]}) vs {rec[k]} ({f})")
            old["sources"].append(f)
        else:
            rec["sources"] = [f]
            out[clip][lab] = rec

missing = [(c, l) for c in CLIPS for l in LABELS if l not in out[c]]
if missing:
    sys.exit(f"MISSING published cells: {missing}")
for c in CLIPS:
    for l in LABELS:
        p = out[c][l]["path"]
        if not os.path.exists(p):
            sys.exit(f"render missing on disk: {p}")
        if not p.endswith(".mkv"):
            sys.exit(f"not an FFV1 .mkv render: {p}")
means = {l: sum(out[c][l]["lpips"] for c in CLIPS) / len(CLIPS) for l in LABELS}
json.dump(dict(clips=CLIPS, labels=LABELS, cells=out, means=means), open(OUT, "w"), indent=1)
for l in LABELS:
    print(f"{l:28s} mean12 {means[l]:.6f}  ({means[l]:.4f})")
print("cells", sum(len(out[c]) for c in CLIPS), "files scanned", len(files))
