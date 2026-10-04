#!/usr/bin/env python
"""ays_20261004/ays5 OPTIONAL registered-GT check (PREREG.txt "OPTIONAL"): build the rows JSON that
scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py (UNCHANGED) reads.  CPU only.
Labels (origin_ll FIRST: the scorer takes the scorer window from it; origin_ll and mstudent2_step800_deliv_ll are also
needed by its inspection panel): origin_ll, mstudent2_step800_deliv_ll, deliv_g100_T5pad, AYS5pad8_origin_g100
[+ AYS8_origin_g100 if this lane's S3 scores exist].  Paths/offsets/lpips come from this lane's S2/S3 ROW lines;
mstudent2_step800_deliv_ll from eval_robustness PUBLISHED_ROWS.json.  usage: build_rows_registered_v1.py OUT.json"""
import glob, json, os, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
L = "scripts/distill/runs/ays_20261004/ays5"
PUB = json.load(open("scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json"))
CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
OUT = sys.argv[1]
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"


def load(tag):
    V = {}
    for f in sorted(glob.glob(f"{L}/SCORES_AYS5_{tag}_*.txt")):
        if "SCORE_DONE rc=0" not in open(f).read():
            continue
        for ln in open(f):
            if ln.startswith("ROW "):
                d = dict(kv.split("=", 1) for kv in ln.split()[1:])
                c = d["clip"]
                V[(c, d["tag"][len(c) + 1:])] = dict(dy=int(d["dy"]), dx=int(d["dx"]), lpips=float(d["lpips"]),
                                                      sharp=float(d["sharp"]), n=int(d["n"]), path=d["path"], src=f)
    return V


S2, S3 = load("S2"), load("S3")
labels = ["origin_ll", "mstudent2_step800_deliv_ll", "deliv_g100_T5pad", "AYS5pad8_origin_g100"]
if all((c, "AYS8_origin_g100") in S3 for c in CLIPS):
    labels.append("AYS8_origin_g100")
cells = {}
for c in CLIPS:
    cells[c] = {}
    for lab in labels:
        if lab == "mstudent2_step800_deliv_ll":
            p = PUB["cells"][c][lab]
            cells[c][lab] = dict(dy=p["dy"], dx=p["dx"], lpips=p["lpips"], path=p["path"], src="eval_robustness/PUBLISHED_ROWS.json")
        else:
            v = (S3 if lab == "AYS8_origin_g100" else S2)[(c, lab)]
            cells[c][lab] = dict(dy=v["dy"], dx=v["dx"], lpips=v["lpips"], path=v["path"], src=v["src"])
json.dump(dict(clips=CLIPS, labels=labels, cells=cells), open(OUT, "w"), indent=1)
print(f"wrote {OUT}: labels={labels}")
