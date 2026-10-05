#!/usr/bin/env python
"""final_judge: build ROWS_v1.json = {labels, cells[clip][label] = {path}} for judge_score_v1.py (CPU, read-only)."""
import json, os, sys
os.chdir("/home/kawa/master_project/StereoCrafter")
OUT = "scripts/distill/runs/deep_20261004/final_judge/ROWS_v1.json"
assert not os.path.exists(OUT), OUT
TEST = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
pub = json.load(open("scripts/distill/runs/more_20261004/eval_robustness/PUBLISHED_ROWS.json"))
sb = lambda root, c, tag: f"{root}/{c}_{tag}/{c}_inpainting_results_sbs.mkv"
R = [
    ("origin_ll", lambda c: pub["cells"][c]["origin_ll"]["path"]),
    ("mstudent2_step800_deliv_ll", lambda c: pub["cells"][c]["mstudent2_step800_deliv_ll"]["path"]),
    ("s25_ll", lambda c: pub["cells"][c]["s25_ll"]["path"]),
    ("AYS8_origin_g101", lambda c: sb("outputs/more_20261004/judge/ays/clips", c, "AYS8_origin_g101")),
    ("origin_g100_sd7", lambda c: sb("outputs/deep_20261004/sdedit/clips", c, "origin_g100_sd7")),
    ("origin_g100_sd1fill", lambda c: sb("outputs/deep_20261004/sdedit/clips", c, "origin_g100_sd1fill")),
    ("deliv_g100_sd7", lambda c: sb("outputs/deep_20261004/sdedit/clips", c, "deliv_g100_sd7")),
    ("deliv_g100_sd1fill", lambda c: sb("outputs/deep_20261004/sdedit/clips", c, "deliv_g100_sd1fill")),
    ("m2svid_fa_w16_ll", lambda c: sb("outputs/deep_20261004/external_models/clips", c, "m2svid_fa_w16_ll")),
    ("INPUT_fill", lambda c: sb("outputs/deep_20261004/sdedit/input_rows/clips", c, "INPUT_fill")),
    ("selMain250_s8", lambda c: sb("outputs/deep_20261004/scale_gt/renders", c, "selMain250_s8")),
    ("selMain250_s25", lambda c: sb("outputs/deep_20261004/scale_gt/renders", c, "selMain250_s25")),
    ("HIRES_B", lambda c: sb("outputs/deep_20261004/blur_diag/hiresB_r2/clips", c, "origin_upx175")),
    ("HIRES_B_L", lambda c: sb("outputs/deep_20261004/blur_diag/hiresB_r2/lanczos", c, "origin_upx175_L")),
    ("LEFT_AS_RIGHT", lambda c: "LEFTASRIGHT:" + pub["cells"][c]["origin_ll"]["path"]),
]
labels = [l for l, _ in R]
cells = {}
for c in TEST:
    cells[c] = {}
    for l, f in R:
        p = f(c)
        if os.path.exists(p.split("LEFTASRIGHT:")[-1]):
            cells[c][l] = dict(path=p)
json.dump(dict(labels=labels, cells=cells, clips=TEST), open(OUT, "w"), indent=1)
for c in TEST:
    print(c, len(cells[c]), [l for l in labels if l not in cells[c]])
