#!/usr/bin/env python
"""ays_20261004 / verify -- build the inputs of the GPU batch (CPU only, read-only on every other lane's files).

Writes (refuses to overwrite any of them):
  ROWS_reg_v1.json   rows file for score_registered_v1.py (V2: clips 0042 0259; origin_ll first)
  specs_V1.txt       one line per clip: "<clip> <spec> <spec> ..." for score_clip_ll.py (V1)
  specs_V4.txt       one line per clip for score_temporal_ll.py (V4; origin first)
  V0_BOOKKEEPING.txt writer_md5.txt sbs md5 vs the first field of the .mkv.md5 file, every render the batch reads
Published values come from MY parse of the ROW lines of the named score files (not from another lane's rows JSON).
Paths follow PREREG.txt (AYS5 origin per MODE_S1 = reuse; 0204 AYS8@1.00 from _r1).
usage: python build_rows_verify_v1.py      (exit 0 = all written, bookkeeping all OK; 3 = a bookkeeping mismatch)
"""
import json
import os
import re
import sys

os.chdir("/home/kawa/master_project/StereoCrafter")
L = "scripts/distill/runs/ays_20261004/verify"
OUTS = {k: f"{L}/{k}" for k in ("ROWS_reg_v1.json", "specs_V1.txt", "specs_V4.txt", "V0_BOOKKEEPING.txt",
                                  "V1_PUBLISHED_REFS.json")}
for p in OUTS.values():
    assert not os.path.exists(p), f"refusing to overwrite {p}"

CLIPS = "0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301".split()
REGIME = ["0052", "0147", "0204", "0301"]
REG_CLIPS = ["0042", "0259"]
A5L = "scripts/distill/runs/ays_20261004/ays5"
MODE = open(f"{A5L}/MODE_S1.txt").read().split()[0]
assert MODE == "reuse", MODE


def path(c, lab):
    if lab == "origin_ll":
        return f"outputs/beyond4_lossless/clips/{c}_origin_ll/{c}_inpainting_results_sbs.mkv"
    if lab == "AYS8_origin_g101":
        return f"outputs/more_20261004/judge/ays/clips/{c}_AYS8_origin_g101/{c}_inpainting_results_sbs.mkv"
    if lab == "AYS8_origin_g100":
        d = f"{c}_AYS8_origin_g100_r1" if c == "0204" else f"{c}_AYS8_origin_g100"
        return f"outputs/ays_20261004/ays5/clips/{d}/{c}_inpainting_results_sbs.mkv"
    if lab == "AYS5pad8_origin_g100":
        root = "outputs/more_20261004/judge/ays" if c in REGIME else "outputs/ays_20261004/ays5"
        return f"{root}/clips/{c}_AYS5pad8_origin_g100/{c}_inpainting_results_sbs.mkv"
    if lab == "mstudent2_step800_deliv_ll":
        return f"outputs/beyond_distil_mamba_scaled/clips/{c}_mstudent2_step800_deliv_ll/{c}_inpainting_results_sbs.mkv"
    if lab in ("deliv_g100_T5pad", "deliv_g100_T5nat", "origin_g100_T5pad"):
        return f"outputs/finalcheck_20261004/speed/clips/{c}_{lab}/{c}_inpainting_results_sbs.mkv"
    if lab == "s25_ll":
        return f"outputs/skeptic1_stack/clips/{c}_s25_ll/{c}_inpainting_results_sbs.mkv"
    if lab == "AYS5pad8_deliv_g100":
        assert c in REGIME
        return f"outputs/ays_20261004/ays5/clips/{c}_AYS5pad8_deliv_g100/{c}_inpainting_results_sbs.mkv"
    raise KeyError(lab)


ROWRE = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(-?\d+) dx=(-?\d+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                   r"gtSharp=(\S+) rightPSNR=(\S+) n=(\d+) path=(\S+)\s*$")


def rows_of(files):
    """path -> list of (file, dict) for every ROW line in the files."""
    out = {}
    for f in files:
        if not os.path.exists(f):
            continue
        for ln in open(f):
            m = ROWRE.match(ln)
            if not m:
                continue
            c, tag, dy, dx, lp, lpips, sh, gs, rp, n, p = m.groups()
            out.setdefault(p, []).append((f, dict(clip=c, tag=tag, dy=int(dy), dx=int(dx), leftPSNR=float(lp),
                                                  lpips=float(lpips), sharp=float(sh), gtSharp=float(gs),
                                                  rightPSNR=float(rp), n=int(n))))
    return out


def src_files(c):
    J = "scripts/distill/runs/more_20261004/judge"
    fs = [f"{A5L}/SCORES_AYS5_S2_{c}.txt", f"{A5L}/SCORES_AYS5_S3_{c}.txt",
          f"{J}/SCORES_J3c_AYS_ext8_{c}.txt", f"{J}/SCORES_J3c_AYS_regime_{c}.txt"]
    if c == "0301":
        fs.append(f"{J}/SCORES_J3c_AYS_0301.txt")
    return fs


def published(c, lab):
    p = path(c, lab)
    hits = rows_of(src_files(c)).get(p, [])
    if not hits:
        return p, None, []
    vals = {(h["dy"], h["dx"], h["n"], h["lpips"]) for _, h in hits}
    assert len(vals) == 1, f"published ROW values disagree for {p}: {vals}"
    return p, hits[0][1], [f for f, _ in hits]


lines_v0, bad = [], 0


def bookkeeping(p):
    global bad
    d = os.path.dirname(p)
    w = [ln.split()[0] for ln in open(f"{d}/writer_md5.txt") if "_sbs" in ln]
    m = open(p + ".md5").read().split()[0] if os.path.exists(p + ".md5") else None
    ok = os.path.exists(p) and len(w) == 1 and w[0] == m
    bad += (not ok)
    lines_v0.append(f"{'OK ' if ok else 'BAD'} writer={w[0] if w else None} mkv.md5={m} size={os.path.getsize(p) if os.path.exists(p) else None} {p}")
    return w[0] if w else None


# ---------------------------------------------------------------- V2 rows file
REG_LABELS = ["origin_ll", "AYS8_origin_g101", "mstudent2_step800_deliv_ll", "deliv_g100_T5pad", "origin_g100_T5pad",
              "deliv_g100_T5nat", "s25_ll", "AYS5pad8_origin_g100", "AYS8_origin_g100"]
cells = {}
for c in REG_CLIPS:
    cells[c] = {}
    for lab in REG_LABELS:
        p, h, srcs = published(c, lab)
        assert h is not None, f"no published ROW for {c} {lab} ({p})"
        cells[c][lab] = dict(path=p, dy=h["dy"], dx=h["dx"], lpips=h["lpips"], sharp=h["sharp"], n=h["n"],
                             rightPSNR=h["rightPSNR"], sources=srcs, writer_md5=bookkeeping(p))
json.dump(dict(clips=REG_CLIPS, labels=REG_LABELS, cells=cells), open(OUTS["ROWS_reg_v1.json"], "w"), indent=1)

# ---------------------------------------------------------------- V1 / V4 specs (+ published V1 references)
V1_LABELS = ["origin_ll", "AYS5pad8_origin_g100", "deliv_g100_T5pad", "AYS8_origin_g100"]
ref = {}
with open(OUTS["specs_V1.txt"], "w") as f1, open(OUTS["specs_V4.txt"], "w") as f4:
    for c in CLIPS:
        sp = []
        for lab in V1_LABELS:
            p, h, srcs = published(c, lab)
            assert h is not None, f"no published ROW for {c} {lab} ({p})"
            ref[f"{c}/{lab}"] = dict(path=p, **{k: h[k] for k in ("dy", "dx", "n", "lpips", "sharp")}, sources=srcs,
                                     writer_md5=bookkeeping(p))
            sp.append(f"{c}={p}")
        f1.write(c + " " + " ".join(sp) + "\n")
        v4 = ["origin_ll", "deliv_g100_T5pad", "AYS5pad8_origin_g100"] + (["AYS5pad8_deliv_g100"] if c in REGIME else [])
        for lab in v4:
            if lab == "AYS5pad8_deliv_g100":
                bookkeeping(path(c, lab))
        f4.write(c + " " + " ".join(f"{c}={path(c, lab)}" for lab in v4) + "\n")
json.dump(ref, open(OUTS["V1_PUBLISHED_REFS.json"], "x"), indent=1)

open(OUTS["V0_BOOKKEEPING.txt"], "w").write(
    "V0 bookkeeping (PREREG): writer_md5.txt sbs md5 == first field of <mkv>.md5, every render the batch reads\n"
    + "\n".join(sorted(set(lines_v0))) + f"\nSUMMARY renders={len(set(lines_v0))} bad={bad}\n")
print(open(OUTS["V0_BOOKKEEPING.txt"]).read().splitlines()[-1])
sys.exit(3 if bad else 0)
