#!/usr/bin/env python
"""ays_20261004 / robust -- PREREG (1b): readiness + pairing check of the 12 AYS5 origin renders, then the rows file
for the AYS5 registered pass.  CPU only, read-only on every other lane's files.

AYS5 render of a clip = resolved exactly like the ays5 lane's own scorer (score_ays5_v1.sh, MODE_S1.txt):
  MODE reuse -> judge's outputs/more_20261004/judge/ays/clips/<c>_AYS5pad8_origin_g100 for 0052 0147 0204 0301,
                the ays5 lane's outputs/ays_20261004/ays5/clips/<c>_AYS5pad8_origin_g100 for the other 8
  MODE own   -> the ays5 lane's dir for all 12
A clip is COMPLETE iff its dir has the sbs .mkv, writer_md5.txt and speed_log.json, AND the producing lane's timing log
has a RUN line for it with rc=0 and the same md5.  Pairing/config: speed_log.json guid 1.0, the judge's AYS5 sigmas,
rng_pad_to 8, unet_state None, and per-window init_md5 identical to finalcheck <c>_origin_g100_T5pad (all windows).
exit 0 = rows written; 10 = not ready yet (poll again); 20 = a pairing/config check FAILED (never score).
usage: python build_rows_ays5_v1.py <out_rows.json> <out_check.txt>
"""
import glob
import json
import math
import os
import re
import sys

os.chdir("/home/kawa/master_project/StereoCrafter")
OUT, CHK = sys.argv[1], sys.argv[2]
for p in (OUT, CHK):
    assert not os.path.exists(p), f"refusing to overwrite {p}"
R = "scripts/distill/runs/ays_20261004/robust"
A5L = "scripts/distill/runs/ays_20261004/ays5"
A5O = "outputs/ays_20261004/ays5"
JA = "outputs/more_20261004/judge/ays"
F = "outputs/finalcheck_20261004/speed/clips"
REGIME = ["0052", "0147", "0204", "0301"]
LAB = "AYS5pad8_origin_g100"
SIG = [700.0, 11.25711441040039, 1.7890000343322754, 0.26404356956481934, 0.0020000000949949026]
ROWS1 = json.load(open(f"{R}/ROWS_pass1.json"))
CLIPS = ROWS1["clips"]
lines = []


def out(s):
    lines.append(s)
    print(s, flush=True)


if not os.path.exists(f"{A5L}/MODE_S1.txt"):
    print("NOT_READY MODE_S1.txt missing")
    sys.exit(10)
MODE = open(f"{A5L}/MODE_S1.txt").read().split()[0]
if MODE not in ("reuse", "own"):
    print(f"NOT_READY bad MODE {MODE!r}")
    sys.exit(10)


def resolve(c):
    if MODE == "reuse" and c in REGIME:
        return f"{JA}/clips/{c}_{LAB}", JA
    return f"{A5O}/clips/{c}_{LAB}", A5O


def run_ok(root, c, md5):
    for t in glob.glob(f"{root}/timing_gpu*.txt"):
        for ln in open(t, errors="replace"):
            if ln.startswith(f"RUN {c} {LAB} ") and " rc=0 " in ln and f"md5={md5}" in ln:
                return True
    return False


notready, fails, cells = [], [], {}
for c in CLIPS:
    d, root = resolve(c)
    mkv = f"{d}/{c}_inpainting_results_sbs.mkv"
    if not (os.path.exists(mkv) and os.path.exists(f"{d}/writer_md5.txt") and os.path.exists(f"{d}/speed_log.json")):
        notready.append(c)
        continue
    md5 = open(f"{d}/writer_md5.txt").read().split()[0]
    if not run_ok(root, c, md5):
        notready.append(c)
        continue
    j = json.load(open(f"{d}/speed_log.json"))
    ref = json.load(open(f"{F}/{c}_origin_g100_T5pad/speed_log.json"))
    fa, fb = [w["init_md5"] for w in j["windows"]], [w["init_md5"] for w in ref["windows"]]
    same = sum(x == y for x, y in zip(fa, fb))
    errs = []
    if not (same == len(fa) == len(fb)):
        errs.append(f"init_md5 {same}/{len(fa)} identical (ref {len(fb)})")
    if abs(float(j["guid"]) - 1.0) > 1e-9:
        errs.append(f"guid {j['guid']}")
    if [float(x) for x in j["sigmas"]] != SIG:
        errs.append(f"sigmas {j['sigmas']}")
    if j.get("rng_pad_to") != 8:
        errs.append(f"rng_pad_to {j.get('rng_pad_to')}")
    if j.get("unet_state") is not None:
        errs.append(f"unet_state {j.get('unet_state')}")
    if errs:
        fails.append((c, errs))
    cells[c] = dict(path=mkv, md5=md5, pairing=f"{same}/{len(fa)}", dir=d)

if fails:
    for c, e in fails:
        out(f"PAIRING/CONFIG FAIL {c}: {e}")
    open(CHK, "w").write("\n".join(lines) + "\n")
    sys.exit(20)
if notready:
    print(f"NOT_READY mode={MODE} incomplete clips: {notready}")
    sys.exit(10)

# published lpips of the AYS5 rows (for the scorer's log line and gate R0'): ays5 lane S2 scores, else the judge's
ROW = re.compile(r"^ROW clip=(\S+) tag=(\S+) dy=(\S+) dx=(\S+) leftPSNR=(\S+) lpips=(\S+) sharp=(\S+) "
                 r"gtSharp=(\S+) rightPSNR=(\S+) n=(\S+) path=(\S+)")
pub = {}
for f in sorted(glob.glob(f"{A5L}/SCORES_*.txt") + glob.glob("scripts/distill/runs/more_20261004/judge/SCORES_J3c_AYS_*.txt")):
    for ln in open(f, errors="replace"):
        m = ROW.match(ln.strip())
        if m and m.group(2) == f"{m.group(1)}_{LAB}" and m.group(1) in cells and m.group(11) == cells[m.group(1)]["path"]:
            rec = dict(lpips=float(m.group(6)), sharp=float(m.group(7)), dy=int(m.group(3)), dx=int(m.group(4)),
                       n=int(m.group(10)), source=f)
            old = pub.get(m.group(1))
            if old and (old["lpips"], old["dy"], old["dx"], old["n"]) != (rec["lpips"], rec["dy"], rec["dx"], rec["n"]):
                out(f"ROW CONFLICT {m.group(1)}: {old} vs {rec}")
                open(CHK, "w").write("\n".join(lines) + "\n")
                sys.exit(20)
            pub.setdefault(m.group(1), rec)

labels = ["origin_ll", "mstudent2_step800_deliv_ll", "deliv_g100_T5pad", "origin_g100_T5pad", LAB]
res = {c: {lab: ROWS1["cells"][c][lab] for lab in labels[:4]} for c in CLIPS}
for c in CLIPS:
    p = pub.get(c)
    res[c][LAB] = dict(path=cells[c]["path"], lpips=(p["lpips"] if p else float("nan")),
                       sharp=(p["sharp"] if p else float("nan")), dy=(p["dy"] if p else None), dx=(p["dx"] if p else None),
                       n=(p["n"] if p else None), sources=([p["source"]] if p else []), writer_md5=cells[c]["md5"])
    out(f"OK {c} mode={MODE} {cells[c]['dir']} md5 {cells[c]['md5']} pairing {cells[c]['pairing']} vs "
        f"{F}/{c}_origin_g100_T5pad; ROW lpips {res[c][LAB]['lpips']} ({res[c][LAB]['sources'][0] if p else 'none yet'})")
json.dump(dict(clips=CLIPS, labels=labels, cells=res, mode=MODE), open(OUT, "w"), indent=1)
out(f"ROWS_WRITTEN {OUT} mode={MODE}")
open(CHK, "w").write("\n".join(lines) + "\n")
sys.exit(0)
