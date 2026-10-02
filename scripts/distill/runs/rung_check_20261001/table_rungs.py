#!/usr/bin/env python
"""TASK 1 TABLE: the six configs on all 12 TEST clips, lossless FFV1, deployed config.

Reads the ROW lines of the scoring logs in outputs/rung_check_20261001/ and prints one table
with the rows origin / shipped 5-slot Mamba / step200 / step400 / step800 (the deliverable) /
origin+s25 -- per clip and mean, delta vs origin, clips improved, sharpness/GT per clip and mean,
worst clip.

sh/GT is the MEAN OF PER-CLIP RATIOS (which is how the published 1.071 was computed), not the
ratio of means.
"""
import glob
import json
import os
import re
import sys

import numpy as np

OUT = "/home/kawa/master_project/StereoCrafter/outputs/rung_check_20261001"
CLIPS = ["0042", "0052", "0125", "0128", "0141", "0147",
         "0170", "0204", "0225", "0251", "0259", "0301"]
ROWS = [
    ("origin (deployed 8 steps)", "origin_ll"),
    ("mamba 5-slot (SHIPPED)", "mamba_ll"),
    ("mstudent2 step200", "mstudent2_step200_ll"),
    ("mstudent2 step400", "mstudent2_step400_ll"),
    ("step800 = THE DELIVERABLE", "mstudent2_step800_deliv_ll"),
    ("origin + s25 (2.4x cost)", "s25_ll"),
]
REF = {"origin_ll": 0.3933, "mamba_ll": 0.3927,
       "mstudent2_step800_deliv_ll": 0.3804, "s25_ll": 0.3786}

D, GTS = {}, {}
pat = re.compile(r"^ROW clip=(\S+) tag=(\S+) .*lpips=([\d.]+) sharp=([\d.]+) gtSharp=([\d.]+) "
                 r"rightPSNR=([-\d.]+) n=(\d+) path=(\S+)")
nrow = 0
for f in sorted(glob.glob(f"{OUT}/scores_*.txt")):
    for line in open(f):
        m = pat.match(line.strip())
        if not m:
            continue
        clip, tag, lp, sh, gs, rp, n, path = m.groups()
        label = tag[len(clip) + 1:]
        D[(clip, label)] = dict(lpips=float(lp), sharp=float(sh), gtSharp=float(gs),
                                rightPSNR=float(rp), n=int(n), path=path)
        GTS[clip] = float(gs)
        nrow += 1

rep = open(f"{OUT}/TABLE_RUNGS_12CLIP.txt", "w")


def tee(*s):
    print(*s, flush=True)
    print(*s, file=rep, flush=True)


missing = [(c, t) for c in CLIPS for _, t in ROWS if (c, t) not in D]
tee("=" * 142)
tee("TASK 1 -- THE CONSERVATIVE RUNGS ON THE 12 TEST CLIPS")
tee("12 TEST clips, LOSSLESS FFV1 (never the cv2 mp4v writer), real-GT LPIPS, DEPLOYED CONFIG:")
tee("8 sampler steps, EulerDiscrete Karras, guidance 1.01, 14-frame windows overlap 3, 576x1024 centre crop.")
tee("scorer scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py  SCORE_STEP=4   (ROW lines parsed: "
    f"{nrow})")
tee("step200/step400 were rendered by this run through scripts/distill/runs/skeptic1/infer_ll_hook.py with")
tee("SK_UNET=<the shipped 5-slot Mamba> and SK_CK=<rung>.pt -- the identical path the dev ladder used, and")
tee("the path RESULTS.txt section 6 proved bit-identical to the merged deliverable file on all 1518 params.")
tee("=" * 142)
if missing:
    tee(f"!! MISSING {len(missing)} cells: {missing}")

tee("")
tee("--- PER CLIP: LPIPS (lower is better) ---")
tee(f"  {'row':28s}" + "".join(f"{c:>8s}" for c in CLIPS))
tee(f"  {'GT sharpness':28s}" + "".join(f"{GTS.get(c, float('nan')):8.4f}" for c in CLIPS))
for lab, tag in ROWS:
    tee(f"  {lab:28s}" + "".join(f"{D[(c,tag)]['lpips']:8.4f}" if (c, tag) in D else f"{'--':>8s}"
                                 for c in CLIPS))

ov = np.array([D[(c, "origin_ll")]["lpips"] for c in CLIPS])
tee("")
tee("--- PER CLIP: delta vs deployed ORIGIN (negative = better) ---")
for lab, tag in ROWS[1:]:
    tee(f"  {lab:28s}" + "".join(f"{D[(c,tag)]['lpips']-D[(c,'origin_ll')]['lpips']:+8.4f}"
                                 for c in CLIPS))

tee("")
tee("--- PER CLIP: sharpness / GT sharpness (1.0 = matches the real right eye) ---")
for lab, tag in ROWS:
    tee(f"  {lab:28s}" + "".join(f"{D[(c,tag)]['sharp']/D[(c,tag)]['gtSharp']:8.3f}" for c in CLIPS))

tee("")
tee("=" * 142)
tee("MEANS over n=12 (all test clips).  step200/step400/step800 trained only on TRAIN-split clips,")
tee("so all three are fully held out here; origin / mamba / s25 involve no student at all.")
tee("=" * 142)
tee(f"  {'row':28s} {'meanLPIPS':>10s} {'d vs origin':>11s} {'d vs mamba':>10s} {'improved':>9s} "
    f"{'d vs s800':>10s} {'sh/GT':>7s} {'sh/mamba':>9s} {'worst clip':>18s} {'rPSNR':>7s} {'ref':>8s}")
MEANS, SUM = {}, {}
mamba = np.array([D[(c, "mamba_ll")]["lpips"] for c in CLIPS])
s800 = np.array([D[(c, "mstudent2_step800_deliv_ll")]["lpips"] for c in CLIPS])
for lab, tag in ROWS:
    x = np.array([D[(c, tag)]["lpips"] for c in CLIPS])
    shg = np.array([D[(c, tag)]["sharp"] / D[(c, tag)]["gtSharp"] for c in CLIPS])
    shm = np.array([D[(c, tag)]["sharp"] / D[(c, "mamba_ll")]["sharp"] for c in CLIPS])
    rp = np.array([D[(c, tag)]["rightPSNR"] for c in CLIPS])
    d = x - ov
    imp = int((d < 0).sum())
    wi = int(d.argmax())
    ref = REF.get(tag)
    MEANS[tag] = float(x.mean())
    SUM[tag] = dict(label=lab, mean=float(x.mean()), dOrigin=float(d.mean()),
                    dMamba=float((x - mamba).mean()), dS800=float((x - s800).mean()),
                    improved=imp, shGT=float(shg.mean()), shMamba=float(shm.mean()),
                    worstClip=CLIPS[wi], worstDelta=float(d[wi]), rPSNR=float(rp.mean()),
                    perClip={c: float(v) for c, v in zip(CLIPS, x)},
                    perClipShGT={c: float(v) for c, v in zip(CLIPS, shg)})
    tee(f"  {lab:28s} {x.mean():10.4f} {d.mean():+11.4f} {(x-mamba).mean():+10.4f} "
        f"{f'{imp}/12':>9s} {(x-s800).mean():+10.4f} {shg.mean():7.3f} {shm.mean():9.4f} "
        f"{f'{CLIPS[wi]} {d[wi]:+.4f}':>18s} {rp.mean():7.3f} "
        f"{(f'{ref:.4f}' if ref else '-'):>8s}")

tee("")
tee("HARNESS ANCHOR -- the four reused rows were RE-SCORED, not transcribed:")
for tag, ref in REF.items():
    got = MEANS[tag]
    tee(f"  {tag:30s} re-scored {got:.5f}   published {ref:.4f}   diff {got-ref:+.5f}"
        f"   {'OK' if abs(got-ref) < 0.0001 else 'MISMATCH'}")

tee("")
tee("--- RANK by 12-clip mean LPIPS ---")
for i, (tag, m) in enumerate(sorted(MEANS.items(), key=lambda kv: kv[1]), 1):
    tee(f"  {i}. {SUM[tag]['label']:28s} {m:.4f}   d vs origin {SUM[tag]['dOrigin']:+.4f}   "
        f"improved {SUM[tag]['improved']}/12   sh/GT {SUM[tag]['shGT']:.3f}")

tee("")
tee("--- THE PRE-REGISTERED RULE APPLIED (outputs/rung_check_20261001/SHIP_RULE_PREREG.txt) ---")
for tag in ("mstudent2_step200_ll", "mstudent2_step400_ll"):
    give = SUM[tag]["mean"] - SUM["mstudent2_step800_deliv_ll"]["mean"]
    tee(f"  {SUM[tag]['label']:22s} gives up {give:+.4f} LPIPS vs step800  -> "
        f"{'R2(a) PASS (<=+0.0005)' if give <= 0.0005 else ('R4 band (+0.0005..+0.0020)' if give <= 0.0020 else 'R3 FAIL (>+0.0020): step800 stands')}"
        f"   improved {SUM[tag]['improved']}/12 {'(R5 ok)' if SUM[tag]['improved'] == 12 else '(R5 FAIL)'}")

json.dump(SUM, open(f"{OUT}/table_rungs.json", "w"), indent=1)
tee("")
tee("--- RENDER PROVENANCE ---")
for lab, tag in ROWS:
    roots = sorted({os.path.dirname(os.path.dirname(D[(c, tag)]["path"])) for c in CLIPS})
    tee(f"  {lab:28s} label={tag:28s} roots={roots}")
rep.close()
print("\nDONE table")
