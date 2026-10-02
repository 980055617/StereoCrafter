#!/usr/bin/env python
"""Resolve labelled lossless runs to their FFV1 files and score them with the canonical scorer.

All quality numbers in this directory come from
    scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py   (real-GT LPIPS, SCORE_STEP=4)
on FFV1 .mkv output written by beyond4/infer_lossless.py.  utils/inpainting.py's cv2 mp4v writer moves
measured LPIPS by -0.0100..+0.0104 per clip -- larger than the entire s25 gain and sign-changing -- so no
mp4v row is ever mixed in here.

A label is looked up, in order, under
    outputs/beyond_distil_mamba/clips/   (this run)
    outputs/skeptic1_stack/clips/        (the shipped-Mamba / stacking rows)
    outputs/beyond4_lossless/clips/      (the origin / s25 baselines)
so the established rows are REUSED, never recomputed.

usage: score_rows_v1.py <out.txt> <clips csv> <labels csv>
"""
import os, subprocess, sys

REPO = "/home/kawa/master_project/StereoCrafter"
os.chdir(REPO)
SEARCH = ["outputs/beyond_distil_mamba_scaled/clips", "outputs/beyond_distil_mamba/clips",
          "outputs/skeptic1_stack/clips", "outputs/beyond4_lossless/clips"]
SCORER = "scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py"
PY = "/home/kawa/miniconda3/envs/stereocrafter/bin/python"

out_path, clips_csv, labels_csv = sys.argv[1], sys.argv[2], sys.argv[3]
CLIPS = [c for c in clips_csv.split(",") if c]
LABELS = [l for l in labels_csv.split(",") if l]

args, missing = [], []
for cl in CLIPS:
    for lab in LABELS:
        hit = None
        for root in SEARCH:
            d = os.path.join(root, f"{cl}_{lab}")
            for name in (f"{cl}_inpainting_results_sbs.mkv", f"{cl}_inpainting_results_sbs.mp4"):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    hit = p
                    break
            if hit:
                break
        if hit is None:
            missing.append(f"{cl}_{lab}")
        elif hit.endswith(".mp4"):
            missing.append(f"{cl}_{lab} (mp4 only -- REFUSED, lossless required)")
        else:
            args.append(f"{cl}={hit}")

print(f"[score] {len(args)} rows, {len(missing)} missing", flush=True)
for m in missing:
    print(f"[score][missing] {m}", flush=True)
if not args:
    sys.exit(2)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1"), SCORE_STEP="4")
res = subprocess.run([PY, SCORER] + args, capture_output=True, text=True, env=env)
keep = [l for l in res.stdout.splitlines()
        if l.strip() and not any(t in l.lower() for t in ("warning", "setting up", "loading model", "self.load_state"))
        and not l.startswith("/home/kawa")]
with open(out_path, "w") as fh:
    fh.write("\n".join(keep) + "\n")
    if missing:
        fh.write("\n# MISSING ROWS (not scored):\n" + "\n".join("#   " + m for m in missing) + "\n")
print("\n".join(keep))
if res.returncode != 0:
    print(res.stderr[-2000:], file=sys.stderr)
sys.exit(res.returncode)
