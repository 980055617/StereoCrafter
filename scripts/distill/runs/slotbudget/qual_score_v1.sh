#!/bin/bash
# SLOT BUDGET quality lane scoring: 12-clip lossless LPIPS, origin vs the 2-slot Mamba alone.
# origin rows come from the existing lossless baseline (outputs/beyond4_lossless);
# 4 of the 12 mamba_down0 rows are the EXISTING skeptic1 runs (byte-identical config, reused, not re-run);
# the 8 new ones live in outputs/slotbudget_down0_ll.  Nothing is overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/slotbudget
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
B4=outputs/beyond4_lossless/clips; SK=outputs/skeptic1_stack/clips; NEW=outputs/slotbudget_down0_ll/clips
ARGS=""
for CL in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  ARGS="$ARGS $CL=$B4/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv"
  if [ -f "$NEW/${CL}_mamba_down0_ll/${CL}_inpainting_results_sbs.mkv" ]; then
    ARGS="$ARGS $CL=$NEW/${CL}_mamba_down0_ll/${CL}_inpainting_results_sbs.mkv"
  else
    ARGS="$ARGS $CL=$SK/${CL}_mamba_down0_ll/${CL}_inpainting_results_sbs.mkv"
  fi
done
CUDA_VISIBLE_DEVICES=0 $PY $SC $ARGS 2>&1 \
  | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $R/SCORES_DOWN0_12CLIP.txt
echo QUAL_SCORE_DONE $(date +%T)
