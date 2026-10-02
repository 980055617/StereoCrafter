#!/bin/bash
# Score the 4-clip stacking matrix, ALL LOSSLESS.
# rows: origin | origin+s25 | mamba | mamba+s25 | origin+student | (0301) mamba+student
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
B4=outputs/beyond4_lossless/clips; SK=outputs/skeptic1_stack/clips
O=scripts/distill/runs/skeptic1/SCORES_STACK_lossless.txt
export CUDA_VISIBLE_DEVICES=0
ARGS=""
for CL in 0301 0204 0052 0147; do
  ARGS="$ARGS $CL=$B4/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$B4/${CL}_s25_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_s25_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_student_ll/${CL}_inpainting_results_sbs.mkv"
  [ -d $SK/${CL}_mamba_student_ll ] && ARGS="$ARGS $CL=$SK/${CL}_mamba_student_ll/${CL}_inpainting_results_sbs.mkv"
done
echo "START $(date +%T)"
$PY $SC $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $O
echo "EXIT $? $(date +%T)"
