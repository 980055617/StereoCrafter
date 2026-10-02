#!/bin/bash
# Independently re-score beyond4's LOSSLESS origin_ll vs g125_ll on all 12 clips (their PART2 headline),
# and, for the codec-effect claim, the shipped mp4v origin of the same clips in the same pass.
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
L=outputs/beyond4_lossless/clips; C=outputs/fulldata_v2/clips
O=scripts/distill/runs/skeptic1/RESCORE_g125_12clip_lossless.txt
export CUDA_VISIBLE_DEVICES=1
ARGS=""
for CL in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  ARGS="$ARGS $CL=$L/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$L/${CL}_g125_ll/${CL}_inpainting_results_sbs.mkv"
done
echo "START $(date +%T)"
$PY $SC $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $O
echo "EXIT $? $(date +%T)"
