#!/bin/bash
# SKEPTIC re-scoring: independently re-run the project scorer on the ON-DISK mp4v outputs of the
# step-distillation lane (origin / s25 / student) plus the oracle+rescale controls, for all 12 clips.
# Uses beyond4/score_clip_ll.py (a verified faithful copy of the tracked score_clip.py that ALSO
# prints the frame count n and rightPSNR).  Nothing is overwritten: output goes to a new dir.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
OUT=scripts/distill/runs/skeptic1
C=outputs/fulldata_v2/clips
B=outputs/beyond_distil
export CUDA_VISIBLE_DEVICES=0
student() {
  case $1 in
    0301) echo $B/0301_smoke1_step800/0301_inpainting_results_sbs.mp4;;
    0204) echo $B/0204_smoke1_step800/0204_inpainting_results_sbs.mp4;;
    0052|0128|0125|0170) echo $B/$1_heldout_smoke1s800/$1_inpainting_results_sbs.mp4;;
    *) echo $B/$1_heldout_s800rest/$1_inpainting_results_sbs.mp4;;
  esac
}
ARGS=""
for CL in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  ARGS="$ARGS $CL=$C/${CL}_origin/${CL}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $CL=$C/${CL}_origin_s25/${CL}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $CL=$C/${CL}_all_8k_v2/${CL}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $CL=$(student $CL)"
done
for CL in 0301 0204; do
  ARGS="$ARGS $CL=$B/${CL}_oracle_m4/${CL}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $CL=$B/${CL}_oracle_m4_456/${CL}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $CL=$B/${CL}_rescale456/${CL}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $CL=$B/${CL}_onpolicy2_step400/${CL}_inpainting_results_sbs.mp4"
done
echo "START $(date +%T)"
$PY $SC $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $OUT/RESCORE_12clip_mp4v.txt
echo "EXIT $? $(date +%T)"
