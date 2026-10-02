#!/bin/bash
# (1) score the 4-clip stacking matrix as soon as the stacking lanes finish
# (2) then, after the extension lanes, score the full 12-clip LOSSLESS origin/s25/student matrix
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
D=scripts/distill/runs/skeptic1
until ! systemctl --user is-active --quiet sk-gpu0 && ! systemctl --user is-active --quiet sk-gpu1; do sleep 20; done
echo "stack lanes done $(date +%T)"
bash $D/score_stack_v1.sh
echo "stack scored $(date +%T)"
until ! systemctl --user is-active --quiet sk-ext0 && ! systemctl --user is-active --quiet sk-ext1; do sleep 20; done
echo "ext lanes done $(date +%T)"
B4=outputs/beyond4_lossless/clips; SK=outputs/skeptic1_stack/clips
ARGS=""
for CL in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  ARGS="$ARGS $CL=$B4/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv"
  if [ -d $B4/${CL}_s25_ll ]; then ARGS="$ARGS $CL=$B4/${CL}_s25_ll/${CL}_inpainting_results_sbs.mkv"
  else ARGS="$ARGS $CL=$SK/${CL}_s25_ll/${CL}_inpainting_results_sbs.mkv"; fi
  ARGS="$ARGS $CL=$SK/${CL}_student_ll/${CL}_inpainting_results_sbs.mkv"
done
CUDA_VISIBLE_DEVICES=0 $PY $SC $ARGS 2>&1 \
  | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $D/SCORES_12CLIP_LOSSLESS.txt
echo "12clip lossless scored $(date +%T)"
echo SCORE_CHAIN_DONE
