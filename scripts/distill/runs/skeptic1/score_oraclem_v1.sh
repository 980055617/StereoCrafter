#!/bin/bash
# Score the Mamba-side oracle gate against mamba and mamba+s25.
set -u; cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
B4=outputs/beyond4_lossless/clips; SK=outputs/skeptic1_stack/clips
D=scripts/distill/runs/skeptic1
until ! systemctl --user is-active --quiet sk-oraclem; do sleep 20; done
ARGS=""
for CL in 0301 0052; do
  [ -d $SK/${CL}_mamba_oracle456_ll ] || continue
  ARGS="$ARGS $CL=$B4/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$B4/${CL}_s25_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_s25_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_oracle456_ll/${CL}_inpainting_results_sbs.mkv"
done
CUDA_VISIBLE_DEVICES=1 $PY $SC $ARGS 2>&1 \
  | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $D/SCORES_ORACLEM.txt
echo ORACLEM_SCORED $(date +%T)
