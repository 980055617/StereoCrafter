#!/bin/bash
# THE CHEAPEST EXPERIMENT THAT COULD MAKE THE TWO WINS COEXIST, and it needs no training:
# install the Mamba deliverable in its TWO down_blocks.0 slots only (MAMBA_SELF_ATTN_INCLUDE=down_blocks.0.*),
# leaving up_blocks.3.attn1 as real attention, which then accepts the 15 distilled tensors DIRECTLY.
# Rows: mamba_down0 (2-slot Mamba alone) and mamba_down0 + student.
set -u; cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/skeptic1
until ! systemctl --user is-active --quiet sk-ext0 && ! systemctl --user is-active --quiet sk-ext1 \
      && ! systemctl --user is-active --quiet sk-gpu0 && ! systemctl --user is-active --quiet sk-gpu1 \
      && ! systemctl --user is-active --quiet sk-ctrl && ! systemctl --user is-active --quiet sk-ctrl2 \
      && ! systemctl --user is-active --quiet sk-oraclem; do sleep 20; done
echo "all lanes clear $(date +%T)"
systemd-run --user --collect --unit=sk-d00 -p Environment=PATH=$PATH bash $PWD/$D/stack_driver_v3.sh $PWD/$D/jobs_down0_gpu0.txt 0
systemd-run --user --collect --unit=sk-d01 -p Environment=PATH=$PATH bash $PWD/$D/stack_driver_v3.sh $PWD/$D/jobs_down0_gpu1.txt 1
until ! systemctl --user is-active --quiet sk-d00 && ! systemctl --user is-active --quiet sk-d01; do sleep 20; done
echo "down0 lanes done $(date +%T)"
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
SC=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
B4=outputs/beyond4_lossless/clips; SK=outputs/skeptic1_stack/clips
ARGS=""
for CL in 0301 0204 0052 0147; do
  ARGS="$ARGS $CL=$B4/${CL}_origin_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_student_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_down0_ll/${CL}_inpainting_results_sbs.mkv"
  ARGS="$ARGS $CL=$SK/${CL}_mamba_down0_student_ll/${CL}_inpainting_results_sbs.mkv"
done
CUDA_VISIBLE_DEVICES=0 $PY $SC $ARGS 2>&1 \
  | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $D/SCORES_DOWN0.txt
echo DOWN0_CHAIN_DONE $(date +%T)
