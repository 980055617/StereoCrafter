#!/bin/bash
# final_judge: one judge_score_v1.py process per clip, each under the lock of GPU $GPU (other workflows interleave
# between clips).  usage: run_score_v1.sh <GPU 0|1> <out_dir (new)> <clip> [<clip> ...]
# 0125 uses the eval_robustness ADDENDUM wide grid (ddy -20,20 ddx -240,40), as every published 0125 value does.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/final_judge
GPU=$1; O=$2; shift 2
LOG=$L/$(basename $O)_gpu$GPU.log
mkdir -p "$O"
echo "START $(date '+%F_%T') out=$O gpu=$GPU clips=$*" >> "$LOG"
for c in "$@"; do
  if [ -e "$O/$c.json" ]; then echo "SKIP $c exists" >> "$LOG"; continue; fi
  if [ "$c" = "0125" ]; then GY=-20,20; GX=-240,40; else GY=-10,10; GX=-120,40; fi
  CUDA_VISIBLE_DEVICES=$GPU SCORE_STEP=4 REG_DDY=$GY REG_DDX=$GX flock /tmp/claude-gpu$GPU.lock \
    python $L/judge_score_v1.py "$O" $L/ROWS_v1.json "$c" >> "$LOG" 2>&1
  echo "CLIP_RC $c rc=$? $(date '+%F_%T')" >> "$LOG"
done
echo "END $(date '+%F_%T')" >> "$LOG"
