#!/bin/bash
# PREREG_ADDENDUM_boundary.txt: re-score ONCE, with the widened registration grid, every clip whose score_v1 file has
# a raw or clip optimum on the boundary.  usage: run_score_v1w.sh <clip> [<clip> ...]
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/more_20261004/eval_robustness
O=outputs/more_20261004/eval_robustness/score_v1_wide
LOG=$L/score_v1_wide.log
if [ -e "$O" ]; then echo "REFUSING: $O exists" >> "$LOG"; exit 1; fi
mkdir -p "$O"
echo "SCORE_V1W_START $(date '+%F_%T') out=$O clips=$* grid ddy=-20,20 ddx=-240,40" >> "$LOG"
for c in "$@"; do
  CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 REG_DDY=-20,20 REG_DDX=-240,40 flock /tmp/claude-gpu1.lock \
    python $L/score_registered_v1w.py "$O" $L/PUBLISHED_ROWS.json "$c" >> "$LOG" 2>&1
  echo "CLIP_RC $c rc=$? $(date '+%F_%T')" >> "$LOG"
done
echo "SCORE_V1W_END $(date '+%F_%T')" >> "$LOG"
