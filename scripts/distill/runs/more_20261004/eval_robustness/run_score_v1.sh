#!/bin/bash
# Full 12-clip registered-GT scoring run (eval_robustness lane).  One process per clip, each under the GPU-1 lock,
# so other lanes can interleave between clips.  Output dir must not exist yet (never overwrite).
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/more_20261004/eval_robustness
O=outputs/more_20261004/eval_robustness/score_v1
LOG=$L/score_v1.log
if [ -e "$O" ]; then echo "REFUSING: $O exists" >> "$LOG"; exit 1; fi
mkdir -p "$O"
echo "SCORE_V1_START $(date '+%F_%T') out=$O" >> "$LOG"
for c in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 flock /tmp/claude-gpu1.lock \
    python $L/score_registered_v1.py "$O" $L/PUBLISHED_ROWS.json "$c" >> "$LOG" 2>&1
  echo "CLIP_RC $c rc=$? $(date '+%F_%T')" >> "$LOG"
done
echo "SCORE_V1_END $(date '+%F_%T')" >> "$LOG"
