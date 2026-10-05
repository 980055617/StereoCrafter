#!/bin/bash
# input_side batch 4 (~29 min).  Run ONLY as:  flock /tmp/claude-gpu0.lock bash batch_04.sh
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/deep_20261004/input_side
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH4_LOCKED $(date +%F_%T)"
bash $D/run_driver_input_nolock_v1.sh $D/jobs_b4_renders.txt 0
bash $D/run_score_nolock_v1.sh $D/scorelist_b4.txt outputs/deep_20261004/input_side/scores_v1
echo "BATCH4_DONE $(date +%F_%T)"
