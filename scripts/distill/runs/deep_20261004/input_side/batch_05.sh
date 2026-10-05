#!/bin/bash
# input_side batch 5 (~18 min).  Run ONLY as:  flock /tmp/claude-gpu0.lock bash batch_05.sh
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/deep_20261004/input_side
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH5_LOCKED $(date +%F_%T)"
bash $D/run_driver_input_nolock_v1.sh $D/jobs_b5_renders.txt 0
bash $D/run_score_nolock_v1.sh $D/scorelist_b5.txt outputs/deep_20261004/input_side/scores_v1
echo "BATCH5_DONE $(date +%F_%T)"
