#!/bin/bash
# input_side batch 6 (~30 min, PREREG_ADDENDUM_1_codec).  Run ONLY as:  flock /tmp/claude-gpu0.lock bash batch_06.sh
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/deep_20261004/input_side
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH6_LOCKED $(date +%F_%T)"
bash $D/run_driver_input_nolock_v1.sh $D/jobs_b6_renders.txt 0
bash $D/run_score_nolock_v1.sh $D/scorelist_b6.txt outputs/deep_20261004/input_side/scores_v1
echo "BATCH6_DONE $(date +%F_%T)"
