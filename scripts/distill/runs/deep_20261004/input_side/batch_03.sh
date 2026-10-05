#!/bin/bash
# input_side batch 3 (~31 min).  Run ONLY as:  flock /tmp/claude-gpu0.lock bash batch_03.sh
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/deep_20261004/input_side
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH3_LOCKED $(date +%F_%T)"
bash $D/run_driver_input_nolock_v1.sh $D/jobs_b3_renders.txt 0
echo "BATCH3_DONE $(date +%F_%T)"
