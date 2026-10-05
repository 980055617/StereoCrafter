#!/bin/bash
# S2 render batch (one half): run ONLY as   flock /tmp/claude-gpu0.lock bash .../batch_s2_render_part_v1.sh <jobfile>
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH_S2R $1 locked $(date +%F_%T)"
bash $L/run_jobs_nolock_v2.sh $1 0
echo "BATCH_S2R $1 done $(date +%F_%T)"
