#!/bin/bash
# S2 scoring batch: run ONLY as   flock /tmp/claude-gpu0.lock bash .../batch_s2_score_v1.sh
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH_S2S locked $(date +%F_%T)"
bash $L/score_stage_nolock_v1.sh ext12 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301
echo "BATCH_S2S done $(date +%F_%T)"
