#!/bin/bash
# S2 render batch: run ONLY as   flock /tmp/claude-gpu0.lock bash .../batch_s2_render_v1.sh
# the renders of jobs_02_ext12.txt (P4 passes x 8 non-regime test clips) via run_jobs_nolock_v2.sh, then K2 on all 12 clips.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH_S2R locked $(date +%F_%T)"
bash $L/run_jobs_nolock_v2.sh $L/jobs_02_ext12.txt 0
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v1.py $L/CHECK_K2_ext12.txt 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301 | tail -1
echo "BATCH_S2R done $(date +%F_%T)"
