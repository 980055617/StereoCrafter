#!/bin/bash
# S1 in batch mode (replaces chain_01b, stopped while its first deliv group only WAITED for the lock): one GPU-0 lock
# acquisition for renders + K2 + stage scoring (batch_s1_v1.sh), then the P4 table and the panels on CPU.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN01c start $(date +%F_%T)"
flock /tmp/claude-gpu0.lock bash $L/batch_s1_v1.sh
echo "batch rc=$? $(date +%F_%T)"
grep -q "STAGE regime done" $O/score_regime/stage.log || { echo "S1 stage scoring incomplete -- stopping"; exit 4; }
CUDA_VISIBLE_DEVICES= $PY $L/table_v1.py $O/score_regime P4 $L/TABLE_S1_REGIME.txt $L/TABLE_S1_REGIME.json > $L/table_s1.log 2>&1
echo "table rc=$?"
CUDA_VISIBLE_DEVICES= $PY $L/make_panels_v1.py $O/score_regime $L/TABLE_S1_REGIME.json $O/panels 0301 0204 0052 0147 > $O/panels_s1.log 2>&1
echo "panels rc=$?"
echo "CHAIN01 done $(date +%F_%T)"
