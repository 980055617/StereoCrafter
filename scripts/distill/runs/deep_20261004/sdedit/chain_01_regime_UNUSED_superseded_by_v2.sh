#!/bin/bash
# S1: regime renders (0204 0052 0147; 0301 came from S0) -> K2 -> stage scoring (K3, M1, M1r, M3, M2) -> table with the
# pre-registered P4 rule -> visual-check panels.  GPU lock taken per job / per scoring step inside the called scripts.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN01 start $(date +%F_%T)"
bash $L/run_driver_sdedit_v1.sh $L/jobs_01_regime.txt 0
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v1.py $L/CHECK_K2_regime.txt 0301 0204 0052 0147 | tail -1
grep -q "SUMMARY checked=32 failed=0" $L/CHECK_K2_regime.txt || { echo "K2 regime FAIL or incomplete -- stopping before scoring"; exit 2; }
bash $L/score_stage_v1.sh regime 0301 0204 0052 0147
CUDA_VISIBLE_DEVICES= $PY $L/table_v1.py $O/score_regime P4 $L/TABLE_S1_REGIME.txt $L/TABLE_S1_REGIME.json > $L/table_s1.log 2>&1
echo "table rc=$?"
CUDA_VISIBLE_DEVICES= $PY $L/make_panels_v1.py $O/score_regime $L/TABLE_S1_REGIME.json $O/panels 0301 0204 0052 0147 > $O/panels_s1.log 2>&1
echo "panels rc=$?"
echo "CHAIN01 done $(date +%F_%T)"
