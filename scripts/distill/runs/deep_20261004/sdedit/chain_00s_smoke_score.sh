#!/bin/bash
# S0 smoke scoring (tests the scoring chain end to end on 0301; gates nothing): waits for chain_00 to finish, K2 on 0301,
# score_stage (temporal skipped), table (rule SMOKE), panels.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until grep -qE "CHAIN00 done|K1 FAIL" $L/chain_00_smoke.log; do sleep 20; done
echo "CHAIN00s start $(date +%F_%T)"
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v1.py $L/CHECK_K2_smoke0301.txt 0301 | tail -1
SKIP_TEMPORAL=1 bash $L/score_stage_v1.sh smoke0301 0301
CUDA_VISIBLE_DEVICES= $PY $L/table_v1.py $O/score_smoke0301 SMOKE $L/TABLE_S0_SMOKE0301.txt $L/TABLE_S0_SMOKE0301.json > $L/table_s0.log 2>&1
echo "table rc=$?"
CUDA_VISIBLE_DEVICES= $PY $L/make_panels_v1.py $O/score_smoke0301 $L/TABLE_S0_SMOKE0301.json $O/panels_smoke 0301 > $O/panels_s0.log 2>&1
echo "panels rc=$?"
echo "CHAIN00s done $(date +%F_%T)"
