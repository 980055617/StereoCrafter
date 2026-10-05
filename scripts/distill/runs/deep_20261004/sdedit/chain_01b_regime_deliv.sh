#!/bin/bash
# S1 continuation (replaces the remainder of chain_01_regime_v2.sh, which was stopped while it only polled for the smoke
# scoring to finish): the S0 integrity gates K2 (8/8) and K3 (10/10) are already on disk, so the DELIV regime renders start
# now; the S1 stage scoring waits for the smoke scoring (chain_00s) so the table/panel scripts are tested first.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN01b start $(date +%F_%T)"
grep -q "SUMMARY checked=8 failed=0" $L/CHECK_K2_smoke0301.txt || { echo "S0 K2 not 8/8 PASS -- stopping before deliv renders"; exit 2; }
grep -q "SUMMARY checked=10 failed=0" $O/score_smoke0301/K3_passthrough.txt || { echo "S0 K3 not 10/10 PASS -- stopping"; exit 2; }
echo "S0 integrity K1/K2/K3 PASS -> deliv regime renders $(date +%F_%T)"
bash $L/run_driver_sdedit_v2.sh $L/jobs_01b_regime_deliv.txt 0
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v1.py $L/CHECK_K2_regime.txt 0301 0204 0052 0147 | tail -1
grep -q "SUMMARY checked=32 failed=0" $L/CHECK_K2_regime.txt || { echo "K2 regime FAIL or incomplete -- stopping before scoring"; exit 3; }
until grep -qE "CHAIN00s done" $L/chain_00s_smoke_score.log 2>/dev/null; do sleep 30; done
grep -q "table rc=0" $L/chain_00s_smoke_score.log && grep -q "panels rc=0" $L/chain_00s_smoke_score.log || { echo "smoke table/panels failed -- fix before S1 scoring"; exit 4; }
bash $L/score_stage_v1.sh regime 0301 0204 0052 0147
CUDA_VISIBLE_DEVICES= $PY $L/table_v1.py $O/score_regime P4 $L/TABLE_S1_REGIME.txt $L/TABLE_S1_REGIME.json > $L/table_s1.log 2>&1
echo "table rc=$?"
CUDA_VISIBLE_DEVICES= $PY $L/make_panels_v1.py $O/score_regime $L/TABLE_S1_REGIME.json $O/panels 0301 0204 0052 0147 > $O/panels_s1.log 2>&1
echo "panels rc=$?"
echo "CHAIN01 done $(date +%F_%T)"
