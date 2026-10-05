#!/bin/bash
# S2 (PREREG.txt): renders for every (variant, model) that passed P4 at S1 on the 8 non-regime test clips, K2 on all 12 clips,
# 12-clip stage scoring, the P12 table.  Exits cleanly (no GPU use) when nothing passed P4.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN02 start $(date +%F_%T)"
$PY $L/make_jobs_s2_v1.py $L/TABLE_S1_REGIME.json $L/jobs_02_ext12.txt || { echo "make_jobs failed"; exit 2; }
[ -f $L/jobs_02_ext12.txt ] || { echo "nothing passed P4 -> no S2 (lane verdict: closed)"; echo "CHAIN02 done $(date +%F_%T)"; exit 0; }
NP=$(grep -cE '^[0-9]{4} ' $L/jobs_02_ext12.txt)
bash $L/run_driver_sdedit_v2.sh $L/jobs_02_ext12.txt 0
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v1.py $L/CHECK_K2_ext12.txt 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301 | tail -1
EXP=$((32 + NP))
grep -q "SUMMARY checked=$EXP failed=0" $L/CHECK_K2_ext12.txt || { echo "K2 ext12 FAIL or incomplete (expected $EXP) -- stopping before scoring"; exit 3; }
bash $L/score_stage_v1.sh ext12 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301
CUDA_VISIBLE_DEVICES= $PY $L/table_v1.py $O/score_ext12 P12 $L/TABLE_S2_EXT12.txt $L/TABLE_S2_EXT12.json > $L/table_s2.log 2>&1
echo "table rc=$?"
echo "CHAIN02 done $(date +%F_%T)"
