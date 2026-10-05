#!/bin/bash
# S1 batch: run ONLY as   flock /tmp/claude-gpu0.lock bash scripts/distill/runs/deep_20261004/sdedit/batch_s1_v1.sh
# One lock acquisition (~55 min) for: the 12 deliv regime renders (run_jobs_nolock_v2.sh, identical per-render processes),
# K2 on all 32 regime renders (CPU), then the S1 stage scoring (score_stage_nolock_v1.sh: K3, M2 in the background, M1, M1r,
# M3).  Table and panels run after the lock is released (chain_01c).
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
echo "BATCH_S1 locked $(date +%F_%T)"
grep -q "SUMMARY checked=8 failed=0" $L/CHECK_K2_smoke0301.txt || { echo "S0 K2 not 8/8 PASS -- abort"; exit 2; }
grep -q "SUMMARY checked=10 failed=0" $O/score_smoke0301/K3_passthrough.txt || { echo "S0 K3 not 10/10 PASS -- abort"; exit 2; }
bash $L/run_jobs_nolock_v2.sh $L/jobs_01b_regime_deliv.txt 0
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v1.py $L/CHECK_K2_regime.txt 0301 0204 0052 0147 | tail -1
grep -q "SUMMARY checked=32 failed=0" $L/CHECK_K2_regime.txt || { echo "K2 regime FAIL or incomplete -- stopping before scoring"; exit 3; }
bash $L/score_stage_nolock_v1.sh regime 0301 0204 0052 0147
echo "BATCH_S1 done $(date +%F_%T)"
