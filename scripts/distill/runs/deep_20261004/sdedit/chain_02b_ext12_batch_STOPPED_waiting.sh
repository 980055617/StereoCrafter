#!/bin/bash
# S2 (PREREG.txt) in batch mode: jobs for every P4 pass on the 8 non-regime test clips -> one lock for the renders + K2 ->
# one lock for the 12-clip stage scoring -> P12 table.  Exits without GPU use when nothing passed P4.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
echo "CHAIN02b start $(date +%F_%T)"
$PY $L/make_jobs_s2_v1.py $L/TABLE_S1_REGIME.json $L/jobs_02_ext12.txt || { echo "make_jobs failed"; exit 2; }
[ -f $L/jobs_02_ext12.txt ] || { echo "nothing passed P4 -> no S2"; echo "CHAIN02 done $(date +%F_%T)"; exit 0; }
NP=$(grep -cE '^[0-9]{4} ' $L/jobs_02_ext12.txt)
flock /tmp/claude-gpu0.lock bash $L/batch_s2_render_v1.sh
echo "render batch rc=$? $(date +%F_%T)"
EXP=$((32 + NP))
grep -q "SUMMARY checked=$EXP failed=0" $L/CHECK_K2_ext12.txt || { echo "K2 ext12 FAIL or incomplete (expected $EXP) -- stopping before scoring"; exit 3; }
flock /tmp/claude-gpu0.lock bash $L/batch_s2_score_v1.sh
echo "score batch rc=$? $(date +%F_%T)"
grep -q "STAGE ext12 done" $O/score_ext12/stage.log || { echo "S2 stage scoring incomplete -- stopping"; exit 4; }
CUDA_VISIBLE_DEVICES= $PY $L/table_v1.py $O/score_ext12 P12 $L/TABLE_S2_EXT12.txt $L/TABLE_S2_EXT12.json > $L/table_s2.log 2>&1
echo "table rc=$?"
echo "CHAIN02 done $(date +%F_%T)"
