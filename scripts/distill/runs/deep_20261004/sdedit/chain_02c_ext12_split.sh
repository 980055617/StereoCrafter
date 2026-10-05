#!/bin/bash
# S2 (PREREG.txt) with shorter lock holds (replaces chain_02b, stopped while it only WAITED for the lock):
#   hold A: 28 renders (clips 0042 0125 0128 0141)      hold B: 28 renders (clips 0170 0225 0251 0259)
#   K2 on all 12 clips (CPU) -> stage prep + K3 (CPU) -> M2 decomposition in the background (CPU)
#   hold C: M1 + M1r on 12 clips (score_gpu_steps_v1.sh with SKIP_TEMPORAL=1)     hold D: M3 temporal on 12 clips
#   -> P12 table.  Every render / scorer call is the same as in S1.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
LOCK=/tmp/claude-gpu0.lock
CL="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
echo "CHAIN02c start $(date +%F_%T)"
flock $LOCK bash $L/batch_s2_render_part_v1.sh $L/jobs_02a_ext12.txt
echo "hold A rc=$? $(date +%F_%T)"
flock $LOCK bash $L/batch_s2_render_part_v1.sh $L/jobs_02b_ext12.txt
echo "hold B rc=$? $(date +%F_%T)"
NP=$(grep -chE '^[0-9]{4} ' $L/jobs_02a_ext12.txt $L/jobs_02b_ext12.txt | paste -sd+ | bc)
EXP=$((32 + NP))
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v1.py $L/CHECK_K2_ext12.txt $CL | tail -1
grep -q "SUMMARY checked=$EXP failed=0" $L/CHECK_K2_ext12.txt || { echo "K2 ext12 FAIL or incomplete (expected $EXP) -- stopping before scoring"; exit 3; }
bash $L/score_stage_prep_v1.sh ext12 $CL
D=$O/score_ext12
grep -q "PREP done" $D/stage.log || { echo "stage prep failed"; exit 4; }
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 $PY $L/run_decomp_sdedit_v1.py $D/decomp/decomp.json $D/decomp/DECOMP.txt \
  --input $(cat $D/specs.txt | tr '\n' ' ') > $D/decomp/run.log 2>&1 &
M2PID=$!
SKIP_TEMPORAL=1 flock $LOCK bash $L/score_gpu_steps_v1.sh $D $CL
echo "hold C rc=$? $(date +%F_%T)" | tee -a $D/stage.log
flock $LOCK bash $L/score_gpu_temporal_v1.sh $D $CL
echo "hold D rc=$? $(date +%F_%T)" | tee -a $D/stage.log
wait $M2PID
echo "M2 rc=$?" | tee -a $D/stage.log
echo "STAGE ext12 done $(date +%F_%T)" | tee -a $D/stage.log
CUDA_VISIBLE_DEVICES= $PY $L/table_v1.py $D P12 $L/TABLE_S2_EXT12.txt $L/TABLE_S2_EXT12.json > $L/table_s2.log 2>&1
echo "table rc=$?"
echo "CHAIN02 done $(date +%F_%T)"
