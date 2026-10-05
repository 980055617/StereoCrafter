#!/bin/bash
# S2 continuation after chain_02c stopped at its K2 gate (PREREG_ADDENDUM_3: the A3 zero-hole check assumed bit-exact
# re-encodes; corrected gate A3' in check_k2_v2.py).  Same steps as chain_02c from K2 on: K2 v2 -> stage prep + K3 ->
# M2 (CPU, background) -> hold C (M1 + M1r) -> hold D (M3) -> P12 table -> ADDENDUM 2/3 report lines.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
LOCK=/tmp/claude-gpu0.lock
CL="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
until grep -qE "CHAIN02 done|stopping" $L/chain_02c_ext12_split.log; do sleep 30; done
echo "CHAIN02d start $(date +%F_%T) after: $(tail -1 $L/chain_02c_ext12_split.log)"
[ -e $O/score_ext12 ] && { echo "score_ext12 already exists (chain_02c went further than expected) -- not touching it"; exit 5; }
NP=$(grep -chE '^[0-9]{4} ' $L/jobs_02a_ext12.txt $L/jobs_02b_ext12.txt | paste -sd+ | bc)
EXP=$((32 + NP))
CUDA_VISIBLE_DEVICES= $PY $L/check_k2_v2.py $L/CHECK_K2_ext12_v2.txt $CL | tail -1
grep -q "SUMMARY checked=$EXP failed=0" $L/CHECK_K2_ext12_v2.txt || { echo "K2 v2 ext12 FAIL or incomplete (expected $EXP) -- stopping before scoring"; exit 3; }
CUDA_VISIBLE_DEVICES= $PY $L/diff_0225_v1.py $L/DIFF_0225_sd1fill_vs_sd1.txt
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
CUDA_VISIBLE_DEVICES= $PY $L/p12_sensitivity_v1.py $D $L/TABLE_S2_EXT12.json $L/TABLE_S2_SENSITIVITY.txt > $L/sens_s2.log 2>&1
echo "sensitivity rc=$?"
echo "CHAIN02 done $(date +%F_%T)"
