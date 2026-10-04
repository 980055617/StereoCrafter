#!/bin/bash
# Wait for the chain's smoke-integrity gate, then score the 0301 smoke (LPIPS under the GPU-0 lock; decomposition
# and visual strips on CPU).  Writes only new files under outputs/more_20261004/stripes/.
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/stripes
O=outputs/more_20261004/stripes
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until grep -q "smoke integrity" $O/chain.log; do sleep 10; done
grep -q "smoke integrity PASS" $O/chain.log || { echo "smoke integrity not PASS -- not scoring"; exit 1; }
$L/score_chain_v1.sh smoke0301 0301
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 taskset -c 24-31 nice -n 10 $PY $L/strips_v1.py $O/strips_smoke0301 > $O/strips_smoke0301.log 2>&1
echo "smoke scoring done"
