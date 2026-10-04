#!/bin/bash
# H4: wait until the GPU's chain has finished (CHAIN0_DONE / CHAIN1_DONE), then render origin T5@1.00 at that GPU's resolution.
set -u
cd /home/kawa/master_project/StereoCrafter
I=scripts/distill/runs/finalcheck_20261004/independent
O=outputs/finalcheck_20261004/independent
GPU=$1; R=$2; TAG=$3
N=0; until grep -q "^${TAG}" $O/chain_gpu${GPU}.log 2>/dev/null; do sleep 20; N=$((N+1)); [ $N -gt 540 ] && { echo "H4 gpu$GPU gave up waiting" >> $O/chain_gpu${GPU}.log; exit 6; }; done
$I/run_driver_hres_v1.sh $I/jobs_H4_${R}.txt $GPU
echo "H4_DONE gpu$GPU $(date +%F_%T)" >> $O/chain_gpu${GPU}.log
