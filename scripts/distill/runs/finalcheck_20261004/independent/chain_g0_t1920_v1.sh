#!/bin/bash
set -u
cd /home/kawa/master_project/StereoCrafter
I=scripts/distill/runs/finalcheck_20261004/independent
O=outputs/finalcheck_20261004/independent
N=0; until grep -q '^H4_DONE gpu0' $O/chain_gpu0.log 2>/dev/null; do sleep 15; N=$((N+1)); [ $N -gt 720 ] && exit 6; done
$I/run_driver_hres_v1.sh $I/jobs_H_1920_timing_g0.txt 0
echo "T1920G0_DONE $(date +%F_%T)" >> $O/chain_gpu0.log
