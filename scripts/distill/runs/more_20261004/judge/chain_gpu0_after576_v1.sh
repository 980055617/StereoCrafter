#!/bin/bash
# GPU-0 chain after the J2b timing: J3c AYS (control first, gated) -> J5 dev-clip renders.  Lock taken per render.
set -u
cd /home/kawa/master_project/StereoCrafter
J=scripts/distill/runs/more_20261004/judge
O=outputs/more_20261004/judge
bash $J/chain_J3c_ays_v1.sh > $O/chain_J3c_ays.log 2>&1
echo "J3C_CHAIN_FINISHED $(date +%F_%T)" >> $O/chain_J3c_ays.log
bash $J/run_driver_dev_v1.sh $J/jobs_J5_devclips.txt 0 > $O/chain_J5_dev.log 2>&1
echo "J5_RENDERS_FINISHED $(date +%F_%T)" >> $O/chain_J5_dev.log
