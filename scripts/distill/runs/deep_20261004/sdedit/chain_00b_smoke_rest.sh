#!/bin/bash
# S0 remainder with the grouped driver v2 (2 lock acquisitions instead of 6), then the "CHAIN00 done" marker that
# chain_00s_smoke_score.sh waits for is appended to chain_00_smoke.log (the stopped v1 chain never wrote it).
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
echo "CHAIN00b start $(date +%F_%T)"
bash $L/run_driver_sdedit_v2.sh $L/jobs_00c_smoke_rest.txt 0
echo "CHAIN00 done $(date +%F_%T) (remainder rendered by chain_00b_smoke_rest.sh with run_driver_sdedit_v2.sh)" >> $L/chain_00_smoke.log
echo "CHAIN00b done $(date +%F_%T)"
