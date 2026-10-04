#!/bin/bash
# chain C: waits for chain A v2 -> window-0 convergence diagnostic (jobs_02_diag) -> chain B (Stage 1a).
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/more_20261004/teacher
echo "CHAIN_C_START $(date +%F_%T) waiting for chain A"
until grep -q CHAIN_A_DONE $D/chain_A_v2.log; do sleep 10; done
bash $D/run_driver_teacher_v1.sh $D/jobs_02_diag.txt
echo "DIAG_DONE $(date +%F_%T)"
bash $D/chain_B_v1.sh
echo "CHAIN_C_DONE $(date +%F_%T)"
