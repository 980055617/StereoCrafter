#!/bin/bash
# GPU-0 lane, starts the moment the trainer releases GPU 0: the two speed rows that do not need the
# deliverable (origin, and the shipped 5-slot Mamba with its checkpoint actually loaded).
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/beyond_distil_mamba_scaled
echo "waiting for bdms-chainA (training) to release GPU 0 ... $(date +%T)"
until ! systemctl --user is-active --quiet bdms-chainA; do sleep 30; done
echo "GPU 0 free $(date +%T); 60 s settle before an exclusive-GPU benchmark"
sleep 60
bash $D/bench_deliv_v2.sh 0 $D/bench_totals_v1.txt - origin mamba5ck
echo CHAINC1_DONE $(date +%T)
