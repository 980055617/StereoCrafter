#!/bin/bash
# GPU-1 chain: J1 re-scores now; J3c scoring when the AYS chain has finished; J5 scoring when the dev renders are done.
set -u
cd /home/kawa/master_project/StereoCrafter
J=scripts/distill/runs/more_20261004/judge
O=outputs/more_20261004/judge
bash $J/score_J1_v1.sh > $O/score_J1.log 2>&1
until grep -q J3C_CHAIN_FINISHED $O/chain_J3c_ays.log 2>/dev/null; do sleep 10; done
if grep -q CONTROL_PASS $O/chain_J3c_ays.log; then bash $J/score_J3c_ays_v1.sh > $O/score_J3c.log 2>&1; else echo "J3c control failed -> no AYS scoring" > $O/score_J3c.log; fi
until grep -q J5_RENDERS_FINISHED $O/chain_J5_dev.log 2>/dev/null; do sleep 10; done
bash $J/score_J5_dev_v1.sh > $O/score_J5.log 2>&1
echo "GPU1_SCORING_CHAIN_DONE $(date +%F_%T)" >> $O/score_J5.log
