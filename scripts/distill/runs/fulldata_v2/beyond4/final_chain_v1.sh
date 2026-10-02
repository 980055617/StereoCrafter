#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad/ll
B=scripts/distill/runs/fulldata_v2/beyond4
until grep -q ALL_DONE outputs/beyond4_lossless/timing_gpu0.txt 2>/dev/null \
   || ! systemctl --user is-active ll_gpu0_lane >/dev/null 2>&1; do sleep 30; done
sleep 5
$S/score_all_v1.sh final
echo "FINAL_CHAIN_DONE $(date +%H:%M:%S)"
