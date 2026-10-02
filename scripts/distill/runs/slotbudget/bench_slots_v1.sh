#!/bin/bash
# SLOT BUDGET - PASS B driver: per-slot attn1 attribution with bench_slots_v2.py
# (bench2.py protocol + utils/module_timing CUDA-event hooks on the five level-0 spatial slots).
# Same env as the published bench; reps outer, configs inner; GPU 0 only.
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/slotbudget
OUT=$R/bench_slots_v1.txt
export CUDA_VISIBLE_DEVICES=0
mkdir -p $R/profiles $R/logs
echo "BENCH_SLOTS_V1_START $(date +%F_%H:%M:%S)" | tee -a "$OUT"
for rep in 1 2 3; do
  for V in "origin:__nomatch__" "mamba5:down_blocks.0.*,up_blocks.3.*" "mamba2:down_blocks.0.*"; do
    L=${V%%:*}; INC=${V#*:}
    for RES in "576 1024" "1024 1792" "1024 1920"; do
      set -- $RES; H=$1; W=$2
      TAG=${L}_h${H}w${W}_r${rep}
      H=$H W=$W BS=2 GATE=1.0 DS=128 EXP=1 BIDIR=fwd \
        SB_JSON=$PWD/$R/profiles/${TAG}.json \
        python3 $R/bench_slots_v2.py "$TAG" "$INC" "__nomatch__" > $R/logs/slots_${TAG}.log 2>&1
      grep -E '^RESULT' $R/logs/slots_${TAG}.log | tee -a "$OUT"
    done
  done
  echo "  slots rep $rep done $(date +%H:%M:%S)" | tee -a "$OUT"
done
echo "BENCH_SLOTS_V1_DONE $(date +%F_%H:%M:%S)" | tee -a "$OUT"
