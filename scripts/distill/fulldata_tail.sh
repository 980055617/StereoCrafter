#!/bin/bash
# after BOTH lanes: (1) per-slot noise sensitivity re-run with the private generator (GPU0), (2) exclusive-GPU bench ds128 vs ds32
cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata
until grep -q CAPTURE_LANE_DONE $R/capture_lane.log 2>/dev/null; do sleep 120; done
export CUDA_VISIBLE_DEVICES=0; sed -i 's/^export CUDA_VISIBLE_DEVICES=1/export CUDA_VISIBLE_DEVICES=0/' $D/noise_slot.sh
echo "NOISE_V2_START $(date +%H:%M:%S)"; $D/noise_slot.sh > $D/runs/noise_slot_v2.log 2>&1; grep NOISE $D/runs/noise_slot_v2.log | sed 's/LPIPS\/maskPSNR\/sharp=//'
until grep -q EVAL_LANE_DONE $R/eval_lane.log 2>/dev/null; do sleep 120; done
echo "BENCH_START (exclusive) $(date +%H:%M:%S)"
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad
for rep in 1 2 3; do for V in "origin:__nomatch__:128" "light_ds128:down_blocks.0.*,up_blocks.3.*:128" "light_ds32:down_blocks.0.*,up_blocks.3.*:32"; do
  L=${V%%:*}; rest=${V#*:}; INC=${rest%%:*}; DS=${rest##*:}
  BS=2 GATE=1.0 DS=$DS EXP=1 BIDIR=fwd conda run -n stereocrafter --no-capture-output python3 $D/bench2.py "${L}_r$rep" "$INC" "__nomatch__" 2>&1 | grep -E '^RESULT' | cut -c1-160
done; done | tee $R/bench_tail.txt
echo "TAIL_DONE $(date +%H:%M:%S)"
