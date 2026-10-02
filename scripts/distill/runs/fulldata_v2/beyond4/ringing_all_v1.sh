#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad/ll
B=scripts/distill/runs/fulldata_v2/beyond4
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until grep -q ALL_DONE outputs/beyond4_lossless/timing_gpu0.txt 2>/dev/null \
   || ! systemctl --user is-active ll_gpu0_lane >/dev/null 2>&1; do sleep 30; done
{
for spec in "0042 -12 -12" "0052 -12 -12" "0125 -12 -12" "0128 -12 -12" "0141 -12 -12" "0147 -12 -12" \
            "0170 -28 0" "0204 -28 0" "0225 -28 0" "0251 -28 0" "0259 -28 0" "0301 -28 0"; do
  set -- $spec; C=$1; DY=$2; DX=$3
  A=""
  for K in origin g125; do
    F=outputs/beyond4_lossless/clips/${C}_${K}_ll/${C}_inpainting_results_sbs.mkv
    [ "$K" = origin ] && F=outputs/beyond4_lossless/clips/${C}_origin_ll/${C}_inpainting_results_sbs.mkv
    [ -f "$F" ] && A="$A ${K}=$F"
  done
  [ -n "$A" ] && SCORE_STEP=8 $PY $S/ringing_metrics.py $C $DY $DX $A 2>&1 | grep -viE "^/home/kawa|futurewarning|@torch"
done
} > $B/RINGING_12CLIP.txt 2>&1
echo "RINGING_ALL_DONE $(date +%H:%M:%S)"
