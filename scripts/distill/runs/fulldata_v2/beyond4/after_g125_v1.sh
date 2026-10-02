#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad/ll
B=scripts/distill/runs/fulldata_v2/beyond4
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until [ -f outputs/beyond4_lossless/clips/0147_g125_ll/0147_inpainting_results_sbs.mkv.md5 ] \
   || ! systemctl --user is-active ll_gpu0_lane >/dev/null 2>&1; do sleep 20; done
export CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4
A=""
for C in 0052 0147 0204 0301; do for K in origin_ll g125_ll s25_ll; do
  F=outputs/beyond4_lossless/clips/${C}_${K}/${C}_inpainting_results_sbs.mkv; [ -f "$F" ] && A="$A ${C}=$F"; done; done
$PY $S/score_clip_ll.py $A 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/SCORES_g125_4clip.txt
$PY $S/score_dcmatch.py $A 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/DCMATCH_g125_4clip.txt
echo "G125_4CLIP_SCORED $(date +%H:%M:%S)"
# crops + ringing on the three visual-read clips
for spec in "0052 -12 -12" "0147 -12 -12" "0301 -28 0"; do
  set -- $spec; C=$1; DY=$2; DX=$3
  for FI in 40 75 110; do
    $PY $S/make_crops.py $C $DY $DX $FI outputs/beyond4_lossless/crops/${C} \
      "deployed origin 8st g1.01"=outputs/beyond4_lossless/clips/${C}_origin_ll/${C}_inpainting_results_sbs.mkv \
      "g125 8st guid 1.25"=outputs/beyond4_lossless/clips/${C}_g125_ll/${C}_inpainting_results_sbs.mkv \
      2>&1 | grep -viE "^/home/kawa|futurewarning|@torch"
  done
  SCORE_STEP=8 $PY $S/ringing_metrics.py $C $DY $DX \
    origin=outputs/beyond4_lossless/clips/${C}_origin_ll/${C}_inpainting_results_sbs.mkv \
    g125=outputs/beyond4_lossless/clips/${C}_g125_ll/${C}_inpainting_results_sbs.mkv \
    s25=outputs/beyond4_lossless/clips/${C}_s25_ll/${C}_inpainting_results_sbs.mkv \
    2>&1 | grep -viE "^/home/kawa|futurewarning|@torch"
done > $B/CROPS_AND_RINGING.txt 2>&1
echo "CROPS_DONE $(date +%H:%M:%S)"
