#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad/ll
B=scripts/distill/runs/fulldata_v2/beyond4
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
until [ -f outputs/beyond4_lossless/clips/0259_g125_ll/0259_inpainting_results_sbs.mkv.md5 ] \
   || ! systemctl --user is-active ll_gpu0_lane >/dev/null 2>&1; do sleep 15; done
export CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4
A=""
for C in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  for K in origin_ll g125_ll; do
    F=outputs/beyond4_lossless/clips/${C}_${K}/${C}_inpainting_results_sbs.mkv
    [ -f "$F" ] && A="$A ${C}=$F"; done; done
$PY $S/score_clip_ll.py $A 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/SCORES_g125_12clip.txt
$PY $S/score_dcmatch.py $A 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/DCMATCH_g125_12clip.txt
$PY $S/summarize.py $B/SCORES_g125_12clip.txt > $B/TABLES_g125_12clip.txt 2>&1
echo "G125_12CLIP_DONE $(date +%H:%M:%S)"
