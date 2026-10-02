#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad/ll
B=scripts/distill/runs/fulldata_v2/beyond4
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4
ARGS=""
for C in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
  for K in origin_ll g115_ll g125_ll g140_ll s25_ll; do
    F=outputs/beyond4_lossless/clips/${C}_${K}/${C}_inpainting_results_sbs.mkv
    [ -f "$F" ] && ARGS="$ARGS ${C}=$F"; done; done
echo "scoring $(echo $ARGS | wc -w) outputs $(date +%H:%M:%S)"
$PY $S/score_clip_ll.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/SCORES_final.txt
echo "SCORE_RC=$? rows=$(grep -c '^ROW' $B/SCORES_final.txt) $(date +%H:%M:%S)"
$PY $S/score_dcmatch.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/DCMATCH_final.txt
echo "DC_RC=$? $(date +%H:%M:%S)"
$PY $S/summarize.py $B/SCORES_final.txt > $B/TABLES_final.txt 2>&1
echo "FINAL_V2_DONE $(date +%H:%M:%S)"
