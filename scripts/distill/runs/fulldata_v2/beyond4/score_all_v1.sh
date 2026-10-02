#!/bin/bash
# Score every lossless output that exists.  Read-only w.r.t. the video outputs.
set -u; cd /home/kawa/master_project/StereoCrafter
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad/ll
B=scripts/distill/runs/fulldata_v2/beyond4
TAG=${1:-final}
export CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4
CLIPS="0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301"
ARGS=""; DARGS=""
for C in $CLIPS; do
  for K in origin_ll g115_ll g125_ll g140_ll s25_ll; do
    F=outputs/beyond4_lossless/clips/${C}_${K}/${C}_inpainting_results_sbs.mkv
    if [ -f "$F" ]; then ARGS="$ARGS ${C}=$F"; DARGS="$DARGS ${C}=$F"; fi
  done
done
echo "scoring: $(echo $ARGS | wc -w) outputs"
python $S/score_clip_ll.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/SCORES_$TAG.txt
echo "SCORE_RC=$? $(date +%H:%M:%S)"
python $S/score_dcmatch.py $DARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" > $B/DCMATCH_$TAG.txt
echo "DC_RC=$? $(date +%H:%M:%S)"
python $S/summarize.py $B/SCORES_$TAG.txt > $B/TABLES_$TAG.txt 2>&1
echo "SUM_RC=$? $(date +%H:%M:%S)"
tail -5 $B/TABLES_$TAG.txt
