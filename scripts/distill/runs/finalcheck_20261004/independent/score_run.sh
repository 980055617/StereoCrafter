#!/bin/bash
# Score a list of clip=path specs with the UNCHANGED lossless scorer (SCORE_STEP=4).
# usage: score_run.sh <out_scores.txt> <list file> <gpu>
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export CUDA_VISIBLE_DEVICES=$3 SCORE_STEP=4
OUTF=$1; LIST=$2
[ -e "$OUTF" ] && { echo "refusing to overwrite $OUTF"; exit 3; }
for p in $(grep -v '^#' $LIST | cut -d= -f2); do [ -f "$p" ] || { echo "MISSING $p" | tee -a $OUTF; exit 4; }; done
echo "SCORE_START $(date +%F_%T) list=$LIST scorer=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=$SCORE_STEP gpu=$3 md5(scorer)=$(md5sum scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py | cut -c1-32)" > $OUTF
/usr/bin/time -v $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $(grep -v '^#' $LIST | tr '\n' ' ') >> $OUTF 2>&1
echo "SCORE_DONE rc=$? $(date +%F_%T)" >> $OUTF
