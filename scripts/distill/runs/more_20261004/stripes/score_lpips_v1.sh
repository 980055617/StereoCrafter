#!/bin/bash
# COPY of scripts/distill/runs/finalcheck_20261004/speed/score_v1.sh, changed only to hold the GPU-0 lock for the
# scorer call.  Scores a list of clip=path specs with the UNCHANGED lossless scorer (SCORE_STEP=4) on GPU 0.
# usage: score_lpips_v1.sh <out_scores.txt> <list file: one clip=path per line, grouped by clip>
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4
OUTF=$1; LIST=$2
[ -e "$OUTF" ] && { echo "refusing to overwrite $OUTF"; exit 3; }
for p in $(grep -v '^#' $LIST | cut -d= -f2); do [ -f "$p" ] || { echo "MISSING $p" | tee -a $OUTF; exit 4; }; done
echo "SCORE_START $(date +%F_%T) list=$LIST scorer=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py SCORE_STEP=$SCORE_STEP" > $OUTF
flock /tmp/claude-gpu0.lock $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $(grep -v '^#' $LIST | tr '\n' ' ') >> $OUTF 2>&1
echo "SCORE_DONE rc=$? $(date +%F_%T)" >> $OUTF
