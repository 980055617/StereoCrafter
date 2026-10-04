#!/bin/bash
# Run score_temporal_tr_v1.py on GPU 1 under the GPU-1 lock (held for this one scoring job only).
# usage: score_temporal_run.sh <out.json> <specs file: one clip=path[#ov=N][#dcs=N] per line, grouped by clip, origin first>
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
OUTJ=$1; SPECS=$2; LOG=${OUTJ%.json}.log
[ -e "$OUTJ" ] && { echo "refusing to overwrite $OUTJ"; exit 3; }
[ -e "$LOG" ] && { echo "refusing to overwrite $LOG"; exit 3; }
for p in $(grep -v '^#' $SPECS | cut -d= -f2- | cut -d'#' -f1); do [ -f "$p" ] || { echo "MISSING $p" | tee -a $LOG; exit 4; }; done
export CUDA_VISIBLE_DEVICES=1
unset STEP
echo "TSCORE_START $(date +%F_%T) specs=$SPECS" > $LOG
flock /tmp/claude-gpu1.lock $PY scripts/distill/runs/more_20261004/temporal/score_temporal_tr_v1.py $OUTJ $(grep -v '^#' $SPECS | tr '\n' ' ') >> $LOG 2>&1
echo "TSCORE_DONE rc=$? $(date +%F_%T)" >> $LOG
