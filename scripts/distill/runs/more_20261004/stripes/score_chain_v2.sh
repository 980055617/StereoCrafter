#!/bin/bash
# v2 = score_chain_v1.sh using make_lists_v2.sh (no C labels).
# Stripes lane scoring chain (after the renders exist).  GPU steps hold /tmp/claude-gpu0.lock per step.
# usage: score_chain_v1.sh <tag> <clip> [<clip> ...]
#   LPIPS  -> outputs/more_20261004/stripes/SCORES_<tag>.txt          (score_clip_ll.py, SCORE_STEP=4)
#   decomp -> outputs/more_20261004/stripes/decomp_<tag>/             (CPU)
#   temporal (only if TEMPORAL=1) -> outputs/more_20261004/stripes/temporal_<tag>/temporal.json
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/stripes
O=outputs/more_20261004/stripes
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
TAG=$1; shift
PRE=$O/lists_$TAG
$L/make_lists_v2.sh $PRE "$@" || exit 3
$L/score_lpips_v1.sh $O/SCORES_$TAG.txt ${PRE}_scorelist.txt
echo "LPIPS done $(tail -1 $O/SCORES_$TAG.txt)"
if [ "${TEMPORAL:-0}" = "1" ]; then
  mkdir -p $O/temporal_$TAG
  [ -e $O/temporal_$TAG/temporal.json ] && { echo "refusing to overwrite temporal_$TAG"; exit 4; }
  CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock $PY scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py \
    $O/temporal_$TAG/temporal.json $(cat ${PRE}_scorelist.txt | tr '\n' ' ') > $O/temporal_$TAG/temporal.log 2>&1
  echo "temporal done rc=$?"
fi
mkdir -p $O/decomp_$TAG
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 taskset -c 24-31 nice -n 10 $PY $L/run_decomp_v1.py $O/decomp_$TAG/decomp.json \
  $O/decomp_$TAG/DECOMP.txt $(cat ${PRE}_specs.txt | tr '\n' ' ') > $O/decomp_$TAG/run.log 2>&1
echo "decomp done rc=$?"
