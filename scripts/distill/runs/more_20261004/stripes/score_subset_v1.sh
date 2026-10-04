#!/bin/bash
# Score an explicit spec list (clip:label=path per line, grouped by clip, origin_ll FIRST in every clip group).
# usage: score_subset_v1.sh <tag> <specs_file>   [env TEMPORAL=1 to add the RAFT temporal metrics]
#   LPIPS  -> outputs/more_20261004/stripes/SCORES_<tag>.txt   (score_clip_ll.py unchanged, SCORE_STEP=4, GPU-0 lock)
#   decomp -> outputs/more_20261004/stripes/decomp_<tag>/       (CPU)
#   temporal -> outputs/more_20261004/stripes/temporal_<tag>/   (score_temporal_ll.py unchanged, GPU-0 lock)
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/more_20261004/stripes
O=outputs/more_20261004/stripes
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
TAG=$1; SPECS=$2
SL=$O/lists_${TAG}_scorelist.txt
[ -e $SL ] && { echo "refusing to overwrite $SL"; exit 3; }
sed -E 's/^([0-9]+):[^=]+=/\1=/' $SPECS > $SL
cp $SPECS $O/lists_${TAG}_specs.txt
$L/score_lpips_v1.sh $O/SCORES_$TAG.txt $SL
echo "LPIPS done $(tail -1 $O/SCORES_$TAG.txt)"
mkdir -p $O/decomp_$TAG
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 taskset -c 24-31 nice -n 10 $PY $L/run_decomp_v1.py $O/decomp_$TAG/decomp.json \
  $O/decomp_$TAG/DECOMP.txt $(cat $SPECS | tr '\n' ' ') > $O/decomp_$TAG/run.log 2>&1
echo "decomp done rc=$?"
if [ "${TEMPORAL:-0}" = "1" ]; then
  mkdir -p $O/temporal_$TAG
  [ -e $O/temporal_$TAG/temporal.json ] && { echo "refusing to overwrite temporal_$TAG"; exit 4; }
  CUDA_VISIBLE_DEVICES=0 flock /tmp/claude-gpu0.lock $PY scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py \
    $O/temporal_$TAG/temporal.json $(cat $SL | tr '\n' ' ') > $O/temporal_$TAG/temporal.log 2>&1
  echo "temporal done rc=$?"
fi
