#!/bin/bash
# scale_gt dev REGISTERED scoring: one score_registered_scalegt_v1.py call per dev clip with ALL labels given (registration is
# computed once per clip, identical for every label), then the dev analysis / selection (analyze_v1.py dev).
# usage: chain_devreg_v1.sh <tag> <label,label,...>      (first label must be origin_s8; UNREG score files must exist)
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
TAG=$1; LABS=$2
export CUDA_VISIBLE_DEVICES=1
DEV="0040 0082 0091 0184 0245 0268"
LOGS="$L/score_dev_origin_s8.txt"
for lab in ${LABS//,/ }; do
  [ "$lab" = origin_s8 ] && continue
  RS=$(echo $lab | sed -E 's/^(.*)_s([0-9]+)_s8$/\1_s\2/')
  LOGS="$LOGS $L/score_dev_${RS}.txt"
done
ROWS=$L/rows_dev_${TAG}.json
python $L/make_rows_v1.py $ROWS $LABS $LOGS || exit 1
O=outputs/deep_20261004/scale_gt/score_reg_dev_${TAG}
[ -e $O ] && { echo "REFUSING: $O exists"; exit 1; }
mkdir -p $O
for c in $DEV; do
  SCORE_STEP=4 flock /tmp/claude-gpu1.lock python $L/score_registered_scalegt_v1.py $O $ROWS $c >> $L/score_reg_dev_${TAG}.log 2>&1 < /dev/null
  echo "REGSCORE dev $c rc=$? $(date +%F_%T)"
done
python $L/analyze_v1.py dev $O $L/TABLE_DEV_${TAG}.txt $LOGS
echo "DEVREG_DONE $TAG $(date +%F_%T)"
