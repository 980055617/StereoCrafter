#!/bin/bash
# T1: TRACKED scripts/distill/score_temporal.py (unchanged), STEP unset, GPU 1, one process per clip (0259, 0225).
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
I=scripts/distill/runs/finalcheck_20261004/independent
O=outputs/finalcheck_20261004/independent/t1_temporal_tracked
source $I/paths.sh
mkdir -p $O
export CUDA_VISIBLE_DEVICES=1
unset STEP NOFLOW GT_DIR
echo "T1_START $(date '+%F %T') md5(score_temporal.py)=$(md5sum scripts/distill/score_temporal.py | cut -c1-32)" | tee -a $O/t1_run.log
for c in 0259 0225; do
  J=$O/tracked_${c}.json
  [ -e $J ] && { echo "exists $J, skipped" | tee -a $O/t1_run.log; continue; }
  ARGS=""
  for m in origin_ll mamba_ll deliv s25_ll deliv_g100_s8 deliv_g100_T5pad deliv_g100_T5nat; do
    p=$(path_of $c $m); [ -f "$p" ] || { echo "MISSING $p" | tee -a $O/t1_run.log; exit 2; }; ARGS="$ARGS $c=$p"; done
  /usr/bin/time -v $PY scripts/distill/score_temporal.py $J $ARGS > $O/tracked_${c}.log 2>&1
  echo "T1 $c rc=$? $(date '+%F %T')" | tee -a $O/t1_run.log
done
echo "T1_DONE $(date '+%F %T')" | tee -a $O/t1_run.log
