#!/bin/bash
# GPU 1 chain: wait for T1 -> T2 (validate's equivalence-proven temporal copy, 12 clips, T5@1.00 rows) -> wait for H0
# PASS -> H1/H2 renders at 1024x1920
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
I=scripts/distill/runs/finalcheck_20261004/independent
O=outputs/finalcheck_20261004/independent
TS=scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py
LOG=$O/chain_gpu1.log
source $I/paths.sh
echo "CHAIN1_START $(date +%F_%T)" >> $LOG
until grep -q '^T1_DONE' $O/t1_temporal_tracked/t1_run.log 2>/dev/null; do sleep 20; done
echo "T1 seen done $(date +%F_%T)" >> $LOG
mkdir -p $O/t2_temporal
J=$O/t2_temporal/temporal_T5g100_12clip.json
if [ ! -e $J ]; then
  ARGS=""
  for c in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
    for m in origin_ll deliv deliv_g100_s8 deliv_g100_T5pad deliv_g100_T5nat; do
      p=$(path_of $c $m); [ -f "$p" ] || { echo "MISSING $p" >> $LOG; exit 2; }; ARGS="$ARGS $c=$p"; done
  done
  echo "T2_START $(date +%F_%T) md5(score_temporal_ll.py)=$(md5sum $TS | cut -c1-32)" >> $LOG
  ( export CUDA_VISIBLE_DEVICES=1; unset STEP NOFLOW GT_DIR; /usr/bin/time -v $PY $TS $J $ARGS > $O/t2_temporal/temporal_T5g100_12clip.log 2>&1 )
  echo "T2 rc=$? $(date +%F_%T)" >> $LOG
fi
N=0; until [ -f $O/H0_RESULT.txt ]; do sleep 20; N=$((N+1)); [ $N -gt 270 ] && { echo "H0 result never appeared" >> $LOG; exit 6; }; done
grep -q '^H0 PASS' $O/H0_RESULT.txt || { echo "H0 not PASS -> no 1920 renders" >> $LOG; exit 5; }
$I/run_driver_hres_v1.sh $I/jobs_H_1920.txt 1
echo "CHAIN1_DONE $(date +%F_%T)" >> $LOG
