#!/bin/bash
# deep_20261004 / sdedit lane driver v2: groups consecutive job lines with the same (CLIP, MODEL) and runs each group
# under ONE acquisition of /tmp/claude-gpu<GPU>.lock (about 4 renders, ~10-12 min), via run_jobs_nolock_v2.sh.  Each render
# is still its own python process with exactly the v1 hook, env and arguments (K2/K3 verify every render).
# usage: run_driver_sdedit_v2.sh <jobfile> <gpu>        (job line format identical to v1)
set -u
cd /home/kawa/master_project/StereoCrafter
L=scripts/distill/runs/deep_20261004/sdedit
O=outputs/deep_20261004/sdedit
JOBS=$1; GPU=$2
LOCK=/tmp/claude-gpu$GPU.lock
LOG=$O/timing_gpu$GPU.txt
GD=$O/groups/$(basename $JOBS .txt)_$(date +%Y%m%d_%H%M%S)
mkdir -p $GD
echo "LANE_START(v2 grouped) gpu$GPU jobs=$JOBS groups=$GD $(date +%F_%T)" | tee -a $LOG
n=0; prev=""
grep -vE '^\s*(#|$)' $JOBS | while read -r CLIP LABEL MODEL REST; do
  key="${CLIP}_${MODEL}"
  if [ "$key" != "$prev" ]; then n=$((n+1)); prev=$key; fi
  echo "$CLIP $LABEL $MODEL $REST" >> $GD/group_$(printf %02d $n)_$key.txt
done
for g in $GD/group_*.txt; do
  T0=$(date +%s.%N)
  flock $LOCK bash -c "echo GROUP_LOCKED_AT \$(date +%s.%N) $g >> $GD/locks.txt; bash $L/run_jobs_nolock_v2.sh $g $GPU"
  RC=$?
  T1=$(date +%s.%N)
  echo "GROUP $(basename $g) rc=$RC wall=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}') $(date +%H:%M:%S)" | tee -a $LOG
done
echo "LANE_DONE(v2 grouped) gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
