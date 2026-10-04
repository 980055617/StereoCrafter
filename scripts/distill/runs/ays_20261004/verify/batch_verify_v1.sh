#!/bin/bash
# ays_20261004 / verify -- the lane's ONLY GPU work (PREREG.txt V1, V2, V4), GPU 1.
# Must run INSIDE one outer `flock /tmp/claude-gpu1.lock` (launched by launch_batch_verify_v1.sh as a transient systemd
# user unit); it aborts if that lock is not held.  No inner flock.  The scorers are run UNCHANGED from their own paths.
# Never overwrites: refuses existing score files / JSONs / output dirs.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/ays_20261004/verify
O=outputs/ays_20261004/verify
LOG=$L/batch_verify_v1.log
S_U=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
S_R=scripts/distill/runs/more_20261004/eval_robustness/score_registered_v1.py
S_T=scripts/distill/runs/finalcheck_20261004/validate/score_temporal_ll.py
export CUDA_VISIBLE_DEVICES=1
echo "BATCH_START $(date '+%F_%T') host=$(hostname) python=$(which python) VERIFY_LOCK=${VERIFY_LOCK:-unset}" >> $LOG
if [ "${VERIFY_LOCK:-}" != "gpu1" ]; then echo "ABORT not launched by the outer-lock launcher" >> $LOG; exit 2; fi
if flock -n /tmp/claude-gpu1.lock true; then echo "ABORT /tmp/claude-gpu1.lock is NOT held" >> $LOG; exit 2; fi
echo "LOCK_HELD_CHECK ok (non-blocking flock attempt refused) $(date '+%F_%T')" >> $LOG
md5sum -c >> $LOG 2>&1 <<EOF || { echo "ABORT scorer md5 guard failed" >> $LOG; exit 3; }
a6695eb066971e1ca211e4ad9c85fac7  $S_U
c9532d5a5cde42f7024a2caebb855c3f  $S_R
24bf1bcbdf11b454038f89bd96f70701  $S_T
EOF
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $LOG 2>&1
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG 2>&1

# ------------------------------------------------------------------ V1 unregistered, 12 clips, one call per clip
while read -r C SPECS; do
  [ -z "$C" ] && continue
  OUTF=$L/SCORES_V1_$C.txt
  if [ -e "$OUTF" ]; then echo "V1 SKIP $C: $OUTF exists (never overwritten)" >> $LOG; continue; fi
  echo "SCORE_START $(date '+%F_%T') scorer=$S_U SCORE_STEP=4 gpu=1 stage=V1" > $OUTF
  SCORE_STEP=4 python $S_U $SPECS >> $OUTF 2>&1 < /dev/null
  RC=$?
  echo "SCORE_DONE rc=$RC $(date '+%F_%T')" >> $OUTF
  echo "V1 $C rc=$RC rows=$(grep -c '^ROW' $OUTF) $(date '+%F_%T')" >> $LOG
done < $L/specs_V1.txt
echo "V1_END $(date '+%F_%T')" >> $LOG

# ------------------------------------------------------------------ V2 registered, clips 0042 0259
O2=$O/score_reg_v1
if [ -e "$O2" ]; then
  echo "V2 SKIPPED: $O2 exists (never overwritten)" >> $LOG
else
  mkdir -p "$O2"
  for C in 0042 0259; do
    SCORE_STEP=4 python $S_R "$O2" $L/ROWS_reg_v1.json "$C" >> $L/score_reg_v1.log 2>&1 < /dev/null
    echo "V2 $C rc=$? $(date '+%F_%T')" >> $LOG
  done
fi
echo "V2_END $(date '+%F_%T')" >> $LOG

# ------------------------------------------------------------------ V4 temporal (every frame, STEP unset), origin first
OT=$O/temporal_v1
mkdir -p "$OT"
while read -r C SPECS; do
  [ -z "$C" ] && continue
  OUTJ=$OT/$C.json
  if [ -e "$OUTJ" ] || [ -e "$OT/$C.log" ]; then echo "V4 SKIP $C: output exists (never overwritten)" >> $LOG; continue; fi
  echo "V4_START $C $(date '+%F_%T')" >> $LOG
  env -u STEP -u NOFLOW -u GT_DIR python $S_T "$OUTJ" $SPECS > $OT/$C.log 2>&1 < /dev/null
  RC=$?
  RAFT=$(grep -c "^\[temporal\] RAFT loaded" $OT/$C.log)
  echo "V4 $C rc=$RC raft_loaded=$RAFT $(date '+%F_%T')" >> $LOG
done < $L/specs_V4.txt
echo "BATCH_DONE $(date '+%F_%T')" >> $LOG
