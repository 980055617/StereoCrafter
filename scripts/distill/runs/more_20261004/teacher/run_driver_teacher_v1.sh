#!/bin/bash
# more_20261004 / teacher lane driver (GPU 1 only).  Derived from
# scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh, changed as follows:
#   - OUT=outputs/more_20261004/teacher ; runs infer_ll_hook_teacher_v1.py (this directory)
#   - MODEL is always plain ORIGIN (no unet state, MAMBA_SELF_ATTN_INCLUDE='__nomatch__', other MAMBA_* unset --
#     exactly the speed driver's `origin` branch = the beyond4 origin_ll env); guidance 1.01 (deployed, CFG active)
#   - every python process runs under `flock /tmp/claude-gpu1.lock` with CUDA_VISIBLE_DEVICES=1 (lock held per job)
#   - secs= is measured from lock acquisition (lock wait reported separately as wait=)
# usage: run_driver_teacher_v1.sh <jobfile>
# jobfile lines:  CLIP LABEL SAMPLER N PAD SCHURN MAXCH      ('-' = unset; SAMPLER '-' = no scheduler swap)
# euler_churn uses EDM's ImageNet-64 window/noise: S_tmin 0.05, S_tmax 50, S_noise 1.003 (fixed here).
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/more_20261004/teacher/infer_ll_hook_teacher_v1.py
OUT=outputs/more_20261004/teacher
JOBS=$1
LOG=$OUT/timing_gpu1.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=1.01
unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
export MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
export SK_UNET=''
echo "LANE_START gpu1 jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL SAMP NN PAD SCH MAXCH REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/clips/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  if [ -e "$OD" ]; then echo "SKIP $CLIP $LABEL dir exists without an sbs.mkv (failed run?) -- not overwritten" | tee -a $LOG; continue; fi
  if [ "$SAMP" = "-" ]; then export SK_SAMPLER=''; else export SK_SAMPLER=$SAMP; fi
  export SK_N=$NN
  if [ "$PAD" = "-" ]; then export SK_PAD=''; else export SK_PAD=$PAD; fi
  if [ "$SCH" = "-" ]; then export SK_SCHURN=0 SK_STMIN=0 SK_STMAX=inf SK_SNOISE=1; else export SK_SCHURN=$SCH SK_STMIN=0.05 SK_STMAX=50 SK_SNOISE=1.003; fi
  if [ "$MAXCH" = "-" ]; then export SK_MAXCHUNKS=''; else export SK_MAXCHUNKS=$MAXCH; fi
  export SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  T0=$(date +%s.%N)
  flock /tmp/claude-gpu1.lock bash -c "date +%s.%N > $OD.t_lock; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' | tr -d ' ' > $OD.gpus_at_start; exec $PY $RUN" > $OD.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  TL=$(cat $OD.t_lock 2>/dev/null || echo $T0)
  SECS=$(awk -v a=$TL -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  WAIT=$(awk -v a=$T0 -v b=$TL 'BEGIN{printf "%.1f", b-a}')
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  echo "RUN $CLIP $LABEL sampler=$SAMP N=$NN pad=$PAD schurn=$SCH maxch=$MAXCH guid=$SK_GUID rc=$RC secs=$SECS wait=$WAIT md5=$MD5 gpus=$(cat $OD.gpus_at_start 2>/dev/null) $(date +%H:%M:%S) dir=$OD :: $SP" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu1 jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
