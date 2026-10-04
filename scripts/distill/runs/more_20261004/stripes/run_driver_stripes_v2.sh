#!/bin/bash
# more_20261004 / stripes lane driver v2 = run_driver_stripes_v1.sh with RUN = hook v2 and a 9th job column ROUTE
# (both|vae_only|clip_only -> STRIPE_ROUTE).  Original v1 header follows.
# more_20261004 / stripes lane driver.  COPY of scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh
# (kept verbatim next to this file as run_driver_speed_v1_ORIG_COPY.sh), changed ONLY as follows:
#   - OUT=outputs/more_20261004/stripes ; RUN = this lane's hook infer_ll_hook_stripes_v1.py
#   - two extra job columns FILL (none|rowlin|telea) and MASK (keep|shrink) -> STRIPE_FILL / STRIPE_MASK
#   - every python job runs under   flock /tmp/claude-gpu<GPU>.lock   (the lock is held PER JOB, not per lane)
#   - the origin_cli mode is dropped (not used here)
# usage: run_driver_stripes_v1.sh <jobfile> <gpu>
# jobfile lines:  CLIP LABEL MODEL GUID SIGMAS PAD FILL MASK
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/more_20261004/stripes/infer_ll_hook_stripes_v2.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
OUT=outputs/more_20261004/stripes/mechanism
JOBS=$1; GPU=$2
LOCK=/tmp/claude-gpu$GPU.lock
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=${SK_GUID:-1.01}
GDEF=$SK_GUID
export SK_STEPS=${SK_STEPS:-8}
echo "LANE_START gpu$GPU jobs=$JOBS lock=$LOCK $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG PAD FILL MASK ROUTE REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/clips/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  if [ -e "$OD" ]; then echo "SKIP $CLIP $LABEL dir exists without an sbs.mkv (failed run?) -- not overwritten" | tee -a $LOG; continue; fi
  if [ "$G" = "-" ]; then GG=$GDEF; else GG=$G; fi
  case "$MODEL" in
    deliv)
      export MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*' MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
      export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
      export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
      export SK_UNET=$DELIV ;;
    origin)
      unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
      export MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
      export SK_UNET='' ;;
    *) echo "BADMODEL $CLIP $LABEL $MODEL" | tee -a $LOG; continue ;;
  esac
  case "${FILL:-}" in none|rowlin|telea) ;; *) echo "BADFILL $CLIP $LABEL '${FILL:-}'" | tee -a $LOG; continue ;; esac
  case "${MASK:-}" in keep|shrink) ;; *) echo "BADMASK $CLIP $LABEL '${MASK:-}'" | tee -a $LOG; continue ;; esac
  case "${ROUTE:-}" in both|vae_only|clip_only) ;; *) echo "BADROUTE $CLIP $LABEL '${ROUTE:-}'" | tee -a $LOG; continue ;; esac
  if [ "$SIG" = "-" ]; then export SK_SIGMAS=''; else export SK_SIGMAS=$SIG; fi
  if [ "$PAD" = "-" ]; then export SK_RNG_PAD_TO=''; else export SK_RNG_PAD_TO=$PAD; fi
  export SK_GUID=$GG SK_CK='' SK_CLIP=$CLIP SK_OUT=$OD STRIPE_FILL=$FILL STRIPE_MASK=$MASK STRIPE_ROUTE=$ROUTE
  mkdir -p $OD
  T0=$(date +%s.%N)
  flock $LOCK bash -c "echo LOCKED_AT \$(date +%s.%N) > $OD.lockinfo; $PY $RUN > $OD.log 2>&1"
  RC=$?
  T1=$(date +%s.%N)
  TL=$(cut -d' ' -f2 $OD.lockinfo 2>/dev/null)
  SECS=$(awk -v a=${TL:-$T0} -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  WAIT=$(awk -v a=$T0 -v b=${TL:-$T0} 'BEGIN{printf "%.1f", b-a}')
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  ST=$(grep -oE '^\[stripes\] window .*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  echo "RUN $CLIP $LABEL model=$MODEL guid=$GG sigmas=${SIG} pad=${PAD} fill=$FILL mask=$MASK route=$ROUTE rc=$RC secs=$SECS lockwait=$WAIT md5=$MD5 $(date +%H:%M:%S) dir=$OD :: $SP :: $ST" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
