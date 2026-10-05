#!/bin/bash
# deep_20261004 / sdedit lane: NO-LOCK job runner v2 = run_driver_sdedit_v1.sh with the per-job flock REMOVED.  It must
# only be called by run_driver_sdedit_v2.sh, which holds /tmp/claude-gpu<GPU>.lock around a whole GROUP of jobs (the 4 SDEdit
# variants of one clip x model), so the lock is taken once per group instead of once per render.  Every render is still its
# own python process with the identical hook, env and arguments as v1.  Original v1 header follows.
# deep_20261004 / sdedit lane driver.  COPY of scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh
# (kept verbatim next to this file as run_driver_speed_v1_ORIG_COPY.sh, md5 924b1c255fb1cd7e0be372d4c5a60d4d), changed
# ONLY as follows (the flock-per-job pattern is the one of more_20261004/stripes/run_driver_stripes_v2.sh):
#   - OUT=outputs/deep_20261004/sdedit ; RUN = this lane's hook infer_ll_hook_sdedit_v1.py
#   - SIGMAS column accepts the aliases s31 | s7 | s1 = the tail of the deployed default-8 Karras grid starting at
#     30.993608474731445 | 7.276163101196289 | 1.1675708293914795 (exact repr strings of the float32 grid values)
#   - one extra job column SDMODE (off|warp|warpfill) -> SD_MODE
#   - every python job runs under   flock /tmp/claude-gpu<GPU>.lock   (lock held PER JOB)
#   - the origin_cli mode is dropped (not used here)
# usage: run_driver_sdedit_v1.sh <jobfile> <gpu>
# jobfile lines:  CLIP LABEL MODEL GUID SIGMAS PAD SDMODE
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/deep_20261004/sdedit/infer_ll_hook_sdedit_v1.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
OUT=outputs/deep_20261004/sdedit
JOBS=$1; GPU=$2
LOCK=/tmp/claude-gpu$GPU.lock
LOG=$OUT/timing_gpu$GPU.txt
S31="30.993608474731445,7.276163101196289,1.1675708293914795,0.09738767892122269,0.0020000000949949026"
S7="7.276163101196289,1.1675708293914795,0.09738767892122269,0.0020000000949949026"
S1="1.1675708293914795,0.09738767892122269,0.0020000000949949026"
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=${SK_GUID:-1.01}
GDEF=$SK_GUID
export SK_STEPS=${SK_STEPS:-8}
echo "GROUP_START gpu$GPU jobs=$JOBS (outer lock $LOCK) $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG PAD SDMODE REST; do
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
  case "${SDMODE:-}" in off|warp|warpfill) ;; *) echo "BADSDMODE $CLIP $LABEL '${SDMODE:-}'" | tee -a $LOG; continue ;; esac
  case "$SIG" in
    -) export SK_SIGMAS='' ;;
    s31) export SK_SIGMAS=$S31 ;;
    s7) export SK_SIGMAS=$S7 ;;
    s1) export SK_SIGMAS=$S1 ;;
    *) export SK_SIGMAS=$SIG ;;
  esac
  if [ "$PAD" = "-" ]; then export SK_RNG_PAD_TO=''; else export SK_RNG_PAD_TO=$PAD; fi
  export SK_GUID=$GG SK_CK='' SK_CLIP=$CLIP SK_OUT=$OD SD_MODE=$SDMODE
  mkdir -p $OD
  T0=$(date +%s.%N)
  echo "LOCKED_AT $(date +%s.%N) (group lock held by run_driver_sdedit_v2.sh)" > $OD.lockinfo
  $PY $RUN > $OD.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  TL=$(cut -d' ' -f2 $OD.lockinfo 2>/dev/null)
  SECS=$(awk -v a=${TL:-$T0} -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  WAIT=$(awk -v a=$T0 -v b=${TL:-$T0} 'BEGIN{printf "%.1f", b-a}')
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  SD=$(grep -oE '^\[sdedit\] windows=.*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  echo "RUN $CLIP $LABEL model=$MODEL guid=$GG sigmas=${SIG} pad=${PAD} sdmode=$SDMODE rc=$RC secs=$SECS lockwait=$WAIT md5=$MD5 $(date +%H:%M:%S) dir=$OD :: $SP :: $SD" \
    | tee -a $LOG
done < "$JOBS"
echo "GROUP_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
