#!/bin/bash
# scale_gt lane render driver.  Derived from scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh (model=origin
# branch only), running an UNCHANGED copy of its hook (infer_ll_hook_scalegt_v1.py == infer_ll_hook_speed_v1.py, md5 7f5bf0be...):
#   origin weights (no unet state, MAMBA_SELF_ATTN_INCLUDE='__nomatch__', exactly the beyond4 origin_ll env), optional SK_CK =
#   a state dict of the 15 up_blocks.3 attn1 tensors swapped in before the first call, SK_STEPS (8 deployed / 25), SK_GUID 1.01,
#   no custom sigmas, no RNG padding, FFV1 lossless sbs (LOSSLESS_SBS=1).
# EVERY job takes the GPU-1 lock separately (flock /tmp/claude-gpu1.lock), so other lanes interleave between renders.
# usage: run_render_v1.sh <jobfile>      jobfile lines:  CLIP LABEL CK STEPS     (CK '-' = plain origin)
# Every render gets its own NEW directory outputs/deep_20261004/scale_gt/renders/<CLIP>_<LABEL>; an existing one is skipped.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
PY=$(which python)
RUN=scripts/distill/runs/deep_20261004/scale_gt/infer_ll_hook_scalegt_v1.py
OUT=outputs/deep_20261004/scale_gt/renders
JOBS=$1
LOG=scripts/distill/runs/deep_20261004/scale_gt/render_log_v1.txt
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
export MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
export SK_UNET='' SK_SIGMAS='' SK_RNG_PAD_TO='' SK_GUID=1.01
echo "DRIVER_START jobs=$JOBS $(date +%F_%T)" >> $LOG
while read -r CLIP LABEL CK STEPS REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" >> $LOG; continue; fi
  if [ -e "$OD" ]; then echo "SKIP $CLIP $LABEL dir exists without an sbs.mkv (failed run?) -- not overwritten" >> $LOG; continue; fi
  if [ "$CK" = "-" ]; then export SK_CK=''; else export SK_CK=$CK; [ -f "$CK" ] || { echo "BADCK $CLIP $LABEL $CK" >> $LOG; continue; }; fi
  export SK_STEPS=$STEPS SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  T0=$(date +%s.%N)
  flock /tmp/claude-gpu1.lock $PY $RUN > $OD.log 2>&1 < /dev/null
  RC=$?
  T1=$(date +%s.%N)
  SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  NDIFF=$(grep -oE 'tensors that differed from the loaded model: [0-9]+/[0-9]+' $OD.log | tail -1)
  echo "RUN $CLIP $LABEL ck=$CK steps=$STEPS rc=$RC secs=$SECS md5=$MD5 [$NDIFF] $(date +%F_%T) dir=$OD" >> $LOG
done < "$JOBS"
echo "DRIVER_DONE jobs=$JOBS $(date +%F_%T)" >> $LOG
