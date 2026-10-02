#!/bin/bash
# SKEPTIC stacking driver.  usage: stack_driver_v1.sh <jobfile> <gpu>
# jobfile lines: CLIP LABEL STEPS GUID UNETKEY CKKEY
#   UNETKEY: none | mamba          CKKEY: none | student
# Every run goes to its own NEW directory under outputs/skeptic1_stack/.  Never overwrites.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/skeptic1/infer_ll_hook.py
MAMBA=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
STUDENT=scripts/distill/runs/beyond_distil/smoke1/step800.pt
OUT=outputs/skeptic1_stack
JOBS=$1; GPU=$2
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
while read -r CLIP LABEL STEPS G UK CKK REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/clips/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null || compgen -G "$OD/*_sbs.mp4" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  mkdir -p $OD
  if [ "$UK" = "mamba" ]; then
    export SK_UNET=$MAMBA
    export MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*'
    export MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
    export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
    export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
  else
    export SK_UNET=''
    export MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
    unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT 2>/dev/null || true
  fi
  if [ "$CKK" = "student" ]; then export SK_CK=$STUDENT; else export SK_CK=''; fi
  export SK_CLIP=$CLIP SK_OUT=$OD SK_STEPS=$STEPS SK_GUID=$G
  T0=$(date +%s)
  $PY $RUN > $OD.log 2>&1
  RC=$?
  GATE=$(grep -c 'updated 5 gated modules' $OD.log)
  echo "RUN $CLIP $LABEL steps=$STEPS guid=$G unet=$UK ck=$CKK rc=$RC secs=$(( $(date +%s) - T0 )) gateline=$GATE $(date +%H:%M:%S) dir=$OD" | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU $(date +%H:%M:%S)" | tee -a $LOG
