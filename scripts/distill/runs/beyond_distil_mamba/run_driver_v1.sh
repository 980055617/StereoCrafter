#!/bin/bash
# Deployed-config lossless inference driver for the Mamba-side student.
# usage: run_driver_v1.sh <jobfile> <gpu>
# jobfile lines:  CLIP LABEL CKPATH        (CKPATH = "none" for the plain shipped Mamba)
# Always: the SHIPPED 5-slot Mamba UNet state + gate 1.0, 8 steps, guidance 1.01, FFV1 lossless writer.
# Reuses scripts/distill/runs/skeptic1/infer_ll_hook.py UNCHANGED -- its swap mapping finds Mamba keys
# DIRECTLY (no .attn1.origin_attn.* remap), which the log line "swap mapping: N direct" confirms.
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/skeptic1/infer_ll_hook.py
MAMBA=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
OUT=outputs/beyond_distil_mamba
JOBS=$1; GPU=$2
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*' MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
export SK_UNET=$MAMBA SK_STEPS=8 SK_GUID=1.01
while read -r CLIP LABEL CK REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/clips/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  mkdir -p $OD
  if [ "$CK" = "none" ]; then export SK_CK=''; else export SK_CK=$CK; fi
  export SK_CLIP=$CLIP SK_OUT=$OD
  T0=$(date +%s)
  $PY $RUN > $OD.log 2>&1
  RC=$?
  MAP=$(grep -oE 'swap mapping: [0-9]+ direct, [0-9]+ remapped[^,]*, [0-9]+ NOT FOUND' $OD.log | tail -1)
  DIFF=$(grep -oE 'tensors that differed from the loaded model: [0-9]+/[0-9]+' $OD.log | tail -1)
  echo "RUN $CLIP $LABEL ck=$CK rc=$RC secs=$(( $(date +%s) - T0 )) [$MAP] [$DIFF] $(date +%H:%M:%S) dir=$OD" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU $(date +%H:%M:%S)" | tee -a $LOG
