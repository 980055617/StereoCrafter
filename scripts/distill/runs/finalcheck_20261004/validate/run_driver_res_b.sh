#!/bin/bash
# finalcheck_20261004/validate: lossless (FFV1) renders at an explicit resolution, GPU 1 only.  [_b: lowmem reader hook]
# usage: run_driver_res.sh <jobfile> <out_root>
# jobfile lines:  CLIP MODEL H W [SUFFIX]     MODEL = origin | deliv ; SUFFIX (e.g. _r2) only for a retry
#   origin: plain shipped UNet (unet_state_path None), MAMBA_SELF_ATTN_INCLUDE='__nomatch__' -- as
#           scripts/distill/runs/fulldata_v2/beyond4/run_jobs_v1.sh rendered the origin_ll rows
#   deliv : the deliverable state + gate 1.0 with the 5-slot Mamba env of
#           scripts/distill/runs/beyond_distil_mamba_scaled/run_driver_v3.sh (which rendered the deliverable rows)
# Always: 8 steps, guidance 1.01 (min=max), seed 1234 from the config, tile_num=1, FFV1 writer.
# NOTE: SK_UNET is set explicitly per row -- no ${SK_UNET:-...} default (an empty value must stay empty).
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/finalcheck_20261004/validate/infer_ll_hook_res_lowmem.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
JOBS=$1; OUT=$2
LOG=outputs/finalcheck_20261004/validate/timing_gpu1.txt
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_STEPS=8 SK_GUID=1.01 SK_CK=''
while read -r CLIP MODEL H W SUF REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  LABEL=${MODEL}_ll_${H}x${W}${SUF:-}   # optional 5th column = retry suffix (e.g. _r2) -> a NEW directory
  OD=$OUT/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null || [ -e "$OD.log" ]; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  mkdir -p $OD
  if [ "$MODEL" = "origin" ]; then
    unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
    export MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
    export SK_UNET=''
  elif [ "$MODEL" = "deliv" ]; then
    export MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*' MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
    export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
    export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
    export SK_UNET=$DELIV
  else echo "BAD MODEL $MODEL" | tee -a $LOG; continue; fi
  export SK_CLIP=$CLIP SK_OUT=$OD SK_H=$H SK_W=$W
  T0=$(date +%s)
  $PY $RUN > $OD.log 2>&1
  RC=$?
  SECS=$(( $(date +%s) - T0 ))
  # H2 log gates
  G=PASS
  grep -q "^\[lossless\] wrote FFV1 " $OD.log || G=FAIL_nowrite
  grep -q "res=${H}x${W}" $OD.log || G=FAIL_res
  grep -q "reader = lowmem_reader" $OD.log || G=FAIL_reader
  if [ "$MODEL" = "origin" ]; then
    grep -q "unet=None" $OD.log || G=FAIL_unet
    grep -q "Partial UNet state load" $OD.log && G=FAIL_partialload
  else
    grep -q "missing=1428 unexpected=0" $OD.log || G=FAIL_missing
    grep -q "updated 5 gated modules" $OD.log || G=FAIL_gate
  fi
  MD5=$(grep -oE "md5\(pre-encode\)=[0-9a-f]+" $OD.log | tail -1)
  echo "RUN $CLIP $LABEL rc=$RC secs=$SECS gate=$G $MD5 $(date '+%F %T') dir=$OD" | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE $(basename $JOBS) $(date '+%F %T')" | tee -a $LOG
