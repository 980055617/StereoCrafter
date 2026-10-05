#!/bin/bash
# deep_20261004 / input_side lane driver, NOLOCK variant (= run_driver_input_v1.sh minus the per-job flock; the
# caller -- a batch_*.sh run as `flock /tmp/claude-gpu0.lock bash batch_*.sh` -- holds the GPU-0 lock).
# Original header:  COPY of scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh
# (verbatim copy next to this file: run_driver_speed_v1_ORIG_COPY.sh), changed ONLY as follows:
#   - OUT=outputs/deep_20261004/input_side/clips ; runs infer_ll_hook_input_v1.py (the speed hook + the array reader)
#   - job lines:  CLIP LABEL MODEL INPUT        (no guidance/sigma/pad columns: everything is the DEPLOYED config,
#                 8 Euler steps, guidance 1.01, no RNG pad)
#       MODEL  deliv  = the deliverable .pt + the 5-slot Mamba env (exactly the speed driver's 'deliv')
#              origin = no unet state, MAMBA_SELF_ATTN_INCLUDE='__nomatch__' (exactly the speed driver's 'origin')
#       INPUT  a directory name under $INROOT/<clip>/ holding warped.npy + mask.npy (uint8 [T,576,1024,3]);
#              the left window is $PREP/<clip>/deployed_left.npy, fps from $PREP/<clip>/meta.json
#   - EVERY job runs under   flock /tmp/claude-gpu$GPU.lock   (lock held per job only)
# usage: run_driver_input_v1.sh <jobfile> <gpu>
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
if flock -n /tmp/claude-gpu0.lock true; then echo "NOT UNDER /tmp/claude-gpu0.lock -> abort"; exit 7; fi
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/deep_20261004/input_side/infer_ll_hook_input_v1.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
PREP=/mnt/ssd_data/deep_20261004/input_side/prep_v2
INROOT=/mnt/ssd_data/deep_20261004/input_side/inputs_v1
OUT=outputs/deep_20261004/input_side/clips
JOBS=$1; GPU=$2
LOG=outputs/deep_20261004/input_side/timing_gpu$GPU.txt
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=1.01 SK_STEPS=8 SK_SIGMAS='' SK_RNG_PAD_TO='' SK_CK=''
echo "LANE_START gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL INPUT REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  if [ -e "$OD" ]; then echo "SKIP $CLIP $LABEL dir exists without an sbs.mkv (failed run?) -- not overwritten" | tee -a $LOG; continue; fi
  IND=$INROOT/$CLIP/$INPUT
  if [ ! -f "$IND/warped.npy" ] || [ ! -f "$IND/mask.npy" ]; then echo "BADINPUT $CLIP $LABEL $IND" | tee -a $LOG; continue; fi
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
  FPS=$($PY -c "import json;print(json.load(open('$PREP/$CLIP/meta.json'))['fps'])")
  export IS_INPUT_DIR=$IND IS_LEFT=$PREP/$CLIP/deployed_left.npy IS_FPS=$FPS SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  T0=$(date +%s.%N)
  $PY $RUN > $OD.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  echo "RUN $CLIP $LABEL model=$MODEL input=$INPUT rc=$RC secs=$SECS md5=$MD5 $(date +%H:%M:%S) dir=$OD :: $SP" | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
