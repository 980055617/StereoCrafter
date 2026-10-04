#!/bin/bash
# [v3: identical to run_driver_stage_v2.sh except RUN = infer_stage_hook_v3.py and the per-job knob reset also clears SK_VAE_C3D2D SK_VAE_COMPILE TORCHINDUCTOR_CACHE_DIR]
# more_20261004 / pipeline_speed driver (GPU 0 only).  Derived from
# scripts/distill/runs/finalcheck_20261004/independent/run_driver_hres_v1.sh (kept as run_driver_hres_v1_ORIG_COPY.sh):
#   - RUN = infer_stage_hook_v3.py (stage timers + inert-unless-set fix knobs); reader = tracked unless SK_READER set
#   - every job runs INSIDE `flock /tmp/claude-gpu0.lock` (lock held for the WHOLE render = whole timing measurement);
#     the process wall-clock is taken inside the lock, so waiting for the lock is never counted
#   - job columns: CLIP LABEL MODEL GUID SIGMAS H W ENVS
#       SIGMAS '-' (scheduler default, SK_STEPS=8) | 'T5' (the T5 list) ; H W '-' '-' = config 576x1024
#       ENVS   '-' or comma-separated K=V knobs (SK_* of the hook, TRITON_*); all knobs are reset before each job
# usage: run_driver_stage_v1.sh <jobfile> <outroot>
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/more_20261004/pipeline_speed/infer_stage_hook_v3.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
T5=700.0,7.276163101196289,1.1675708293914795,0.09738767892122269,0.0020000000949949026
JOBS=$1; OUTROOT=$2
LOG=$OUTROOT/timing_gpu0.txt
mkdir -p $OUTROOT/clips
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_STEPS=8 SK_CK=''
echo "LANE_START gpu0 jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG H W ENVS REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUTROOT/clips/${CLIP}_${LABEL}
  if [ -e "$OD" ] || [ -e "$OD.log" ]; then echo "SKIP $CLIP $LABEL exists -- not overwritten" | tee -a $LOG; continue; fi
  unset SK_READER SK_DECODE_CHUNK SK_CUDNN_BENCH SK_CHANNELS_LAST SK_SKIP_DISCARDED SK_NOAUG_SKIP SK_DIRECT_POST \
        SK_AT_LOAD SK_AT_SAVE SK_SAVE_LAT SK_MAX_CHUNKS SK_REPEAT TRITON_CACHE_DIR TRITON_PRINT_AUTOTUNING SK_VAE_C3D2D SK_VAE_COMPILE TORCHINDUCTOR_CACHE_DIR
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
  case "$SIG" in -) export SK_SIGMAS='';; T5) export SK_SIGMAS=$T5;; *) export SK_SIGMAS=$SIG;; esac
  export SK_RNG_PAD_TO=''
  if [ "$H" = "-" ]; then export SK_H='' SK_W=''; RESTAG=config; else export SK_H=$H SK_W=$W; RESTAG=${H}x${W}; fi
  if [ "${ENVS:--}" != "-" ]; then
    IFS=',' read -ra KV <<< "$ENVS"
    for kv in "${KV[@]}"; do export "$kv"; done
  fi
  export SK_GUID=$G SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  flock /tmp/claude-gpu0.lock bash scripts/distill/runs/more_20261004/pipeline_speed/job_inner_v1.sh \
      "$PY" "$RUN" "$OD" "$LOG" "$CLIP $LABEL model=$MODEL guid=$G sigmas=$SIG res=$RESTAG envs=${ENVS:--}"
done < "$JOBS"
echo "LANE_DONE gpu0 jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
