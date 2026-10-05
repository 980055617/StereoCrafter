#!/bin/bash
# deep_20261004 / decoder_ft lane: LATENT-CAPTURE render driver.
# COPY of scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh (kept verbatim next to this file as
# run_driver_speed_v1_ORIG_COPY.sh), changed ONLY as follows:
#   - OUT=/mnt/ssd_data/deep_20261004/decoder_ft/capture   (these renders duplicate existing lossless renders; they exist to
#     prove the capture is passive -- writer md5 must equal the existing render's md5)
#   - RUN = capture_hook_v1.py (the speed hook + a passive decode_latents wrapper that saves every window's pre-decode latents)
#   - DF_LAT_DIR=/mnt/ssd_data/deep_20261004/decoder_ft/latents/<CLIP>_<LABEL>
#   - deployed settings only: guidance 1.01, 8 steps, no SK_SIGMAS, no RNG pad (the job columns GUID/SIGMAS/PAD must be '-')
#   - the caller holds the GPU lock for the whole batch (flock /tmp/claude-gpu<N>.lock bash run_capture_v1.sh <jobs> <N>)
# usage: flock /tmp/claude-gpu0.lock bash run_capture_v1.sh <jobfile> <gpu>
# jobfile lines:  CLIP LABEL MODEL - - -
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/deep_20261004/decoder_ft/capture_hook_v1.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
OUT=/mnt/ssd_data/deep_20261004/decoder_ft/capture
LATR=/mnt/ssd_data/deep_20261004/decoder_ft/latents
JOBS=$1; GPU=$2
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips $LATR
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=1.01 SK_STEPS=8
echo "LANE_START gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG PAD REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  if [ "$G" != "-" ] || [ "$SIG" != "-" ] || [ "$PAD" != "-" ]; then echo "BADJOB $CLIP $LABEL deployed settings only" | tee -a $LOG; continue; fi
  OD=$OUT/clips/${CLIP}_${LABEL}
  LD=$LATR/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  if [ -e "$OD" ] || [ -e "$LD" ]; then echo "SKIP $CLIP $LABEL dir exists without an sbs.mkv (failed run?) -- not overwritten" | tee -a $LOG; continue; fi
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
  export SK_SIGMAS='' SK_RNG_PAD_TO='' SK_CK='' SK_CLIP=$CLIP SK_OUT=$OD DF_LAT_DIR=$LD
  mkdir -p $OD
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' | tr -d ' ')
  T0=$(date +%s.%N)
  $PY $RUN > $OD.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  NL=$(ls $LD/w*.pt 2>/dev/null | wc -l)
  echo "RUN $CLIP $LABEL model=$MODEL rc=$RC secs=$SECS md5=$MD5 latwin=$NL load1=$LOAD gpus=$GPUS $(date +%H:%M:%S) dir=$OD :: $SP" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
