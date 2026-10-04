#!/bin/bash
# more_20261004 / temporal lane driver.  COPY of scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh
# (kept verbatim next to this file as run_driver_SPEED_ORIG_COPY.sh), changed ONLY as follows:
#   - OUT=outputs/more_20261004/temporal ; runs infer_ll_hook_temporal_v1.py (speed hook + SK_OVERLAP/SK_PREVW/SK_DCS/
#     SK_NOISE knobs and a passive window-schedule check)
#   - GPU must be 1; EVERY python call runs under `flock /tmp/claude-gpu1.lock` (lock held per render, so other lanes
#     interleave between renders; a render's timing is taken while it holds the lock = nothing else on GPU 1)
#   - job columns: CLIP LABEL MODEL GUID SIGMAS PAD OVERLAP PREVW DCS NOISE   ('-' = config default / unset)
#   - origin_cli mode dropped (not used here)
# usage: run_driver_temporal_v1.sh <jobfile> 1
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/more_20261004/temporal/infer_ll_hook_temporal_v1.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
OUT=outputs/more_20261004/temporal
JOBS=$1; GPU=$2
[ "$GPU" = "1" ] || { echo "this lane runs on GPU 1 only"; exit 2; }
LOCK=/tmp/claude-gpu1.lock
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=${SK_GUID:-1.01}
GDEF=$SK_GUID
export SK_STEPS=${SK_STEPS:-8}
echo "LANE_START gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG PAD OV PW DCS NZ REST; do
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
  if [ "$SIG" = "-" ]; then export SK_SIGMAS=''; else export SK_SIGMAS=$SIG; fi
  if [ "$PAD" = "-" ]; then export SK_RNG_PAD_TO=''; else export SK_RNG_PAD_TO=$PAD; fi
  if [ "${OV:--}" = "-" ]; then export SK_OVERLAP=''; else export SK_OVERLAP=$OV; fi
  if [ "${PW:--}" = "-" ]; then export SK_PREVW=''; else export SK_PREVW=$PW; fi
  if [ "${DCS:--}" = "-" ]; then export SK_DCS=''; else export SK_DCS=$DCS; fi
  if [ "${NZ:--}" = "-" ]; then export SK_NOISE=''; else export SK_NOISE=$NZ; fi
  export SK_GUID=$GG SK_CK='' SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  TW0=$(date +%s.%N)
  # lock held for exactly this render; T0/T1 bracket the python process only (lock wait excluded)
  flock $LOCK bash -c '
    GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr "\n" ";" | tr -d " ")
    echo "$GPUS" > "$0.gpus_at_start"
    T0=$(date +%s.%N); "$1" "$2" > "$0.log" 2>&1; RC=$?; T1=$(date +%s.%N)
    echo "$RC $T0 $T1" > "$0.rc"' "$OD" "$PY" "$RUN"
  TW1=$(date +%s.%N)
  read -r RC T0 T1 < $OD.rc
  SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  WAIT=$(awk -v a=$TW0 -v b=$T0 'BEGIN{printf "%.1f", b-a}')
  GPUS=$(cat $OD.gpus_at_start)
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  echo "RUN $CLIP $LABEL model=$MODEL guid=$GG sigmas=${SIG} pad=${PAD} ov=${OV} pw=${PW} dcs=${DCS} noise=${NZ} rc=$RC secs=$SECS lockwait=$WAIT md5=$MD5 load1=$LOAD gpus=$GPUS $(date +%H:%M:%S) dir=$OD :: $SP" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
