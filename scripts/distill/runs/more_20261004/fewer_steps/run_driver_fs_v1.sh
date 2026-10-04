#!/bin/bash
# more_20261004 / fewer_steps lane driver.  COPY of scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh
# (kept verbatim next to this file as run_driver_speed_v1_ORIG_COPY.sh, md5 924b1c255fb1cd7e0be372d4c5a60d4d), changed ONLY:
#   - conda env activated first (project rule; sets PYTHONPATH / LD_LIBRARY_PATH exactly as an interactive run would)
#   - OUT=outputs/more_20261004/fewer_steps
#   - RUN = this lane's VERBATIM copy of the speed hook (infer_ll_hook_speed_v1_COPY.py, md5 7f5bf0becdd96e43a4a09075f35ad29d
#     == the original); RUN can be overridden per job file via the env var FS_RUN (used only for the oracle script)
#   - every python call is wrapped in `flock /tmp/claude-gpu$GPU.lock` -- the lock is held PER JOB, not for the chain;
#     secs= therefore includes any wait for the lock (the hook's own hook_total_s / call_s / unet_ms do not)
#   - GPU must be 0 (this lane's GPU); the origin_cli branch is removed (not used here)
# usage: run_driver_fs_v1.sh <jobfile> <gpu>
# jobfile lines:  CLIP LABEL MODEL GUID SIGMAS PAD      (MODEL: deliv | origin | mamba)
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set +u; source /home/kawa/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=${FS_RUN:-scripts/distill/runs/more_20261004/fewer_steps/infer_ll_hook_speed_v1_COPY.py}
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
MAMBA=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
OUT=outputs/more_20261004/fewer_steps
JOBS=$1; GPU=$2
[ "$GPU" = "0" ] || { echo "this lane runs on GPU 0 only"; exit 2; }
LOCK=/tmp/claude-gpu$GPU.lock
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=${SK_GUID:-1.00}     # this lane: default 1.00 (every job line gives GUID explicitly anyway)
GDEF=$SK_GUID
export SK_STEPS=${SK_STEPS:-8}
echo "LANE_START gpu$GPU jobs=$JOBS run=$RUN $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG PAD REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/clips/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  if [ -e "$OD" ]; then echo "SKIP $CLIP $LABEL dir exists without an sbs.mkv (failed run?) -- not overwritten" | tee -a $LOG; continue; fi
  if [ "$G" = "-" ]; then GG=$GDEF; else GG=$G; fi
  case "$MODEL" in
    deliv|mamba)
      export MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*' MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
      export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
      export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
      if [ "$MODEL" = "deliv" ]; then export SK_UNET=$DELIV; else export SK_UNET=$MAMBA; fi ;;
    origin)
      unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
      export MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
      export SK_UNET='' ;;
    *) echo "BADMODEL $CLIP $LABEL $MODEL" | tee -a $LOG; continue ;;
  esac
  if [ "$SIG" = "-" ]; then export SK_SIGMAS=''; else export SK_SIGMAS=$SIG; fi
  if [ "$PAD" = "-" ]; then export SK_RNG_PAD_TO=''; else export SK_RNG_PAD_TO=$PAD; fi
  export SK_GUID=$GG SK_CK='' SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  T0=$(date +%s.%N)
  flock $LOCK bash -c "echo LOCKED_AT \$(date +%H:%M:%S.%N) > $OD.lockwait; \
    GPUS=\$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' | tr -d ' '); \
    echo GPUS_AT_START \$GPUS >> $OD.lockwait; \
    $PY $RUN > $OD.log 2>&1"
  RC=$?
  T1=$(date +%s.%N)
  SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  GS=$(grep GPUS_AT_START $OD.lockwait 2>/dev/null | cut -d' ' -f2)
  echo "RUN $CLIP $LABEL model=$MODEL guid=$GG sigmas=${SIG} pad=${PAD} rc=$RC secs_incl_lockwait=$SECS md5=$MD5 gpus_at_start=$GS $(date +%H:%M:%S) dir=$OD :: $SP" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
