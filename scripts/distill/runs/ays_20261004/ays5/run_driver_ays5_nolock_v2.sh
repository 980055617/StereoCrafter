#!/bin/bash
# [ays5 v2 = run_driver_ays5_v1.sh with ONLY the inner flock removed (PREREG ADDENDUM 2): run it ONLY under an outer
#  flock /tmp/claude-gpu0.lock held by the caller (batch_S3_v2.sh); CUDA_VISIBLE_DEVICES stays the driver argument (0).]
# [ays_20261004/ays5 COPY of scripts/distill/runs/more_20261004/judge/run_driver_ays_v1.sh; changes ONLY: RUN = the
#  byte-identical hook copy in this lane dir (md5 48924a205766e108bd05ded5a415a86f), OUT = outputs/ays_20261004/ays5.]
# [more_20261004/judge COPY of scripts/distill/runs/finalcheck_20261004/speed/run_driver_speed_v1.sh; changes ONLY:
#  RUN = judge/infer_ll_hook_ays_v1.py, OUT = outputs/more_20261004/judge/ays, the python call runs under
#  flock /tmp/claude-gpu$GPU.lock (secs then includes lock wait; timing is not used from this driver).]
# finalcheck_20261004 / speed lane driver.  COPY of scripts/distill/runs/beyond_distil_mamba_scaled/run_driver_v3.sh
# (kept verbatim next to this file as run_driver_v3_ORIG_COPY.sh), changed ONLY as follows:
#   - OUT=outputs/finalcheck_20261004/speed
#   - SK_GUID is inherited (default 1.01) and can be overridden per job (GUID column; '-' = inherited)
#   - per-job MODEL column:
#       deliv   the deliverable .pt as SK_UNET + the 5-slot Mamba env (exactly what rendered the deliverable)
#       origin  NO unet state (SK_UNET='' AFTER the old ${SK_UNET:-...} line can no longer fire), the other
#               MAMBA_* knobs unset and MAMBA_SELF_ATTN_INCLUDE='__nomatch__' (exactly the beyond4 origin_ll env)
#       origin_cli  the tracked beyond4 infer_lossless.py run directly by CLI (exactly run_jobs_v1.sh's invocation);
#                   only used as a cross-check that the hook path == the infer_lossless.py CLI path
#   - per-job SIGMAS column ('-' or a comma list ending at sigma_min) -> SK_SIGMAS, PAD column ('-' or M) -> SK_RNG_PAD_TO
#   - runs the copied hook infer_ll_hook_speed_v1.py (the skeptic1 hook + the two knobs + passive timing)
#   - logs process wall-clock with sub-second resolution plus the hook's [speed] summary
# usage: run_driver_speed_v1.sh <jobfile> <gpu>
# jobfile lines:  CLIP LABEL MODEL GUID SIGMAS PAD
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/ays_20261004/ays5/infer_ll_hook_ays_v1.py
ILCLI=scripts/distill/runs/fulldata_v2/beyond4/infer_lossless.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
OUT=outputs/ays_20261004/ays5
JOBS=$1; GPU=$2
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_GUID=${SK_GUID:-1.01}     # speed lane: inherited, default 1.01
GDEF=$SK_GUID
export SK_STEPS=${SK_STEPS:-8}
echo "LANE_START gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG PAD REST; do
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
    origin|origin_cli)
      unset MAMBA_SELF_ATTN_EXCLUDE MAMBA_SELF_ATTN_D_STATE MAMBA_SELF_ATTN_EXPAND MAMBA_BIDIRECTIONAL_MODE MAMBA_SELF_ATTN_REPLACEMENT
      export MAMBA_SELF_ATTN_INCLUDE='__nomatch__'
      export SK_UNET='' ;;
    *) echo "BADMODEL $CLIP $LABEL $MODEL" | tee -a $LOG; continue ;;
  esac
  if [ "$SIG" = "-" ]; then export SK_SIGMAS=''; else export SK_SIGMAS=$SIG; fi
  if [ "$PAD" = "-" ]; then export SK_RNG_PAD_TO=''; else export SK_RNG_PAD_TO=$PAD; fi
  export SK_GUID=$GG SK_CK='' SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' | tr -d ' ')
  T0=$(date +%s.%N)
  if [ "$MODEL" = "origin_cli" ]; then
    [ "$SIG" != "-" ] && { echo "BADJOB origin_cli takes no SIGMAS" | tee -a $LOG; continue; }
    $PY $ILCLI --config=config/0160_overfit_inference_matched.json \
      --unet_state_path=None --num_inference_steps=$SK_STEPS \
      --min_guidance_scale=$GG --max_guidance_scale=$GG \
      --input_video_path=video_data/splatting/${CLIP}_splatting_results.mp4 \
      --save_dir=$OD > $OD.log 2>&1
  else
    $PY $RUN > $OD.log 2>&1   # [v2: inner flock removed; the caller holds /tmp/claude-gpu0.lock for the whole batch]
  fi
  RC=$?
  T1=$(date +%s.%N)
  SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  echo "RUN $CLIP $LABEL model=$MODEL guid=$GG sigmas=${SIG} pad=${PAD} rc=$RC secs=$SECS md5=$MD5 load1=$LOAD gpus=$GPUS $(date +%H:%M:%S) dir=$OD :: $SP" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
