#!/bin/bash
# finalcheck_20261004 / independent lane driver.  COPY of speed/run_driver_speed_v1.sh changed ONLY as follows:
#   - RUN = independent/infer_ll_hook_speed_res_v1.py (the speed hook + validate's SK_H/SK_W knob + lowmem reader)
#   - OUT = outputs/finalcheck_20261004/independent/hres
#   - two extra job columns H W ('-' '-' = the config's 576x1024, nothing passed) -> SK_H / SK_W
#   - origin_cli mode dropped (not used here)
#   - H2-style log gates appended to the RUN line (wrote FFV1, lowmem reader, res, model load lines)
# usage: run_driver_hres_v1.sh <jobfile> <gpu>
# jobfile lines:  CLIP LABEL MODEL GUID SIGMAS PAD H W
# Every run gets its own NEW directory; an existing one is skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
RUN=scripts/distill/runs/finalcheck_20261004/independent/infer_ll_hook_speed_res_v1.py
DELIV=/mnt/ssd_data/stereocrafter_weights/_distill_injected/mamba5slot_plus_stepdistil_up3_train10clip_step800_20261001.pt
OUT=outputs/finalcheck_20261004/independent/hres
JOBS=$1; GPU=$2
LOG=$OUT/timing_gpu$GPU.txt
mkdir -p $OUT/clips
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
export SK_STEPS=8
echo "LANE_START gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
while read -r CLIP LABEL MODEL G SIG PAD H W REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  OD=$OUT/clips/${CLIP}_${LABEL}
  if compgen -G "$OD/*_sbs.mkv" >/dev/null; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  if [ -e "$OD" ]; then echo "SKIP $CLIP $LABEL dir exists without an sbs.mkv (failed run?) -- not overwritten" | tee -a $LOG; continue; fi
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
  if [ "$H" = "-" ]; then export SK_H='' SK_W=''; RESTAG=config; else export SK_H=$H SK_W=$W; RESTAG=${H}x${W}; fi
  export SK_GUID=$G SK_CK='' SK_CLIP=$CLIP SK_OUT=$OD
  mkdir -p $OD
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';' | tr -d ' ')
  T0=$(date +%s.%N)
  /usr/bin/time -v $PY $RUN > $OD.log 2>&1
  RC=$?
  T1=$(date +%s.%N)
  SECS=$(awk -v a=$T0 -v b=$T1 'BEGIN{printf "%.1f", b-a}')
  GATE=PASS
  grep -q "^\[lossless\] wrote FFV1 " $OD.log || GATE=FAIL_nowrite
  grep -q "reader = lowmem_reader" $OD.log || GATE=FAIL_reader
  grep -q "res=${RESTAG}" $OD.log || GATE=FAIL_res
  if [ "$MODEL" = "origin" ]; then
    grep -q "unet=None" $OD.log || GATE=FAIL_unet
    grep -q "Partial UNet state load" $OD.log && GATE=FAIL_partialload
  else
    grep -q "missing=1428 unexpected=0" $OD.log || GATE=FAIL_missing
    grep -q "updated 5 gated modules" $OD.log || GATE=FAIL_gate
  fi
  SP=$(grep -oE '^\[speed\] windows=.*' $OD.log | tail -1)
  RSS=$(grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' $OD.log | grep -oE '[0-9]+$')
  MD5=$(cut -d' ' -f1 $OD/writer_md5.txt 2>/dev/null | head -1)
  echo "RUN $CLIP $LABEL model=$MODEL guid=$G sigmas=${SIG} pad=${PAD} res=$RESTAG rc=$RC gate=$GATE secs=$SECS md5=$MD5 maxrss_kb=${RSS:-NA} load1=$LOAD gpus=$GPUS $(date +%H:%M:%S) dir=$OD :: $SP" \
    | tee -a $LOG
done < "$JOBS"
echo "LANE_DONE gpu$GPU jobs=$JOBS $(date +%F_%T)" | tee -a $LOG
