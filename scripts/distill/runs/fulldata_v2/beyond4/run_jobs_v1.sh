#!/bin/bash
# GPU0 lane: deployed-origin / s25 / guidance sweep with the LOSSLESS FFV1 writer.
# Every run gets its own NEW directory; existing dirs with an sbs file are skipped, never overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
S=/tmp/claude-1000/-home-kawa-master-project/f931e1a7-5010-427c-aa15-11fad555d1e2/scratchpad/ll
JOBS="$1"
OUT=outputs/beyond4_lossless
LOG=$OUT/timing_gpu0.txt
mkdir -p $OUT/clips $OUT/control
export CUDA_VISIBLE_DEVICES=0
while read -r CLIP LABEL STEPS G LL REST; do
  [ -z "${CLIP:-}" ] && continue
  case "$CLIP" in \#*) continue;; esac
  if [ "$LL" = "0" ]; then OD=$OUT/control/${CLIP}_${LABEL}; else OD=$OUT/clips/${CLIP}_${LABEL}; fi
  if ls $OD/*_sbs.mkv $OD/*_sbs.mp4 >/dev/null 2>&1; then echo "SKIP $CLIP $LABEL exists" | tee -a $LOG; continue; fi
  mkdir -p $OD
  T0=$(date +%s)
  MAMBA_SELF_ATTN_INCLUDE='__nomatch__' LOSSLESS_SBS=$LL KEEP_ANAGLYPH=0 \
  $PY $S/infer_lossless.py --config=config/0160_overfit_inference_matched.json \
    --unet_state_path=None --num_inference_steps=$STEPS \
    --min_guidance_scale=$G --max_guidance_scale=$G \
    --input_video_path=video_data/splatting/${CLIP}_splatting_results.mp4 \
    --save_dir=$OD > $OD.log 2>&1
  RC=$?
  echo "RUN $CLIP $LABEL steps=$STEPS guid=$G ll=$LL rc=$RC secs=$(( $(date +%s) - T0 )) $(date +%H:%M:%S) dir=$OD" | tee -a $LOG
done < "$JOBS"
echo "ALL_DONE $(date +%H:%M:%S)" | tee -a $LOG
