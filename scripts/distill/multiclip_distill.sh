#!/bin/bash
cd /home/kawa/master_project/StereoCrafter
S=/home/kawa/master_project/StereoCrafter/scripts/distill
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python; R=$S/runs; A=/mnt/ssd_data/attn_cache
until grep -q 'CLIPS_DISTILLED_DONE' $S/clips_distilled.log 2>/dev/null; do sleep 60; done
INIT=$R/onpolicy_r3.pt; [ -f $INIT ] || INIT=$R/onpolicy_r2.pt
for ROUND in 1 2; do
  AGG=""
  for C in 0160 0001 0002 0003; do
    CACHE=$A/mc_r${ROUND}_$C; rm -rf $CACHE
    echo "MC ROUND $ROUND CAPTURE $C START $(date +%H:%M:%S) init=$INIT"
    CAP_ONPOLICY=1 CAP_OUT=$CACHE CAP_KEEP=3 CKPT=$INIT CAP_VIDEO=video_data/splatting/${C}_splatting_results.mp4 $PY $S/capture_attn.py > $R/mc_r${ROUND}_${C}_capture.log 2>&1
    grep -E 'ON-POLICY|Traceback|Error' $R/mc_r${ROUND}_${C}_capture.log | tail -1 | cut -c1-300
    AGG=${AGG:+$AGG:}$CACHE
  done
  echo "MC ROUND $ROUND TRAIN_START $(date +%H:%M:%S) data=$AGG"
  CACHE=$AGG CKPT=$INIT STEPS=4000 LR=2e-4 BATCH=8 EVAL_EVERY=1000 OUT=$R/mc_r$ROUND.json SAVE=$R/mc_r$ROUND.pt $PY $S/distill_standalone.py > $R/mc_r$ROUND.log 2>&1
  grep -E 'SUMMARY|Traceback' $R/mc_r$ROUND.log | cut -c1-400
  $S/eval_light.sh $R/mc_r$ROUND.pt origin_plus_mc_r$ROUND 2>&1 | grep -E "LPIPS|origin_plus_mc_r$ROUND |^RESULT"
  INIT=$R/mc_r$ROUND.pt
done
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual
for C in 0042 0204 0301; do O=outputs/diagnose_0160/clips/${C}_origin_plus_mc; mkdir -p $O
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py \
    --unet_state_path="$INIT" --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 \
    --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir="$O" > "$O.log" 2>&1; echo "clip $C done $(date +%H:%M:%S)"; done
ARGS=""; for C in 0042 0204 0301; do for K in origin light_e210 origin_plus_distilled origin_plus_mc; do ARGS="$ARGS ${C}=outputs/diagnose_0160/clips/${C}_${K}/${C}_inpainting_results_sbs.mp4"; done; done
conda run -n stereocrafter --no-capture-output python3 $S/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa"
echo "MULTICLIP_DONE $(date +%H:%M:%S)"
