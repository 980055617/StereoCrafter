#!/bin/bash
cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata/beyond; O=outputs/fulldata/beyond; mkdir -p $O; export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS=""
for C in 0160 0042; do
  ARGS="$ARGS ${C}=outputs/fulldata/clips/${C}_origin/${C}_inpainting_results_sbs.mp4"
  for V in "steps16:--num_inference_steps=16" "steps25:--num_inference_steps=25" "guid15:--min_guidance_scale=1.5 --max_guidance_scale=1.5" "guid10:--min_guidance_scale=1.0 --max_guidance_scale=1.0" "ovl_prev1:--overlap_prev_weight=1.0"; do
    L=${V%%:*}; FL=${V#*:}; OD=$O/${C}_origin_$L; mkdir -p $OD
    ls $OD/*_sbs.mp4 >/dev/null 2>&1 || MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None $FL --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$OD > $OD.log 2>&1
    ARGS="$ARGS ${C}=$OD/${C}_inpainting_results_sbs.mp4"; echo "  $C $L done $(date +%H:%M:%S)"
  done
done
echo "=== inference knobs (origin) ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/steps.txt; echo STEPS_DONE
