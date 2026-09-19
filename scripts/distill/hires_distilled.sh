#!/bin/bash
# Re-test the "resolution brittleness" claim with the distilled light blocks: 1024x1792, tiling off, vs origin at the same resolution.
cd /home/kawa/master_project/StereoCrafter
S=/home/kawa/master_project/StereoCrafter/scripts/distill
until grep -q 'MULTICLIP_BIG_DONE' $S/multiclip_big.log 2>/dev/null; do sleep 60; done
CK=$S/runs/big_r2.pt; [ -f $CK ] || CK=$S/runs/mc_r2.pt; echo "using $CK"
O=outputs/diagnose_0160/hires/light_distilled_1024x1792; mkdir -p $O
MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py \
  --unet_state_path="$CK" --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 \
  --target_height=1024 --target_width=1792 --tile_num=1 --save_dir=$O > $O.log 2>&1
echo "hires inference done $(date +%H:%M:%S)"
conda run -n stereocrafter --no-capture-output python3 $S/score_hires.py \
  outputs/diagnose_0160/hires/origin_1024x1792/0160_inpainting_results_sbs.mp4 \
  outputs/diagnose_0160/hires/mamba_e140_1024x1792/0160_inpainting_results_sbs.mp4 \
  $O/0160_inpainting_results_sbs.mp4 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa"
echo "HIRES_DISTILLED_DONE $(date +%H:%M:%S)"
