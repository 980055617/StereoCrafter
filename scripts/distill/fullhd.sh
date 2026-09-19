#!/bin/bash
# Full-HD frame condition (1920x1024 = the /128-cropped eye, no tiling, no center crop): origin vs the 333-clip model,
# LPIPS on 0160 + the 4 hires test clips, then an exclusive bench at 1920x1024. Runs after the tail (bench) is done.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata
until grep -q TAIL_DONE $R/tail.log 2>/dev/null; do sleep 120; done
export CUDA_VISIBLE_DEVICES=1 MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$R/fits/all_8k.best.pt; HT=$(python3 -c "import json;print(' '.join(json.load(open('$R/hires_test_clips.json'))))"); ARGS=""
echo "FULLHD_START $(date +%H:%M:%S)"
for C in 0160 $HT; do
  O=outputs/fulldata/fullhd/${C}_origin; mkdir -p $O
  [ -f $O/${C}_inpainting_results_sbs.mp4 ] || MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None \
    --target_height=1024 --target_width=1920 --tile_num=1 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O > $O.log 2>&1
  O2=outputs/fulldata/fullhd/${C}_all_8k; mkdir -p $O2
  [ -f $O2/${C}_inpainting_results_sbs.mp4 ] || conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$CK --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 \
    --target_height=1024 --target_width=1920 --tile_num=1 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O2 > $O2.log 2>&1
  ARGS="$ARGS ${C}=$O/${C}_inpainting_results_sbs.mp4 ${C}=$O2/${C}_inpainting_results_sbs.mp4"; echo "  fullhd $C done $(date +%H:%M:%S)"
done
echo "=== FULL HD 1920x1024 (tiling off) ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/lpips/fullhd.txt
echo "BENCH_FULLHD (exclusive) $(date +%H:%M:%S)"; export CUDA_VISIBLE_DEVICES=0
for rep in 1 2; do for V in "origin:__nomatch__:128" "light_ds128:down_blocks.0.*,up_blocks.3.*:128" "light_ds32:down_blocks.0.*,up_blocks.3.*:32"; do
  L=${V%%:*}; rest=${V#*:}; INC=${rest%%:*}; DS=${rest##*:}
  H=1024 W=1920 BS=2 GATE=1.0 DS=$DS EXP=1 BIDIR=fwd conda run -n stereocrafter --no-capture-output python3 $D/bench2.py "${L}_1920x1024_r$rep" "$INC" "__nomatch__" 2>&1 | grep -E '^RESULT' | cut -c1-160
done; done | tee $R/bench_fullhd.txt
echo "FULLHD_DONE $(date +%H:%M:%S)"
