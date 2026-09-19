#!/bin/bash
# GPU1, sequential (one inference at a time for host RAM): ds32 row -> 1024x1792 on the 4 test clips -> 1920x1024 Full-HD rows -> exclusive benches
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata; F=$R/fits; export CUDA_VISIBLE_DEVICES=1
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt
echo "FIXUP_START $(date +%H:%M:%S)"
MAMBA_SELF_ATTN_D_STATE=32 $D/eval_split.sh $F/all_8k_ds32.best.pt all_8k_ds32
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HT=$(python3 -c "import json;print(' '.join(json.load(open('$R/hires_test_clips.json'))))")
run_res() { # run_res <H> <W> <outdir> <clip> <label> <ckpt|ORIGIN>
  local H=$1 W=$2 OD=$3 C=$4 L=$5 CK=$6 O=$3/${4}_$5; mkdir -p $O; [ -f $O/${C}_inpainting_results_sbs.mp4 ] && return 0
  if [ "$CK" = "ORIGIN" ]; then MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None \
      --target_height=$H --target_width=$W --tile_num=1 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O > $O.log 2>&1
  else conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$CK --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 \
      --target_height=$H --target_width=$W --tile_num=1 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O > $O.log 2>&1; fi
  echo "  ${H}x${W} $C $L done $(date +%H:%M:%S)"; }
HARGS="0160=outputs/fulldata/hires/0160_origin/0160_inpainting_results_sbs.mp4 0160=outputs/fulldata/hires/0160_ref13/0160_inpainting_results_sbs.mp4 0160=outputs/fulldata/hires/0160_all_8k/0160_inpainting_results_sbs.mp4 0160=outputs/fulldata/hires/0160_all_8k_hires/0160_inpainting_results_sbs.mp4"
for C in $HT; do
  run_res 1024 1792 outputs/fulldata/hires $C origin ORIGIN; run_res 1024 1792 outputs/fulldata/hires $C ref13 $BEST
  run_res 1024 1792 outputs/fulldata/hires $C all_8k $F/all_8k.best.pt; run_res 1024 1792 outputs/fulldata/hires $C all_8k_hires $F/all_8k_hires.best.pt
  for L in origin ref13 all_8k all_8k_hires; do HARGS="$HARGS ${C}=outputs/fulldata/hires/${C}_$L/${C}_inpainting_results_sbs.mp4"; done
done
echo "=== HIRES 1024x1792 (tiling off) ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_clip.py $HARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/lpips/hires.txt
FARGS=""
for C in 0160 $HT; do
  run_res 1024 1920 outputs/fulldata/fullhd $C origin ORIGIN; run_res 1024 1920 outputs/fulldata/fullhd $C all_8k $F/all_8k.best.pt; run_res 1024 1920 outputs/fulldata/fullhd $C all_8k_hires $F/all_8k_hires.best.pt
  for L in origin all_8k all_8k_hires; do FARGS="$FARGS ${C}=outputs/fulldata/fullhd/${C}_$L/${C}_inpainting_results_sbs.mp4"; done
done
echo "=== FULL HD 1920x1024 (tiling off) ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_clip.py $FARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/lpips/fullhd.txt
echo "BENCH_FULLHD (exclusive) $(date +%H:%M:%S)"; export CUDA_VISIBLE_DEVICES=0
for rep in 1 2; do for V in "origin:__nomatch__:128" "light_ds128:down_blocks.0.*,up_blocks.3.*:128" "light_ds32:down_blocks.0.*,up_blocks.3.*:32"; do
  L=${V%%:*}; rest=${V#*:}; INC=${rest%%:*}; DS=${rest##*:}
  for RES in "1024 1920" "1024 1792"; do set -- $RES; H=$1 W=$2 BS=2 GATE=1.0 DS=$DS EXP=1 BIDIR=fwd conda run -n stereocrafter --no-capture-output python3 $D/bench2.py "${L}_${2}x${1}_r$rep" "$INC" "__nomatch__" 2>&1 | grep -E '^RESULT' | cut -c1-160; done
done; done | tee $R/bench_fullhd.txt
echo "FIXUP_DONE $(date +%H:%M:%S)"
