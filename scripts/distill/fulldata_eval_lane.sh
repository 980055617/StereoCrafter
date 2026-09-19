#!/bin/bash
# GPU1: origin baselines -> reference row -> c13 fit+LPIPS (loader it/s check) -> seed floor -> c13w14/c40/c120 -> 'all' rows -> hires eval
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata; F=$R/fits; export CUDA_VISIBLE_DEVICES=1
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt
echo "EVAL_LANE_START $(date +%H:%M:%S)"
$D/eval_split.sh ORIGIN origin
$D/eval_split.sh $BEST ref_multiclip13
until [ -f $R/prefix_c13.ready ]; do sleep 60; done
$D/fulldata_fit.sh 13 fit_c13; grep -oE '[0-9.]+ it/s' $F/fit_c13.log | tail -5 | tr '\n' ' '; echo " <- loader throughput (fit_c13)"
$D/eval_split.sh $F/fit_c13.best.pt fit_c13
# seed floor: origin and reference student at 3 extra seeds on 0160 (2160) and 0042 (4400) -> LPIPS spread = the resolvable difference
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS=""
for C in 0160 0042; do for S in 1 2 3; do
  O=outputs/fulldata/seedfloor/${C}_origin_s$S; mkdir -p $O; [ -f $O/${C}_inpainting_results_sbs.mp4 ] || MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py \
    --config=config/0160_overfit_inference_matched.json --unet_state_path=None --noise_seed=$S --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O > $O.log 2>&1
  O2=outputs/fulldata/seedfloor/${C}_ref13_s$S; mkdir -p $O2; [ -f $O2/${C}_inpainting_results_sbs.mp4 ] || conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$BEST \
    --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 --noise_seed=$S --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O2 > $O2.log 2>&1
  ARGS="$ARGS ${C}=$O/${C}_inpainting_results_sbs.mp4 ${C}=$O2/${C}_inpainting_results_sbs.mp4"; echo "  seedfloor $C s$S done $(date +%H:%M:%S)"
done; done
ARGS="0160=outputs/fulldata/clips/0160_origin/0160_inpainting_results_sbs.mp4 0160=outputs/fulldata/clips/0160_ref_multiclip13/0160_inpainting_results_sbs.mp4 $ARGS 0042=outputs/fulldata/clips/0042_origin/0042_inpainting_results_sbs.mp4 0042=outputs/fulldata/clips/0042_ref_multiclip13/0042_inpainting_results_sbs.mp4"
echo "=== SEED FLOOR (seed 1234 = the standard row; s1-s3 extra seeds) ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/lpips/seedfloor.txt
for P in c13w14:c13w14 c40:40 c120:120; do
  pref=${P%%:*}; point=${P##*:}; until [ -f $R/prefix_$pref.ready ]; do sleep 60; done
  $D/fulldata_fit.sh $point fit_$pref; $D/eval_split.sh $F/fit_$pref.best.pt fit_$pref
done
for A in all_8k all_24k all_8k_lr1e-3 all_8k_hires all_8k_ds32; do until [ -f $F/$A.best.pt ]; do sleep 60; done; [ "$A" = "all_8k_ds32" ] && export MAMBA_SELF_ATTN_D_STATE=32; $D/eval_split.sh $F/$A.best.pt $A; done
export MAMBA_SELF_ATTN_D_STATE=128
# high-res eval: origin / ref13 / all_8k / all_8k_hires at 1024x1792 (tiling off) on 0160 + 4 test clips (2 per format)
HT=$(python3 -c "import json;print(' '.join(json.load(open('$R/hires_test_clips.json'))))")
HARGS=""
for C in 0160 $HT; do
  O=outputs/fulldata/hires/${C}_origin; mkdir -p $O
  if [ "$C" = "0160" ]; then ln -sf $(realpath outputs/diagnose_0160/hires/origin_1024x1792/0160_inpainting_results_sbs.mp4) $O/0160_inpainting_results_sbs.mp4;
  else [ -f $O/${C}_inpainting_results_sbs.mp4 ] || MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None \
    --target_height=1024 --target_width=1792 --tile_num=1 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O > $O.log 2>&1; fi
  HARGS="$HARGS ${C}=$O/${C}_inpainting_results_sbs.mp4"
  for W in ref13:$BEST all_8k:$F/all_8k.best.pt all_8k_hires:$F/all_8k_hires.best.pt; do L=${W%%:*}; CK=${W#*:}; O2=outputs/fulldata/hires/${C}_$L; mkdir -p $O2
    [ -f $O2/${C}_inpainting_results_sbs.mp4 ] || conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$CK --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 \
      --target_height=1024 --target_width=1792 --tile_num=1 --input_video_path=video_data/splatting/${C}_splatting_results.mp4 --save_dir=$O2 > $O2.log 2>&1
    HARGS="$HARGS ${C}=$O2/${C}_inpainting_results_sbs.mp4"; done
  echo "  hires $C done $(date +%H:%M:%S)"
done
echo "=== HIRES 1024x1792 (tiling off) ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_clip.py $HARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $R/lpips/hires.txt
echo "EVAL_LANE_DONE $(date +%H:%M:%S)"
