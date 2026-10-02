#!/bin/bash
# Out-of-dataset check on the bundle project's real clips (car / animal / human, 1280x720 work res -> 576x1024 center crop):
# origin (seed 1234), origin (seed 1), student (seed 1234); first 20 windows (~223 frames). Distance student<->origin vs origin's own seed spread.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata/ood; O=outputs/fulldata/ood; export CUDA_VISIBLE_DEVICES=0
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_8k_mamba_only.pt; J=../shared_volume/sam2_bundle_jobs
declare -A IN=( [car]=$J/FINNAL_CAR/car_demo_work_1280x720_2x2_video.mp4 [animal]=$J/20260917-animal-fullframe/animal_demo_work_1280x720_2x2_video.mp4 [human]=$J/20260917-human-fullframe/Human_demo_work_1280x720_rose_2x2_video.mp4 )
ARGS=""
for C in car animal human; do
  for V in "origin_s1234:ORIGIN:1234" "origin_s1:ORIGIN:1" "student_s1234:$CK:1234"; do
    L=${V%%:*}; rest=${V#*:}; K=${rest%%:*}; S=${rest##*:}; OD=$O/${C}_$L; mkdir -p $OD
    if [ -f $OD/*_inpainting_results_sbs.mp4 ]; then :; elif [ "$K" = "ORIGIN" ]; then
      MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None \
        --noise_seed=$S --max_profile_chunks=20 --input_video_path="${IN[$C]}" --save_dir=$OD > $OD.log 2>&1
    else
      conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$K --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 \
        --noise_seed=$S --max_profile_chunks=20 --input_video_path="${IN[$C]}" --save_dir=$OD > $OD.log 2>&1; fi
    echo "  $C $L done $(date +%H:%M:%S)"
  done
  f() { ls $O/${C}_$1/*_inpainting_results_sbs.mp4 | head -1; }
  ARGS="$ARGS ${C}:student_vs_origin $(f student_s1234) $(f origin_s1234)  ${C}:origin_seed1_vs_seed1234 $(f origin_s1) $(f origin_s1234)"
done
# in-dataset reference rows for the same metric (0160 / 0042: student vs origin, origin seed1 vs seed1234)
for C in 0160 0042; do ARGS="$ARGS ${C}:student_vs_origin outputs/fulldata/clips/${C}_all_8k/${C}_inpainting_results_sbs.mp4 outputs/fulldata/clips/${C}_origin/${C}_inpainting_results_sbs.mp4  ${C}:origin_seed1_vs_seed1234 outputs/fulldata/seedfloor/${C}_origin_s1/${C}_inpainting_results_sbs.mp4 outputs/fulldata/clips/${C}_origin/${C}_inpainting_results_sbs.mp4"; done
echo "=== OOD: distance to origin (LPIPS/PSNR) vs origin's own seed spread ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_pair.py $ARGS 2>&1 | grep -vE 'Warning|warn|Setting up|Loading model|/home/kawa' | tee $R/ood.txt
echo "OOD_DONE $(date +%H:%M:%S)"
