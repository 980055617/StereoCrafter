#!/bin/bash
# OOD v2: pre-cut each bundle clip to 231 frames (21 windows) so the whole-clip fp32 decode (~80 GB for 1800 frames @1440x2560,
# the host-OOM that killed VS Code at 13:23) never happens; then origin s1234 / origin s1 / student s1234 on the short clips.
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill; R=$D/runs/fulldata/ood; O=outputs/fulldata/ood; export CUDA_VISIBLE_DEVICES=0
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python; J=../shared_volume/sam2_bundle_jobs
declare -A IN=( [car]=$J/FINNAL_CAR/car_demo_work_1280x720_2x2_video.mp4 [animal]=$J/20260917-animal-fullframe/animal_demo_work_1280x720_2x2_video.mp4 [human]=$J/20260917-human-fullframe/Human_demo_work_1280x720_rose_2x2_video.mp4 )
for C in car animal human; do S=$O/inputs/${C}_short_splatting_results.mp4; [ -f $S ] && continue
  $PY - "${IN[$C]}" "$S" <<'PY'
import sys, numpy as np, cv2; from decord import VideoReader, cpu
src, dst = sys.argv[1:3]; vr = VideoReader(src, ctx=cpu(0)); fps = float(vr.get_avg_fps()); n = min(231, len(vr))
f0 = vr[0].asnumpy(); h, w = f0.shape[:2]; out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for i in range(0, n, 16):
    for fr in vr.get_batch(list(range(i, min(i + 16, n)))).asnumpy(): out.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
out.release(); print("wrote", dst, n, "frames", (w, h), fps)
PY
done
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_8k_mamba_only.pt; ARGS=""
for C in car animal human; do S=$O/inputs/${C}_short_splatting_results.mp4
  for V in "origin_s1234:ORIGIN:1234" "origin_s1:ORIGIN:1" "student_s1234:$CK:1234"; do
    L=${V%%:*}; rest=${V#*:}; K=${rest%%:*}; SD=${rest##*:}; OD=$O/${C}_$L; mkdir -p $OD
    if ls $OD/*_inpainting_results_sbs.mp4 >/dev/null 2>&1; then :; elif [ "$K" = "ORIGIN" ]; then
      MAMBA_SELF_ATTN_INCLUDE='__nomatch__' conda run -n stereocrafter --no-capture-output python3 inpainting_inference.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None \
        --noise_seed=$SD --input_video_path="$S" --save_dir=$OD > $OD.log 2>&1
    else
      conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py --unet_state_path=$K --include_patterns='down_blocks.0.*,up_blocks.3.*' --exclude_patterns='__nomatch__' --mamba_gate_override=1.0 \
        --noise_seed=$SD --input_video_path="$S" --save_dir=$OD > $OD.log 2>&1; fi
    echo "  $C $L done $(date +%H:%M:%S)"
  done
  f() { ls $O/${C}_$1/*_inpainting_results_sbs.mp4 | head -1; }
  ARGS="$ARGS ${C}:student_vs_origin $(f student_s1234) $(f origin_s1234)  ${C}:origin_seed1_vs_seed1234 $(f origin_s1) $(f origin_s1234)"
done
for C in 0160 0042; do ARGS="$ARGS ${C}:student_vs_origin outputs/fulldata/clips/${C}_all_8k/${C}_inpainting_results_sbs.mp4 outputs/fulldata/clips/${C}_origin/${C}_inpainting_results_sbs.mp4  ${C}:origin_seed1_vs_seed1234 outputs/fulldata/seedfloor/${C}_origin_s1/${C}_inpainting_results_sbs.mp4 outputs/fulldata/clips/${C}_origin/${C}_inpainting_results_sbs.mp4"; done
echo "=== OOD: distance to origin (LPIPS/PSNR) vs origin's own seed spread ==="; conda run -n stereocrafter --no-capture-output python3 $D/score_pair.py $ARGS 2>&1 | grep -vE 'Warning|warn|Setting up|Loading model|/home/kawa' | tee $R/ood.txt
echo "OOD_DONE $(date +%H:%M:%S)"
