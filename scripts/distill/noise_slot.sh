#!/bin/bash
# Lane 2 (GPU 1): per-slot noise sensitivity on the ORIGIN UNet (gate 0 = origin attention at every slot),
# relative Gaussian noise injected into ONE slot at a time. Tells which slot's error costs quality.
set -u; cd /home/kawa/master_project/StereoCrafter
D=scripts/distill; PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python; R=$D/runs/noise_slot_v2; mkdir -p $R
BEST=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_multiclip13_r2_mamba_only.pt
export CUDA_VISIBLE_DEVICES=0
for SLOT in down_blocks.0.attentions.0 down_blocks.0.attentions.1 up_blocks.3.attentions.0 up_blocks.3.attentions.1 up_blocks.3.attentions.2; do
  for N in 0.0 0.03 0.10 0.30; do
    O=outputs/diagnose_0160/light_lvl0/noise_slot_v2/${SLOT}_$N; mkdir -p $O
    CAP_NOISE=$N CAP_NOISE_SLOT=$SLOT CAP_SAVE=0 CAP_GATE=0.0 CAP_OUT=$O CAP_KEEP=1 CKPT=$BEST $PY $D/capture_attn.py > $O/capture.log 2>&1
    L=$(conda run -n stereocrafter --no-capture-output python3 $D/score_lpips.py outputs/diagnose_0160/origin_base_via_inference_py_guid101/0160_inpainting_results_sbs.mp4 $O/_inference_out/0160_inpainting_results_sbs.mp4 2>&1 | grep -E '^_inference_out' | awk '{print $2, $3, $4}')
    echo "NOISE slot=$SLOT rel=$N LPIPS/maskPSNR/sharp=$L  $(date +%H:%M:%S)"
  done
done
echo "NOISE_SLOT_DONE $(date +%H:%M:%S)"
