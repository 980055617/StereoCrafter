#!/bin/bash
# CORRECTED sensitivity calibration: the first run used the e210 checkpoint as "origin" (its base weights are not origin's).
# Now: origin UNet + Mamba-only state at gate 0 (=> the slots return origin attention) + relative Gaussian noise. 0.0 = sanity baseline.
cd /home/kawa/master_project/StereoCrafter
S=/home/kawa/master_project/StereoCrafter/scripts/distill
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python; CK=$S/runs/big_r2.pt
for N in 0.0 0.03 0.10 0.30; do
  O=outputs/diagnose_0160/light_lvl0/noise_sens_origin_$N; mkdir -p $O
  echo "NOISE $N START $(date +%H:%M:%S)"
  CAP_NOISE=$N CAP_SAVE=0 CAP_GATE=0.0 CAP_OUT=$O CAP_KEEP=1 CKPT=$CK PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $S/capture_attn.py > $O/capture.log 2>&1
  conda run -n stereocrafter --no-capture-output python3 $S/score_lpips.py outputs/diagnose_0160/origin_base_via_inference_py_guid101/0160_inpainting_results_sbs.mp4 $O/_inference_out/0160_inpainting_results_sbs.mp4 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tail -1
done
echo "NOISE_SENS2_DONE $(date +%H:%M:%S)"
