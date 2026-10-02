#!/bin/bash
# CHAIN 1 (GPU 1): THE GATE.  Run the ORACLE sampler -- the deployed 8-step loop with the Euler step replaced by
# an M=4 Karras sub-integration of the same frozen origin UNet -- on 0301 and 0204, in two variants:
#   _oracle_m4      every coarse step substituted   (4x deployed UNet calls)
#   _oracle_m4_456  only k=4,5,6 substituted        (17/8 = 2.1x deployed UNet calls); target_check.py section D
#                   measured k<=3 as pure chain noise (their truncation error is at the bf16 floor 1.7e-3)
# and score against the deployed origin and s25 reference rows.
# PASS = the oracle lands near s25 (0301 0.4083, 0204 0.1950), NOT near deployed (0.4445 / 0.2122).
# Also runs the ZERO-COST scalar-step-rescale control with the section-F best-fit alphas, which section F
# predicts will NOT work (residual off the Euler direction is 86%/78% at k=4/k=5).  Nothing is trained here.
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
H=scripts/distill/runs/beyond_distil
mkdir -p outputs/beyond_distil
newdir(){ local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=$1_$k; done; echo $O; }
ARGS=""
for C in 0301 0204; do
  ARGS="$ARGS $C=outputs/fulldata_v2/clips/${C}_origin/${C}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $C=outputs/fulldata_v2/clips/${C}_origin_s25/${C}_inpainting_results_sbs.mp4"
  for SPEC in "oracle_m4:all" "oracle_m4_456:4,5,6"; do
    TAG=${SPEC%%:*}; SS=${SPEC##*:}
    O=$(newdir outputs/beyond_distil/${C}_${TAG})
    echo "START $TAG $C -> $O $(date +%T)"
    BD_M=4 BD_SUBST=$SS python $H/oracle_infer.py $C $O > ${O}.log 2>&1
    echo "EXIT $TAG $C $? $(date +%T)"
    F=$O/${C}_inpainting_results_sbs.mp4
    [ -f "$F" ] && ARGS="$ARGS $C=$F" || echo "MISSING $F"
  done
  # zero-cost control: Euler step length rescaled by beta_k = 1 + alpha_k (section F best fit, mean of w0/w66)
  O=$(newdir outputs/beyond_distil/${C}_rescale456)
  echo "START rescale $C -> $O $(date +%T)"
  BD_BETA="4:0.99457,5:0.93827,6:0.77198" python $H/rescale_infer.py $C $O > ${O}.log 2>&1
  echo "EXIT rescale $C $? $(date +%T)"
  F=$O/${C}_inpainting_results_sbs.mp4
  [ -f "$F" ] && ARGS="$ARGS $C=$F" || echo "MISSING $F"
done
echo "START score $(date +%T)"
python scripts/distill/score_clip.py $ARGS 2>&1 \
  | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $H/scores_chain1_oracle.txt
echo "EXIT score $? $(date +%T)"; echo CHAIN1_DONE
