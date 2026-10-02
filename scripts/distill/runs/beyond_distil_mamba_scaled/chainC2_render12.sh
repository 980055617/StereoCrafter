#!/bin/bash
# Render the HEADLINE row: the 12 test clips at the deployed config through the LOSSLESS FFV1 path,
# with the MERGED DELIVERABLE loaded through the TRACKED entry point (infer_ll_hook.py calls
# inpainting_inference.run with unet_state_path=<deliverable> expected_partial_unet_state=True
# mamba_gate_override=1.0 and NO hook swap -- SK_CK is empty), so the headline numbers come from the
# shipped file itself and not from a runtime tensor overlay.
# usage: chainC2_render12.sh <gpu> <deliverable.pt> <label> <clip...>
set -u
cd /home/kawa/master_project/StereoCrafter
D=scripts/distill/runs/beyond_distil_mamba_scaled
GPU=$1; DELIV=$2; LABEL=$3; shift 3
J=$D/jobs_headline_gpu$GPU.txt; : > $J
for C in "$@"; do echo "$C $LABEL none" >> $J; done
wc -l $J
export SK_UNET=$DELIV
bash $D/run_driver_v3.sh $J $GPU
echo CHAINC2_DONE_gpu$GPU $(date +%T)
