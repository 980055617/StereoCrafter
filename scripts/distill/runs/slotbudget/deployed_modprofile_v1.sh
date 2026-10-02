#!/bin/bash
# SLOT BUDGET cross-check on the DEPLOYED path: inpainting_inference.py's own
# --module_profile_json / --module_profile_include instrument (utils/module_timing, default *.attn1),
# 2 chunks only (--max_profile_chunks=2), exactly as the existing precedent
# outputs/diagnose_0160/module_timing_reference_matched_2chunks_v2/module_timing.json was produced.
# Deployed config = config/0160_overfit_inference_matched.json: 8 steps, guidance 1.01,
# 14-frame chunks overlap 3, 576x1024, tile_num 1, precision bf16 (NOTE: bench2.py times fp16).
# The mp4v videos these runs write are a byproduct of profiling and MUST NOT be used for quality
# numbers (measurement rule (4)); they live in their own new directory.
set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/slotbudget
MAMBA=/mnt/ssd_data/stereocrafter_weights/_distill_injected/light_lvl0_fulldata333_v2_8k_mamba_only.pt
CLIP=0301
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $R/profiles outputs/slotbudget_modprofile
run_cfg() {   # label include
  local L=$1 INC=$2
  local OD=outputs/slotbudget_modprofile/${CLIP}_${L}
  mkdir -p $OD
  export MAMBA_SELF_ATTN_INCLUDE="$INC" MAMBA_SELF_ATTN_EXCLUDE='__nomatch__'
  export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1
  export MAMBA_BIDIRECTIONAL_MODE=fwd MAMBA_SELF_ATTN_REPLACEMENT=gated_residual MAMBA_ADAPTER_LOG=1
  local EXTRA=""
  if [ "$L" = "origin" ]; then EXTRA="--unet_state_path=None"
  else EXTRA="--unet_state_path=$MAMBA --expected_partial_unet_state=True --mamba_gate_override=1.0"; fi
  python3 inpainting_inference.py --config=config/0160_overfit_inference_matched.json \
    --input_video_path=video_data/splatting/${CLIP}_splatting_results.mp4 \
    --save_dir=$OD $EXTRA \
    --max_profile_chunks=2 \
    --module_profile_json=$PWD/$R/profiles/deployed_${CLIP}_${L}.json \
    --module_profile_include='*.attn1' > $OD.log 2>&1
  echo "  deployed $L rc=$? $(grep -c 'MambaAdapter.\[self-attn\] at=' $OD.log) replaced, $(date +%H:%M:%S)"
  grep -E 'total_replaced|gate_override|totalProfiledMs' $OD.log | tail -3
}
echo "DEPLOYED_MODPROFILE_START $(date +%F_%H:%M:%S)"
run_cfg origin '__nomatch__'
run_cfg mamba5 'down_blocks.0.*,up_blocks.3.*'
run_cfg mamba2 'down_blocks.0.*'
echo "DEPLOYED_MODPROFILE_DONE $(date +%F_%H:%M:%S)"
