#!/bin/bash
# Evaluate one light-level-0 checkpoint: quality (aligned LPIPS vs origin) + verified speed.
# Usage: eval_light.sh <train_state_epochNNNNNN.pt> <label>
set -e
cd /home/kawa/master_project/StereoCrafter
S=/home/kawa/master_project/StereoCrafter/scripts/distill
CK=$1; LABEL=$2
export MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd
export MAMBA_SELF_ATTN_REPLACEMENT=gated_residual MAMBA_ADAPTER_LOG=1
INC='down_blocks.0.*,up_blocks.3.*'; EXC='__nomatch__'
O=outputs/diagnose_0160/light_lvl0/$LABEL; mkdir -p $O
echo "## [$LABEL] gate buffer stored in ckpt:"
conda run -n stereocrafter --no-capture-output python3 -c "
import torch; m=torch.load('$CK',map_location='cpu')['model']
print('   ', sorted({round(float(v),4) for k,v in m.items() if k.endswith('.mamba_gate')}))" 2>/dev/null
echo "## [$LABEL] inference (gate forced 1.0)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n stereocrafter --no-capture-output python3 inpainting_inference_hybrid_exclude_up3_attn1.py \
  --unet_state_path="$CK" --include_patterns="$INC" --exclude_patterns="$EXC" \
  --mamba_gate_override=1.0 --save_dir="$O" > "$O.log" 2>&1
grep -hE "total_replaced|Partial UNet|mamba_gate_override" "$O.log" | head -3
echo "## [$LABEL] aligned LPIPS vs origin"
conda run -n stereocrafter --no-capture-output python3 $S/score_lpips.py \
  outputs/diagnose_0160/origin_base_via_inference_py_guid101/0160_inpainting_results_sbs.mp4 \
  $O/0160_inpainting_results_sbs.mp4 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa"
echo "## [$LABEL] speed, trained weights, origin_attn must be 0 (batch 2 = published guidance 1.01)"
BS=2 CKPT=$CK GATE=1.0 DS=128 EXP=1 BIDIR=fwd \
conda run -n stereocrafter --no-capture-output python3 $S/bench2.py "$LABEL" "$INC" "$EXC" 2>&1 | grep -E "^RESULT"
echo "## [$LABEL] DONE"
