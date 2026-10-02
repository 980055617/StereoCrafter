#!/bin/bash
# GT-supervised fine-tune of ONLY the 5 distilled Mamba blocks (base frozen), 40 clips, 6 epochs, 2 GPUs (DeepSpeed).
# Leash: origin-attention feature loss 0.02 keeps the blocks near the distilled solution; mamba lr 1e-5 (warm start).
set -u; cd /home/kawa/master_project/StereoCrafter
df -BG /mnt/ssd_data | awk 'NR==2{gsub("G","",$4); if(($4+0)<200){print "DISK_LOW"; exit 1}}' || exit 1
[ "$(ps -eo args | grep -cE '^python[^ ]* (-u )?inpainting_(train|inference)' || true)" -gt 0 ] && { echo "GPU busy"; exit 1; }
LOG=logs/gt_finetune_v2_originattn_control_$(date +%Y%m%d_%H%M%S).log; mkdir -p logs
MAMBA_SELF_ATTN_INCLUDE='down_blocks.0.*,up_blocks.3.*' MAMBA_SELF_ATTN_EXCLUDE='__nomatch__' MAMBA_SELF_ATTN_D_STATE=128 MAMBA_SELF_ATTN_EXPAND=1 MAMBA_BIDIRECTIONAL_MODE=fwd \
MAMBA_SELF_ATTN_REPLACEMENT=gated_residual MAMBA_ADAPTER_LOG=1 DS_ZERO_GRAD_FN_MODE=enable_grad PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n stereocrafter --no-capture-output deepspeed --num_gpus=2 --master_port=29581 --enable_each_rank_log logs \
  inpainting_train_gated_residual_mamba_up_only_exclude_up3_attn1.py --config=config/gt_finetune_v2_originattn_control.json \
  --resume_from=/mnt/ssd_data/stereocrafter_weights/_distill_injected/origin_attention_control_e150seed.pt --save_dir=weights/GTfinetune_v2_originattn_control \
  --stage_epochs='[1,2,2]' --include_patterns='__nomatch__' --exclude_patterns='__nomatch__' --mamba_gate_start=1.0 --mamba_gate_end=1.0 --save_interval_epochs=1 > "$LOG" 2>&1
rc=$?; echo "$LOG"; exit $rc
