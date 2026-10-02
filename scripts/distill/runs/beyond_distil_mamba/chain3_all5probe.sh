#!/bin/bash
# DECISION 2, measured in a CLEAN process: can the two down_blocks.0 Mamba slots be in the trainable set
# at all?  Putting trainable parameters in down_blocks.0 forces the backward to traverse the ENTIRE UNet
# instead of stopping at up_blocks.3, so the 15.26 GiB that up3-only peaked at does not transfer.
# verify_mamba_v1 with VM_SLOTS=all5 runs the step-0 backward with all five slots trainable and reports
# either the per-slot gradient norms + peak GiB, or the OOM.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
until ! systemctl --user is-active --quiet bdm-chain2; do sleep 30; done
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VM_CLIP=0301 VM_M=4 VM_WINS=0 VM_SLOTS=all5 VM_SKIP_PIXEL=1 VM_SKIP_ALL5=1 \
  $PY $D/verify_mamba_v1.py > $D/verify_0301_all5.log 2>&1
echo "all5 probe rc=$?"
grep -E "^\[set\]|^\[g0\]|^\[mem\]|^\[PASS\]|^\[FAIL\]|out of memory|CUDA out" $D/verify_0301_all5.log | tail -25
echo CHAIN3_DONE
