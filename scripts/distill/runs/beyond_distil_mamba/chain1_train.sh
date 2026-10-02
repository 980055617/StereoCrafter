#!/bin/bash
# MAMBA-side step distillation, round 1.  Same recipe as the origin-side smoke1 run, like for like:
#   2 clips (0301, 0204), steps {4,5,6}, MEASURED gains {0.7177, 0.8926, 0.9905}, x0-space residual,
#   AdamW lr 1e-5 / betas (0.9,0.999) / wd 0, grad clip 1.0, 800 steps, ckpt at 100/200/400/600/800,
#   M=4 Karras substeps, NO on-policy refresh.
# Trainable set: the three up_blocks.3 Mamba blocks' evaluated tensors (fwd.core.* + time_embed_proj.*),
#   30 tensors / 3,641,325 params.  down_blocks.0 is NOT trainable: verify_mamba_v1 measured that putting
#   trainable params there forces the backward through the whole UNet and OOMs at 24 GB.
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
D=scripts/distill/runs/beyond_distil_mamba
until ! systemctl --user is-active --quiet bdm-chain0b; do sleep 20; done
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BD_CLIPS=0301,0204 BD_STEP_SUBSET=4,5,6 BD_M=4 BD_TRAIN_STEPS=800 BD_LR=1e-5
export BD_SAVE=100,200,400,600,800
export BD_WEIGHTS=4:0.7177,5:0.8926,6:0.9905
export BD_SLOTS=up3
$PY $D/train_beyond_mamba.py mstudent1 > $D/train_mstudent1.log 2>&1
echo "TRAIN rc=$? $(date +%T)"
grep -E "^OUT|BD_TRAIN_DONE|\[post\]|DONE " $D/train_mstudent1.log | tail -20
echo CHAIN1_DONE
