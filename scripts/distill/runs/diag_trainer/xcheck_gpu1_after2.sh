#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill/runs/diag_trainer
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
until grep -q GRADCHAIN_DONE $D/xcheck_gpu1_after.log; do sleep 15; done
OD=$D/infer/0301_origin_fc2_condx0.18215; mkdir -p $OD; [ -f $OD/0301_inpainting_results_sbs.mp4 ] || { echo "START origin_fc2_condx0.18215 $(date +%T)"; python $D/lens1_infer_scaled_cond.py 0.18215 origin 0301 $OD 2 1 > $OD.log 2>&1; echo "EXIT origin_fc2_condx0.18215 $? $(date +%T)"; }
echo XCHECK1B_DONE
