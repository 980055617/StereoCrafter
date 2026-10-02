#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
D=scripts/distill/runs/diag_trainer
OD=$D/infer/0301_origin_condx0.18215; [ -f $OD/0301_inpainting_results_sbs.mp4 ] || { echo "START origin_condx0.18215 $(date +%T)"; python $D/lens1_infer_scaled_cond.py 0.18215 origin 0301 $OD > $OD.log 2>&1; echo "EXIT origin_condx0.18215 $? $(date +%T)"; }
for M in e1all e1high e1low e1up3 e1down0; do OD=$D/infer/0301_hyb_$M; mkdir -p $OD; [ -f $OD/0301_inpainting_results_sbs.mp4 ] && continue
  echo "START hyb_$M $(date +%T)"; python $D/xcheck_hybrid.py $M 0301 $OD > $OD.log 2>&1; echo "EXIT hyb_$M $? $(date +%T)"; done
echo XCHECK0_DONE
