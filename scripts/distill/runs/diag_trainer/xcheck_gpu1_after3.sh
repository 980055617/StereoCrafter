#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill/runs/diag_trainer
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
until grep -q XCHECK1B_DONE $D/xcheck_gpu1_after2.log; do sleep 15; done
echo "START grad_dir_lowsig $(date +%T)"; XSIG=13,16 XTAG=_lowsig python $D/xcheck_grad_dir.py > $D/xcheck_grad_dir_lowsig.log 2>&1; echo "EXIT grad_dir_lowsig $? $(date +%T)"; echo XCHECK1C_DONE
