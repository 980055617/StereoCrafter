#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill/runs/diag_trainer
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
until grep -q XCHECK0_DONE $D/xcheck_gpu0_chain.log; do sleep 15; done
until grep -q SCORE_DONE $D/xcheck_score.log; do sleep 15; done   # scorer uses GPU0 briefly; one job per GPU
for M in "e1range:-9:0.3|e1last3" "e1range:0.3:1.0|e1mid"; do MODE=${M%%|*}; L=${M##*|}; OD=$D/infer/0301_hyb_$L; mkdir -p $OD; [ -f $OD/0301_inpainting_results_sbs.mp4 ] && continue
  echo "START hyb_$L $(date +%T)"; python $D/xcheck_hybrid.py "$MODE" 0301 $OD > $OD.log 2>&1; echo "EXIT hyb_$L $? $(date +%T)"; done
echo XCHECK0B_DONE
