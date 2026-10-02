#!/bin/bash
# control: random perturbation of the 15 tensors with the same per-tensor ||dW|| as null step300, sampled via the hybrid hook (e1all) on 0301, then scored
set -u; cd /home/kawa/master_project/StereoCrafter; M=scripts/distill/runs/diag_trainer/minift
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
O=$M/infer/randpert_matched_null300_0301; k=1; while [ -e "$O" ]; do k=$((k+1)); O=$M/infer/randpert_matched_null300_0301_$k; done
echo "START infer randpert 0301 -> $O $(date +%T)"
MINIFT_CK=$M/randpert/randpert_matched_null300.pt python $M/xcheck_hybrid_minift.py e1all 0301 $O > $O.log 2>&1; echo "EXIT infer randpert $? $(date +%T)"
python scripts/distill/score_clip.py 0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4 0301=$O/0301_inpainting_results_sbs.mp4 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $M/scores_randpert.txt
echo CHAIN_RANDPERT_DONE
