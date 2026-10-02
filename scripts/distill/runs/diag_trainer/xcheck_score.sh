#!/bin/bash
set -u; cd /home/kawa/master_project/StereoCrafter; D=scripts/distill/runs/diag_trainer
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
until grep -q XCHECK0_DONE $D/xcheck_gpu0_chain.log && grep -q XCHECK1_DONE $D/xcheck_gpu1_fc2.log; do sleep 15; done
ARGS="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4 0301=outputs/fulldata_v2/clips/0301_originattn_e001/0301_inpainting_results_sbs.mp4"
for L in e001_condx0.18215 origin_condx0.18215 hyb_originall hyb_e1all hyb_e1high hyb_e1low hyb_e1up3 hyb_e1down0 e001_fc2_condx1 origin_fc2_condx1 e001_fc2_condx0.18215; do F=$D/infer/0301_$L/0301_inpainting_results_sbs.mp4; [ -f $F ] && ARGS="$ARGS 0301=$F" || echo "MISSING $L"; done
CUDA_VISIBLE_DEVICES=0 python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $D/xcheck_scores_0301.txt
echo SCORE_DONE
