#!/bin/bash
# sigma-band bisect of the P1-null drift: null step300 tensors only at sigma 700/287/103 (e1high) or only at sigma <= 31 (e1low), clip 0301, then score
set -u; cd /home/kawa/master_project/StereoCrafter; M=scripts/distill/runs/diag_trainer/minift
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
ARGS="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4 0301=$M/infer/null_step300_0301/0301_inpainting_results_sbs.mp4"
for MODE in e1high e1low; do
  O=$M/infer/null_step300_0301_$MODE; k=1; while [ -e "$O" ]; do k=$((k+1)); O=$M/infer/null_step300_0301_${MODE}_$k; done
  echo "START infer null step300 0301 $MODE -> $O $(date +%T)"
  MINIFT_CK=$M/null/step300.pt python $M/xcheck_hybrid_minift.py $MODE 0301 $O > $O.log 2>&1; echo "EXIT infer null step300 0301 $MODE $? $(date +%T)"
  F=$O/0301_inpainting_results_sbs.mp4; [ -f "$F" ] && ARGS="$ARGS 0301=$F" || echo "MISSING $F"
done
echo "START score $(date +%T)"
python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $M/scores_nullswap.txt
echo "EXIT score $(date +%T)"; echo CHAIN_NULLSWAP_DONE
