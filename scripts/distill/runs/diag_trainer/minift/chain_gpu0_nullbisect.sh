#!/bin/bash
# sigma-band bisect, P1-null target: null_mid1 (sigma 1.17) -> null_low2 (0.097, 0.002) -> null_hi5 (700..7.28); each: 300-step train, hybrid e1all step300 on 0301; score at the end
set -u; cd /home/kawa/master_project/StereoCrafter; M=scripts/distill/runs/diag_trainer/minift
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
ARGS="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4"
for SPEC in "null_mid1 idx:5" "null_low2 idx:6,7" "null_hi5 idx:0,1,2,3,4"; do set -- $SPEC; NAME=$1; SIG=$2
  echo "START train $NAME ($SIG) $(date +%T)"; MINIFT_SIGMAS=$SIG python $M/xcheck_mini_ft.py null $NAME > $M/train_$NAME.log 2>&1; echo "EXIT train $NAME $? $(date +%T)"
  grep -q MINIFT_DONE $M/train_$NAME.log || { echo "TRAIN_FAILED $NAME"; continue; }
  OUTDIR=$(grep '^OUT ' $M/train_$NAME.log | awk '{print $2}'); echo "OUTDIR $OUTDIR"
  O=$M/infer/${NAME}_step300_0301; k=1; while [ -e "$O" ]; do k=$((k+1)); O=$M/infer/${NAME}_step300_0301_$k; done
  echo "START infer $NAME step300 0301 -> $O $(date +%T)"; MINIFT_CK=$OUTDIR/step300.pt python $M/xcheck_hybrid_minift.py e1all 0301 $O > $O.log 2>&1; echo "EXIT infer $NAME $? $(date +%T)"
  F=$O/0301_inpainting_results_sbs.mp4; [ -f "$F" ] && ARGS="$ARGS 0301=$F" || echo "MISSING $F"
done
echo "START score $(date +%T)"; python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $M/scores_nullbisect.txt
echo "EXIT score $(date +%T)"; echo CHAIN_NULLBISECT_DONE
