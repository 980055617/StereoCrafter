#!/bin/bash
# sigma-band bisect, P1-pos target (real GT 0301): pos_hi5 (700..7.28) -> pos_lognormal (ln sigma ~ N(0.7,1.6)); each: 300-step train, hybrid e1all step300 on 0301 and 0042; score at the end
set -u; cd /home/kawa/master_project/StereoCrafter; M=scripts/distill/runs/diag_trainer/minift
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
ARGS0301="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4"; ARGS0042="0042=outputs/fulldata_v2/clips/0042_origin/0042_inpainting_results_sbs.mp4"
for SPEC in "pos_hi5 idx:0,1,2,3,4" "pos_lognormal lognormal:0.7:1.6"; do set -- $SPEC; NAME=$1; SIG=$2
  echo "START train $NAME ($SIG) $(date +%T)"; MINIFT_SIGMAS=$SIG python $M/xcheck_mini_ft.py pos $NAME > $M/train_$NAME.log 2>&1; echo "EXIT train $NAME $? $(date +%T)"
  grep -q MINIFT_DONE $M/train_$NAME.log || { echo "TRAIN_FAILED $NAME"; continue; }
  OUTDIR=$(grep '^OUT ' $M/train_$NAME.log | awk '{print $2}'); echo "OUTDIR $OUTDIR"
  for CLIP in 0301 0042; do
    O=$M/infer/${NAME}_step300_$CLIP; k=1; while [ -e "$O" ]; do k=$((k+1)); O=$M/infer/${NAME}_step300_${CLIP}_$k; done
    echo "START infer $NAME step300 $CLIP -> $O $(date +%T)"; MINIFT_CK=$OUTDIR/step300.pt python $M/xcheck_hybrid_minift.py e1all $CLIP $O > $O.log 2>&1; echo "EXIT infer $NAME $CLIP $? $(date +%T)"
    F=$O/${CLIP}_inpainting_results_sbs.mp4; if [ -f "$F" ]; then if [ $CLIP = 0301 ]; then ARGS0301="$ARGS0301 0301=$F"; else ARGS0042="$ARGS0042 0042=$F"; fi; else echo "MISSING $F"; fi
  done
done
echo "START score $(date +%T)"; python scripts/distill/score_clip.py $ARGS0301 $ARGS0042 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $M/scores_posbisect.txt
echo "EXIT score $(date +%T)"; echo CHAIN_POSBISECT_DONE
