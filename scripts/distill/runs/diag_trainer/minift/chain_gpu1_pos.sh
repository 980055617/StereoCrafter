#!/bin/bash
# P1-pos chain on GPU1: train -> hybrid e1all inference (0301 steps 100/200/300, 0042 steps 100/200/300) -> score
set -u; cd /home/kawa/master_project/StereoCrafter; M=scripts/distill/runs/diag_trainer/minift
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
V=pos
echo "START train $V $(date +%T)"; python $M/xcheck_mini_ft.py $V > $M/train_$V.log 2>&1; echo "EXIT train $V $? $(date +%T)"
grep -q MINIFT_DONE $M/train_$V.log || { echo TRAIN_FAILED; echo CHAIN_POS_DONE; exit 1; }
OUTDIR=$(grep '^OUT ' $M/train_$V.log | awk '{print $2}'); echo "OUTDIR $OUTDIR"; mkdir -p $M/infer
ARGS0301="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4"; ARGS0042="0042=outputs/fulldata_v2/clips/0042_origin/0042_inpainting_results_sbs.mp4"
for CLIP in 0301 0042; do for N in 100 200 300; do
  O=$M/infer/${V}_step${N}_${CLIP}; k=1; while [ -e "$O" ]; do k=$((k+1)); O=$M/infer/${V}_step${N}_${CLIP}_$k; done
  echo "START infer $V step$N $CLIP -> $O $(date +%T)"
  MINIFT_CK=$OUTDIR/step$N.pt python $M/xcheck_hybrid_minift.py e1all $CLIP $O > $O.log 2>&1; echo "EXIT infer $V step$N $CLIP $? $(date +%T)"
  F=$O/${CLIP}_inpainting_results_sbs.mp4; if [ -f "$F" ]; then if [ $CLIP = 0301 ]; then ARGS0301="$ARGS0301 0301=$F"; else ARGS0042="$ARGS0042 0042=$F"; fi; else echo "MISSING $F"; fi
done; done
for MODE in e1high e1low; do
  O=$M/infer/${V}_step300_0301_$MODE; k=1; while [ -e "$O" ]; do k=$((k+1)); O=$M/infer/${V}_step300_0301_${MODE}_$k; done
  echo "START infer $V step300 0301 $MODE -> $O $(date +%T)"
  MINIFT_CK=$OUTDIR/step300.pt python $M/xcheck_hybrid_minift.py $MODE 0301 $O > $O.log 2>&1; echo "EXIT infer $V step300 0301 $MODE $? $(date +%T)"
  F=$O/0301_inpainting_results_sbs.mp4; [ -f "$F" ] && ARGS0301="$ARGS0301 0301=$F" || echo "MISSING $F"
done
echo "START score $(date +%T)"
python scripts/distill/score_clip.py $ARGS0301 $ARGS0042 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $M/scores_$V.txt
echo "EXIT score $(date +%T)"; echo CHAIN_POS_DONE
