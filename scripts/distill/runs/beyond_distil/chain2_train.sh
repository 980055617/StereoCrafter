#!/bin/bash
# CHAIN 2 (GPU 1): SMOKE train (2 clips, <=800 steps) then, at every scored checkpoint, sample the DEPLOYED
# config (8 steps, guidance 1.01) through the hybrid hook and score against deployed origin + s25.
# Every inference writes a NEW directory.  Config comes from the environment so this file is never edited.
#   BD_NAME (out subdir)  BD_CLIPS  BD_STEP_SUBSET  BD_WEIGHTS  BD_M  BD_TRAIN_STEPS  BD_LR  BD_SAVE
#   BD_SCORE_CKS (space list of checkpoint step numbers to sample)  BD_EVAL_CLIPS  BD_CK (warm start)
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
H=scripts/distill/runs/beyond_distil; MF=scripts/distill/runs/diag_trainer/minift
newdir(){ local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=$1_$k; done; echo $O; }
NAME=${BD_NAME:-smoke}
SCK=${BD_SCORE_CKS:-"200 400 800"}
ECL=${BD_EVAL_CLIPS:-"0301 0204"}

echo "START train $NAME $(date +%T)"
python $H/train_beyond.py $NAME > $H/train_$NAME.log 2>&1
echo "EXIT train $NAME $? $(date +%T)"
grep -q BD_TRAIN_DONE $H/train_$NAME.log || { echo "TRAIN_FAILED $NAME"; tail -30 $H/train_$NAME.log; exit 1; }
OUTDIR=$(grep '^OUT ' $H/train_$NAME.log | awk '{print $2}'); echo "OUTDIR $OUTDIR"

ARGS=""
for C in $ECL; do
  ARGS="$ARGS $C=outputs/fulldata_v2/clips/${C}_origin/${C}_inpainting_results_sbs.mp4"
  ARGS="$ARGS $C=outputs/fulldata_v2/clips/${C}_origin_s25/${C}_inpainting_results_sbs.mp4"
  for S in $SCK; do
    [ -f "$OUTDIR/step$S.pt" ] || { echo "NO CK $OUTDIR/step$S.pt"; continue; }
    O=$(newdir outputs/beyond_distil/${C}_${NAME}_step${S})
    echo "START infer $C $NAME step$S -> $O $(date +%T)"
    MINIFT_CK=$OUTDIR/step$S.pt python $MF/xcheck_hybrid_minift.py e1all $C $O > ${O}.log 2>&1
    echo "EXIT infer $C step$S $? $(date +%T)"
    F=$O/${C}_inpainting_results_sbs.mp4
    [ -f "$F" ] && ARGS="$ARGS $C=$F" || echo "MISSING $F"
  done
done
echo "START score $(date +%T)"
python scripts/distill/score_clip.py $ARGS 2>&1 \
  | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $H/scores_${NAME}.txt
echo "EXIT score $? $(date +%T)"; echo CHAIN2_DONE
