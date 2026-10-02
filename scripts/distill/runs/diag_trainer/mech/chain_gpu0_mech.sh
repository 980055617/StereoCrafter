#!/bin/bash
# GPU 0 chain.  (1) TEST A: for each on-trajectory variant, 300-step train then deployed 8-step inference on 0301
# with the trained tensors swapped in.  (2) TEST B tiebreaker: 1-step sampler for origin / null300 / pos300.
# Every inference writes a NEW directory (suffix bumped if it exists).  One score_clip pass at the end.
set -u; cd /home/kawa/master_project/StereoCrafter
H=scripts/distill/runs/diag_trainer/mech; MF=scripts/distill/runs/diag_trainer/minift
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
mkdir -p $H/infer
ARGS="0301=outputs/fulldata_v2/clips/0301_origin/0301_inpainting_results_sbs.mp4"
newdir(){ local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=$1_$k; done; echo $O; }

# ---- TEST A ----------------------------------------------------------------
#   name      variant  step-subset          checkpoints to score
run_variant(){ NAME=$1; V=$2; SUBSET=$3; CKS=$4
  echo "START train $NAME (variant $V, traj steps $SUBSET) $(date +%T)"
  MECH_STEP_SUBSET=$SUBSET python $H/xcheck_traj_ft.py $V $NAME > $H/train_$NAME.log 2>&1
  echo "EXIT train $NAME $? $(date +%T)"
  grep -q MECH_TRAJ_DONE $H/train_$NAME.log || { echo "TRAIN_FAILED $NAME"; return; }
  OUTDIR=$(grep '^OUT ' $H/train_$NAME.log | awk '{print $2}'); echo "OUTDIR $OUTDIR"
  for S in $CKS; do
    O=$(newdir $H/infer/${NAME}_step${S}_0301)
    echo "START infer $NAME step$S -> $O $(date +%T)"
    MINIFT_CK=$OUTDIR/step$S.pt python $MF/xcheck_hybrid_minift.py e1all 0301 $O > $O.log 2>&1
    echo "EXIT infer $NAME step$S $? $(date +%T)"
    F=$O/0301_inpainting_results_sbs.mp4; [ -f "$F" ] && ARGS="$ARGS 0301=$F" || echo "MISSING $F"
  done
}
run_variant a1      a1   0,1,2,3,4,5,6,7  "300"
run_variant a2      a2   0,1,2,3,4,5,6,7  "100 200 300"
run_variant a2x0    a2x0 0,1,2,3,4,5,6,7  "100 200 300"
run_variant a2_hi5  a2   0,1,2,3,4        "100 200 300"

# ---- TEST B tiebreaker: 1-step sampler -------------------------------------
for SPEC in "origin1 originall $MF/null/step300.pt" "null300_1 e1all $MF/null/step300.pt" "pos300_1 e1all $MF/pos/step300.pt"; do
  set -- $SPEC; NAME=$1; MODE=$2; CK=$3
  O=$(newdir $H/infer/${NAME}step_0301)
  echo "START infer1 $NAME -> $O $(date +%T)"
  MECH_STEPS=1 MINIFT_CK=$CK python $H/hybrid_nsteps.py $MODE 0301 $O > $O.log 2>&1
  echo "EXIT infer1 $NAME $? $(date +%T)"
  F=$O/0301_inpainting_results_sbs.mp4; [ -f "$F" ] && ARGS="$ARGS 0301=$F" || echo "MISSING $F"
done

echo "START score $(date +%T)"
python scripts/distill/score_clip.py $ARGS 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|/home/kawa" | tee $H/scores_mech.txt
echo "EXIT score $? $(date +%T)"; echo CHAIN_MECH_DONE
