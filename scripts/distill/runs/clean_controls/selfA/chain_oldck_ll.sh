#!/bin/bash
# Like-for-like control: render the OLD mp4v-target P1-null checkpoints (minift/null/step{100,200,300}.pt, untouched) through the
# same hook + FFV1 path and score them losslessly, so "fraction of the old drift remaining" compares lossless-vs-lossless
# instead of lossless-vs-mp4v-scored.  0301 steps 100/200/300 + held-out 0204 step300.  GPU 0, new dirs only.
set -u
cd /home/kawa/master_project/StereoCrafter
A=scripts/distill/runs/clean_controls/selfA
B4=scripts/distill/runs/fulldata_v2/beyond4
OLD=scripts/distill/runs/diag_trainer/minift/null
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 MAMBA_SELF_ATTN_INCLUDE='__nomatch__' LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
LOG=$A/chain_oldck_ll.log
ts() { date +%T; }
newdir() { local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=${1}_$k; done; echo "$O"; }
echo "CHAIN_OLDCK_START $(ts)" | tee -a $LOG
ARGS0301="0301=$A/infer/0301_origin_nohook_ll/0301_inpainting_results_sbs.mkv"
ARGS0204="0204=$A/infer/0204_origin_nohook_ll/0204_inpainting_results_sbs.mkv"
render() {
  local MODE=$1 CLIP=$2 CK=$3 NAME=$4
  local O; O=$(newdir $A/infer/$NAME); mkdir -p $O
  echo "START render $CLIP $MODE ck=$CK -> $O $(ts)" | tee -a $LOG
  local T0; T0=$(date +%s)
  MINIFT_CK=$CK $PY $A/hybrid_ll.py $MODE $CLIP $O > $O.log 2>&1
  echo "EXIT render $CLIP $MODE $NAME rc=$? secs=$(( $(date +%s) - T0 )) $(ts)" | tee -a $LOG
  grep -E "^\[hybrid\] (tensors|\(t, weights\))|^\[lossless\] wrote" $O.log | tee -a $LOG
  local F=$O/${CLIP}_inpainting_results_sbs.mkv
  if [ -f "$F" ]; then if [ "$CLIP" = 0301 ]; then ARGS0301="$ARGS0301 0301=$F"; else ARGS0204="$ARGS0204 0204=$F"; fi; else echo "MISSING $F" | tee -a $LOG; fi
}
for N in 300 100 200; do render e1all 0301 $OLD/step$N.pt 0301_oldmp4vnull_step$N; done
render e1all 0204 $OLD/step300.pt 0204_oldmp4vnull_step300
echo "START score $(ts)" | tee -a $LOG
echo "ARGS: $ARGS0301 $ARGS0204" >> $LOG
SCORE_STEP=4 $PY $B4/score_clip_ll.py $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $A/scores_oldck_ll.txt
echo "EXIT score rc=${PIPESTATUS[0]} rows=$(grep -c '^ROW' $A/scores_oldck_ll.txt) $(ts)" | tee -a $LOG
echo CHAIN_OLDCK_DONE | tee -a $LOG
