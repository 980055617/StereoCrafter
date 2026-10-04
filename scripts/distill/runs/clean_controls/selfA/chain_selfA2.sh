#!/bin/bash
# CONTROL A2 contingency chain (GPU 0, transient systemd user unit): same as chain_selfA.sh but the trainer also reads cond+mask
# from the splatting file (xcheck_mini_ft_ll_splatcond.py), so (cond, mask, target) is exactly the pipeline's own triple.
#   1. train -> llnull_splatcond   2. render 0301 step100/200/300 (e1all), 0204 step300   3. score (lossless) against this lane's
#   no-hook origin renders (infer/0301_origin_nohook_ll, infer/0204_origin_nohook_ll, produced by chain_selfA.sh).
set -u
cd /home/kawa/master_project/StereoCrafter
A=scripts/distill/runs/clean_controls/selfA
B4=scripts/distill/runs/fulldata_v2/beyond4
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 MAMBA_SELF_ATTN_INCLUDE='__nomatch__' LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
LOG=$A/chain_selfA2.log
ts() { date +%T; }
newdir() { local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=${1}_$k; done; echo "$O"; }
echo "CHAIN2_START $(ts) gpu=$CUDA_VISIBLE_DEVICES" | tee -a $LOG
echo "START train llnull_splatcond $(ts)" | tee -a $LOG
T0=$(date +%s)
$PY $A/xcheck_mini_ft_ll_splatcond.py null llnull_splatcond > $A/train_llnull_splatcond.log 2>&1
echo "EXIT train rc=$? secs=$(( $(date +%s) - T0 )) $(ts)" | tee -a $LOG
grep -q MINIFT_DONE $A/train_llnull_splatcond.log || { echo "TRAIN_FAILED (see $A/train_llnull_splatcond.log)" | tee -a $LOG; echo CHAIN_SELFA2_DONE | tee -a $LOG; exit 1; }
OUTDIR=$(grep '^OUT ' $A/train_llnull_splatcond.log | awk '{print $2}'); echo "OUTDIR $OUTDIR" | tee -a $LOG
grep -E "^\[sanity|^\[mem\]|^step 1 |^DONE" $A/train_llnull_splatcond.log | tee -a $LOG
ARGS0301="0301=$A/infer/0301_origin_nohook_ll/0301_inpainting_results_sbs.mkv"
ARGS0204="0204=$A/infer/0204_origin_nohook_ll/0204_inpainting_results_sbs.mkv"
render() {
  local MODE=$1 CLIP=$2 CK=$3 NAME=$4
  local O; O=$(newdir $A/infer/$NAME); mkdir -p $O
  echo "START render $CLIP $MODE ck=$(basename $CK) -> $O $(ts)" | tee -a $LOG
  local T0; T0=$(date +%s)
  MINIFT_CK=$CK $PY $A/hybrid_ll.py $MODE $CLIP $O > $O.log 2>&1
  echo "EXIT render $CLIP $MODE $NAME rc=$? secs=$(( $(date +%s) - T0 )) $(ts)" | tee -a $LOG
  grep -E "^\[hybrid\] (tensors|\(t, weights\))|^\[lossless\] wrote" $O.log | tee -a $LOG
  local F=$O/${CLIP}_inpainting_results_sbs.mkv
  if [ -f "$F" ]; then if [ "$CLIP" = 0301 ]; then ARGS0301="$ARGS0301 0301=$F"; else ARGS0204="$ARGS0204 0204=$F"; fi; else echo "MISSING $F" | tee -a $LOG; fi
}
for N in 300 100 200; do render e1all 0301 $OUTDIR/step$N.pt 0301_llsplat_step$N; done
render e1all 0204 $OUTDIR/step300.pt 0204_llsplat_step300
echo "START score $(ts)" | tee -a $LOG
echo "ARGS: $ARGS0301 $ARGS0204" >> $LOG
SCORE_STEP=4 $PY $B4/score_clip_ll.py $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $A/scores_selfA2.txt
echo "EXIT score rc=${PIPESTATUS[0]} rows=$(grep -c '^ROW' $A/scores_selfA2.txt) $(ts)" | tee -a $LOG
echo CHAIN_SELFA2_DONE | tee -a $LOG
