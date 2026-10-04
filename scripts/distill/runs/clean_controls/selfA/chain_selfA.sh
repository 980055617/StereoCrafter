#!/bin/bash
# CONTROL A chain on GPU 0 (run as a transient systemd user unit):
#   0. origin reference render, NO hook, beyond4/infer_lossless.py path (must reproduce pre-encode md5 2e533d7755c950d2fc95043f6fb0a51d on 0301)
#   1. train xcheck_mini_ft_ll.py null llnull  (lossless target; 300 steps, ckpt 100/200/300)
#   2. renders through hybrid_ll.py (FFV1): 0301 originall (hook no-op control), step100/200/300 e1all, step300 e1high / e1low;
#      0204 origin (no hook) + step100/200/300 e1all
#   3. score everything with beyond4/score_clip_ll.py (SCORE_STEP=4), lossless path only
# Every output goes to a NEW directory under clean_controls/selfA; nothing existing is overwritten.
set -u
cd /home/kawa/master_project/StereoCrafter
A=scripts/distill/runs/clean_controls/selfA
B4=scripts/distill/runs/fulldata_v2/beyond4
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 MAMBA_SELF_ATTN_INCLUDE='__nomatch__' LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
LOG=$A/chain_selfA.log
ts() { date +%T; }
newdir() { local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=${1}_$k; done; echo "$O"; }
echo "CHAIN_START $(ts) gpu=$CUDA_VISIBLE_DEVICES" | tee -a $LOG

# ---- 0. origin reference, no hook (0301) ----
O=$(newdir $A/infer/0301_origin_nohook_ll); mkdir -p $O
echo "START render 0301 origin nohook -> $O $(ts)" | tee -a $LOG
T0=$(date +%s)
$PY $B4/infer_lossless.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None --num_inference_steps=8 \
   --min_guidance_scale=1.01 --max_guidance_scale=1.01 --input_video_path=video_data/splatting/0301_splatting_results.mp4 --save_dir=$O > $O.log 2>&1
echo "EXIT render 0301 origin nohook rc=$? secs=$(( $(date +%s) - T0 )) $(ts)" | tee -a $LOG
cat $O/writer_md5.txt 2>/dev/null | tee -a $LOG
ARGS0301="0301=outputs/beyond4_lossless/clips/0301_origin_ll/0301_inpainting_results_sbs.mkv"
F=$O/0301_inpainting_results_sbs.mkv; [ -f "$F" ] && ARGS0301="$ARGS0301 0301=$F" || echo "MISSING $F" | tee -a $LOG

# ---- 1. train ----
echo "START train llnull $(ts)" | tee -a $LOG
T0=$(date +%s)
$PY $A/xcheck_mini_ft_ll.py null llnull > $A/train_llnull.log 2>&1
echo "EXIT train rc=$? secs=$(( $(date +%s) - T0 )) $(ts)" | tee -a $LOG
grep -q MINIFT_DONE $A/train_llnull.log || { echo "TRAIN_FAILED (see $A/train_llnull.log)" | tee -a $LOG; echo CHAIN_SELFA_DONE | tee -a $LOG; exit 1; }
OUTDIR=$(grep '^OUT ' $A/train_llnull.log | awk '{print $2}'); echo "OUTDIR $OUTDIR" | tee -a $LOG
grep -E "^\[sanity|^\[mem\]|^step 1 |^DONE" $A/train_llnull.log | tee -a $LOG

# ---- 2. renders through the hook (FFV1) ----
render() {   # render <mode> <clip> <ck> <dirname> ; appends to ARGS<clip>
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
render originall 0301 $OUTDIR/step300.pt 0301_originall_hook_ll
for N in 100 200 300; do render e1all 0301 $OUTDIR/step$N.pt 0301_llnull_step$N; done
render e1high 0301 $OUTDIR/step300.pt 0301_llnull_step300_e1high
render e1low  0301 $OUTDIR/step300.pt 0301_llnull_step300_e1low

# held-out 0204: origin (no hook) + the three checkpoints
O=$(newdir $A/infer/0204_origin_nohook_ll); mkdir -p $O
echo "START render 0204 origin nohook -> $O $(ts)" | tee -a $LOG
T0=$(date +%s)
$PY $B4/infer_lossless.py --config=config/0160_overfit_inference_matched.json --unet_state_path=None --num_inference_steps=8 \
   --min_guidance_scale=1.01 --max_guidance_scale=1.01 --input_video_path=video_data/splatting/0204_splatting_results.mp4 --save_dir=$O > $O.log 2>&1
echo "EXIT render 0204 origin nohook rc=$? secs=$(( $(date +%s) - T0 )) $(ts)" | tee -a $LOG
cat $O/writer_md5.txt 2>/dev/null | tee -a $LOG
ARGS0204="0204=outputs/beyond4_lossless/clips/0204_origin_ll/0204_inpainting_results_sbs.mkv"
F=$O/0204_inpainting_results_sbs.mkv; [ -f "$F" ] && ARGS0204="$ARGS0204 0204=$F" || echo "MISSING $F" | tee -a $LOG
for N in 300 100 200; do render e1all 0204 $OUTDIR/step$N.pt 0204_llnull_step$N; done

# ---- 3. score (lossless path only) ----
echo "START score $(ts)" | tee -a $LOG
echo "ARGS: $ARGS0301 $ARGS0204" >> $LOG
SCORE_STEP=4 $PY $B4/score_clip_ll.py $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $A/scores_selfA.txt
echo "EXIT score rc=${PIPESTATUS[0]} rows=$(grep -c '^ROW' $A/scores_selfA.txt) $(ts)" | tee -a $LOG
echo CHAIN_SELFA_DONE | tee -a $LOG
