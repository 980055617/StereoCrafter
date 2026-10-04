#!/bin/bash
# CONTROL B render + score chain (GPU 1, FFV1 writer, score_clip_ll.py SCORE_STEP=4). Every render -> a NEW dir under regB/infer/.
# usage: chain_regB_v1.sh <regB train run dir>     (e.g. scripts/distill/runs/clean_controls/regB/train/regpos)
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
B=scripts/distill/runs/clean_controls/regB; I=$B/infer; S=$B/scores; RUN=$1
POS=scripts/distill/runs/diag_trainer/minift/pos          # the UNregistered-GT P1 run (existing checkpoints, read-only)
SCORER=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
ORIG_MD5_0301=2e533d7755c950d2fc95043f6fb0a51d
newdir() { local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=$1_$k; done; echo $O; }
render() { # render <tag> <mode> <clip> <ckpt>
  local TAG=$1 MODE=$2 CLIP=$3 CK=$4; local O; O=$(newdir $I/$TAG)
  echo "START $TAG mode=$MODE clip=$CLIP ck=$CK -> $O $(date +%T)"
  MINIFT_CK=$CK python $I/xcheck_hybrid_regB_ll.py $MODE $CLIP $O > $O.log 2>&1; local RC=$?
  echo "EXIT $TAG rc=$RC $(date +%T)"
  if [ $RC -ne 0 ] || [ ! -f $O/${CLIP}_inpainting_results_sbs.mkv ]; then
    echo "RETRY $TAG (once)"; O=$(newdir $I/$TAG); MINIFT_CK=$CK python $I/xcheck_hybrid_regB_ll.py $MODE $CLIP $O > $O.log 2>&1; RC=$?; echo "EXIT-RETRY $TAG rc=$RC $(date +%T)"
  fi
  LAST_OUT=$O
}
ARGS0301=""; ARGS0204=""
add() { local CLIP=$1 DIR=$2; local F=$DIR/${CLIP}_inpainting_results_sbs.mkv; if [ -f "$F" ]; then if [ $CLIP = 0301 ]; then ARGS0301="$ARGS0301 0301=$F"; else ARGS0204="$ARGS0204 0204=$F"; fi; else echo "MISSING $F"; fi; }
ARGS0301="0301=outputs/beyond4_lossless/clips/0301_origin_ll/0301_inpainting_results_sbs.mkv"
ARGS0204="0204=outputs/beyond4_lossless/clips/0204_origin_ll/0204_inpainting_results_sbs.mkv"

# ---- phase A ----
render regB_originall_0301 originall 0301 $RUN/step100.pt; add 0301 $LAST_OUT
MD5=$(grep "_sbs" $LAST_OUT/writer_md5.txt | awk '{print $1}')
if [ "$MD5" = "$ORIG_MD5_0301" ]; then echo "FAITHFULNESS_OK originall pre-encode md5 $MD5 == beyond4 0301_origin_ll"; else echo "FAITHFULNESS_FAIL originall md5 $MD5 != $ORIG_MD5_0301"; fi
for N in 100 200 300; do render regB_step${N}_0301 e1all 0301 $RUN/step$N.pt; add 0301 $LAST_OUT; done
render regB_step300_0204 e1all 0204 $RUN/step300.pt; add 0204 $LAST_OUT
for N in 300 100 200; do render posUnreg_step${N}_0301 e1all 0301 $POS/step$N.pt; add 0301 $LAST_OUT; done
echo "START score phaseA $(date +%T)"
python $SCORER $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $S/scores_regB_phaseA.txt
BEST=$(grep -E '^ROW clip=0301 tag=regB_step[0-9]+_0301 ' $S/scores_regB_phaseA.txt | sed -E 's/.*tag=regB_step([0-9]+)_0301.*lpips=([0-9.]+).*/\2 \1/' | sort -n | head -1 | awk '{print $2}')
echo "BEST_STEP_0301_BY_SAMPLED_LPIPS $BEST"

# ---- phase B ----
if [ "$BEST" != "300" ]; then render regB_step${BEST}_0204 e1all 0204 $RUN/step$BEST.pt; add 0204 $LAST_OUT; fi
for MODE in e1high e1low; do render regB_step${BEST}_0301_$MODE $MODE 0301 $RUN/step$BEST.pt; add 0301 $LAST_OUT; done
echo "START score final $(date +%T)"
python $SCORER $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $S/scores_regB_final.txt
echo "START score registered-GT diagnostic $(date +%T)"
python $S/score_clip_ll_reg.py $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $S/scores_regB_regdiag.txt
echo "CHAIN_REGB_DONE $(date +%T)"
