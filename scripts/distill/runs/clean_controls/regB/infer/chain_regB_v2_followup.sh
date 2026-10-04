#!/bin/bash
# CONTROL B follow-up chain for a second training run (e.g. the per-frame-registered regpos_pf): renders steps 100/200/300 on 0301,
# picks the best step by SAMPLED LPIPS, renders it on held-out 0204, then scores everything (standard scorer + registered-GT diagnostic).
# usage: chain_regB_v2_followup.sh <train run dir> <tag prefix>      (GPU 1, FFV1 writer, SCORE_STEP=4; every render -> NEW dir)
set -u; cd /home/kawa/master_project/StereoCrafter
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4 LOSSLESS_SBS=1 KEEP_ANAGLYPH=0
B=scripts/distill/runs/clean_controls/regB; I=$B/infer; S=$B/scores; RUN=$1; P=$2
SCORER=scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py
newdir() { local O=$1; local k=1; while [ -e "$O" ]; do k=$((k+1)); O=$1_$k; done; echo $O; }
render() { local TAG=$1 MODE=$2 CLIP=$3 CK=$4; local O; O=$(newdir $I/$TAG)
  echo "START $TAG mode=$MODE clip=$CLIP ck=$CK -> $O $(date +%T)"
  MINIFT_CK=$CK python $I/xcheck_hybrid_regB_ll.py $MODE $CLIP $O > $O.log 2>&1; local RC=$?; echo "EXIT $TAG rc=$RC $(date +%T)"
  if [ $RC -ne 0 ] || [ ! -f $O/${CLIP}_inpainting_results_sbs.mkv ]; then echo "RETRY $TAG (once)"; O=$(newdir $I/$TAG); MINIFT_CK=$CK python $I/xcheck_hybrid_regB_ll.py $MODE $CLIP $O > $O.log 2>&1; RC=$?; echo "EXIT-RETRY $TAG rc=$RC $(date +%T)"; fi
  LAST_OUT=$O; }
ARGS0301="0301=outputs/beyond4_lossless/clips/0301_origin_ll/0301_inpainting_results_sbs.mkv"; ARGS0204="0204=outputs/beyond4_lossless/clips/0204_origin_ll/0204_inpainting_results_sbs.mkv"
add() { local CLIP=$1 DIR=$2; local F=$DIR/${CLIP}_inpainting_results_sbs.mkv; if [ -f "$F" ]; then if [ $CLIP = 0301 ]; then ARGS0301="$ARGS0301 0301=$F"; else ARGS0204="$ARGS0204 0204=$F"; fi; else echo "MISSING $F"; fi; }
for N in 100 200 300; do render ${P}_step${N}_0301 e1all 0301 $RUN/step$N.pt; add 0301 $LAST_OUT; done
echo "START score 0301 $(date +%T)"
python $SCORER $ARGS0301 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $S/scores_${P}_0301.txt
BEST=$(grep -E "^ROW clip=0301 tag=${P}_step[0-9]+_0301 " $S/scores_${P}_0301.txt | sed -E "s/.*tag=${P}_step([0-9]+)_0301.*lpips=([0-9.]+).*/\2 \1/" | sort -n | head -1 | awk '{print $2}')
echo "BEST_STEP_0301_BY_SAMPLED_LPIPS $BEST"
render ${P}_step${BEST}_0204 e1all 0204 $RUN/step$BEST.pt; add 0204 $LAST_OUT
echo "START score final $(date +%T)"
python $SCORER $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $S/scores_${P}_final.txt
python $S/score_clip_ll_reg.py $ARGS0301 $ARGS0204 2>&1 | grep -viE "warning|setting up|loading model|self.load_state|^/home/kawa" | tee $S/scores_${P}_regdiag.txt
echo "CHAIN_FOLLOWUP_DONE $(date +%T)"
