#!/bin/bash
# V2 scoring (GPU 1).  S1 reproduction (old mp4v Full-HD render vs candidate GT dirs), S2 tracked-vs-LL scorer
# equivalence at hi-res, then the 16 hi-res lossless renders with score_clip_ll.py SCORE_STEP=4.
# Writes only under outputs/finalcheck_20261004/validate/v2_scores/ (refuses to overwrite an existing file).
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
V=scripts/distill/runs/finalcheck_20261004/validate
R=outputs/finalcheck_20261004/validate/clips
O=outputs/finalcheck_20261004/validate/v2_scores
mkdir -p $O
export CUDA_VISIBLE_DEVICES=1 SCORE_STEP=4
unset GT_DIR STEP
OLD=outputs/fulldata/fullhd/0204_origin/0204_inpainting_results_sbs.mp4
r() { echo $R/$1_$2_ll_$3/$1_inpainting_results_sbs.mkv; }   # clip model HxW
run() {  # outfile cmd...
  local f=$1; shift
  if [ -e "$f" ]; then echo "SKIP $f exists" | tee -a $O/v2_score.log; return; fi
  "$@" > "$f" 2>&1; echo "DONE rc=$? $(date '+%F %T') $f" | tee -a $O/v2_score.log
}
echo "V2_SCORE_START $(date '+%F %T')" | tee -a $O/v2_score.log
# ---- S1: reproduction of fullhd.txt row "0204_origin (-28,0) 48.63 0.1882 0.0057" ----
for GD in video_data/train video_data/train_leftGT_broken video_data/test_train; do
  tag=$(basename $GD)
  run $O/S1_old0204_gt_${tag}.txt env GT_DIR=$GD $PY $V/score_clip_ll_gtdir.py 0204=$OLD
done
# ---- S2: tracked score_clip.py vs score_clip_ll.py on identical inputs (old mp4v render + 2 new FFV1 renders) ----
S2ARGS="0204=$OLD 0204=$(r 0204 origin 1024x1920) 0042=$(r 0042 deliv 1024x1792)"
run $O/S2_tracked_score_clip.txt $PY scripts/distill/score_clip.py $S2ARGS
run $O/S2_score_clip_ll.txt $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $S2ARGS
# ---- V2 main: 16 renders, grouped per clip (GT tile loaded once per clip) ----
ARGS=""
for c in 0170 0204 0042 0052; do
  for res in 1024x1792 1024x1920; do
    for m in origin deliv; do p=$(r $c $m $res); [ -f "$p" ] && ARGS="$ARGS $c=$p" || echo "MISSING $p" | tee -a $O/v2_score.log; done
  done
done
run $O/V2_hires_scores.txt $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS
# ---- same-clip 576x1024 re-score of the EXISTING origin_ll / deliverable renders (reproduces the headline table rows) ----
A576=""
for c in 0170 0204 0042 0052; do
  A576="$A576 $c=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv"
  A576="$A576 $c=outputs/beyond_distil_mamba_scaled/clips/${c}_mstudent2_step800_deliv_ll/${c}_inpainting_results_sbs.mkv"
done
run $O/V2_576_rescore.txt $PY scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $A576
echo "V2_SCORE_DONE $(date '+%F %T')" | tee -a $O/v2_score.log
