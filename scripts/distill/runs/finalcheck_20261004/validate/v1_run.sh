#!/bin/bash
# V1 temporal consistency, 576x1024, 12 test clips x {origin, shipped Mamba, deliverable, origin+s25}, existing FFV1 renders.
# Step 1: equivalence control -- TRACKED scripts/distill/score_temporal.py on 0204 (origin, deliv).
# Step 2: the copy score_temporal_ll.py on all 12 clips (origin first per clip).
# Writes only under outputs/finalcheck_20261004/validate/v1_temporal/ (refuses to overwrite).
set -u
cd /home/kawa/master_project/StereoCrafter
PY=/home/kawa/miniconda3/envs/stereocrafter/bin/python
V=scripts/distill/runs/finalcheck_20261004/validate
O=outputs/finalcheck_20261004/validate/v1_temporal
mkdir -p $O
export CUDA_VISIBLE_DEVICES=1
unset STEP NOFLOW GT_DIR
path_of() {  # clip method -> render path (provenance as in TABLE_HEADLINE_12CLIP.txt)
  local c=$1 m=$2
  case $m in
    origin) echo outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv ;;
    deliv)  echo outputs/beyond_distil_mamba_scaled/clips/${c}_mstudent2_step800_deliv_ll/${c}_inpainting_results_sbs.mkv ;;
    mamba)  case $c in 0052|0147|0204|0301) echo outputs/skeptic1_stack/clips/${c}_mamba_ll/${c}_inpainting_results_sbs.mkv ;;
                       *) echo outputs/beyond_distil_mamba/clips/${c}_mamba_ll/${c}_inpainting_results_sbs.mkv ;; esac ;;
    s25)    case $c in 0052|0147|0204|0301) echo outputs/beyond4_lossless/clips/${c}_s25_ll/${c}_inpainting_results_sbs.mkv ;;
                       *) echo outputs/skeptic1_stack/clips/${c}_s25_ll/${c}_inpainting_results_sbs.mkv ;; esac ;;
  esac
}
echo "V1_START $(date '+%F %T')" | tee -a $O/v1_run.log
# ---- step 1: tracked-script equivalence control on 0204 ----
if [ ! -f $O/control_tracked_0204.json ]; then
  ARGS="0204=$(path_of 0204 origin) 0204=$(path_of 0204 deliv)"
  /usr/bin/time -v $PY scripts/distill/score_temporal.py $O/control_tracked_0204.json $ARGS > $O/control_tracked_0204.log 2>&1
  echo "CONTROL_TRACKED rc=$? $(date '+%F %T')" | tee -a $O/v1_run.log
else echo "CONTROL exists, skipped" | tee -a $O/v1_run.log; fi
# ---- step 2: the copy on all 12 clips ----
if [ ! -f $O/temporal_576_12clip.json ]; then
  ARGS=""
  for c in 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301; do
    for m in origin mamba deliv s25; do p=$(path_of $c $m); [ -f "$p" ] || { echo "MISSING $p" | tee -a $O/v1_run.log; exit 2; }; ARGS="$ARGS $c=$p"; done
  done
  /usr/bin/time -v $PY $V/score_temporal_ll.py $O/temporal_576_12clip.json $ARGS > $O/temporal_576_12clip.log 2>&1
  echo "MAIN_COPY rc=$? $(date '+%F %T')" | tee -a $O/v1_run.log
else echo "MAIN exists, skipped" | tee -a $O/v1_run.log; fi
echo "V1_DONE $(date '+%F %T')" | tee -a $O/v1_run.log
