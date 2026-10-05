#!/bin/bash
# Batched scoring (PREREG_r2.txt) meant to run INSIDE ONE GPU-0 lock hold taken by the caller
# (flock /tmp/claude-gpu0.lock bash score_batch_r2.sh ...): same three steps and same scripts as score_chain_r2.sh
# (UNREG score_clip_ll.py verbatim with origin_ll re-scored -> rows + gate -> registered copy) plus NR, without
# re-acquiring the contended lock per step.  usage: bash score_batch_r2.sh <tag> <clip>...
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4
R=scripts/distill/runs/deep_20261004/external_models
O=outputs/deep_20261004/external_models
TAG=$1; shift
LABEL=${TAG}_ll
REG=$O/score_reg_${TAG}_r2
NR=$O/nr_${TAG}_r2
UN=$R/SCORES_UNREG_${LABEL}.txt
LOG=$R/score_batch_${TAG}_r2.log
mkdir -p "$REG" "$NR" $R/rows
echo "BATCH_START $(date '+%F_%T') clips=[$*]" >> $LOG
SPECS=()
for c in "$@"; do
  M=$O/clips/${c}_${LABEL}/${c}_inpainting_results_sbs.mkv
  [ -s "$M" ] || { echo "MISSING $M" >> $LOG; continue; }
  SPECS+=("$c=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv" "$c=$M")
done
echo "### batch $(date '+%F_%T') $*" >> $UN
python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py "${SPECS[@]}" >> $UN 2>&1
echo "UNREG_RC rc=$? $(date '+%F_%T')" >> $LOG
for c in "$@"; do
  M=$O/clips/${c}_${LABEL}/${c}_inpainting_results_sbs.mkv
  [ -s "$M" ] || continue
  [ -e $R/rows/${c}_${LABEL}.json ] || python $R/build_rows_r2.py "$c" "$LABEL" $UN $R/rows/${c}_${LABEL}.json >> $LOG 2>&1 \
    || { echo "ROWS_FAIL $c" >> $LOG; continue; }
  [ -e $REG/$c.json ] || python $R/score_registered_extm_r2.py "$REG" $R/rows/${c}_${LABEL}.json "$c" >> $LOG 2>&1
  echo "REG_RC $c rc=$? $(date '+%F_%T')" >> $LOG
  [ -e $NR/$c.json ] || PYTHONPATH=/mnt/ssd_data/deep_20261004/external_models/pylib_iqa \
    TORCH_HOME=/mnt/ssd_data/deep_20261004/external_models/torch_home HF_HUB_OFFLINE=1 \
    python $R/nr_metrics_r2.py "$NR" "$c" origin_ll=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv \
    deliv=outputs/beyond_distil_mamba_scaled/clips/${c}_mstudent2_step800_deliv_ll/${c}_inpainting_results_sbs.mkv \
    m2svid=$M >> $LOG 2>&1
  echo "NR_RC $c rc=$? $(date '+%F_%T')" >> $LOG
done
echo "BATCH_END $(date '+%F_%T')" >> $LOG
