#!/bin/bash
# Score M2SVid renders (PREREG_r2.txt): per clip, UNREG (score_clip_ll.py verbatim, origin_ll re-scored in the same call)
# -> rows file (gate: origin reproduces its published lpips) -> REGISTERED (copy of score_registered_v1.py).
# Each GPU step holds the GPU-0 lock only for itself.  usage: bash score_chain_r2.sh <label> <reg_out_dir> <clip>...
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
R=scripts/distill/runs/deep_20261004/external_models
LABEL=$1; REG=$2; shift 2
mkdir -p "$REG" $R/rows
UN=$R/SCORES_UNREG_${LABEL}.txt
for c in "$@"; do
  MKV=outputs/deep_20261004/external_models/clips/${c}_${LABEL}/${c}_inpainting_results_sbs.mkv
  ORI=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv
  if [ ! -s "$MKV" ]; then echo "MISSING $MKV" >> $UN; continue; fi
  echo "### $c $(date '+%F_%T')" >> $UN
  CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 flock /tmp/claude-gpu0.lock \
    python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py "$c=$ORI" "$c=$MKV" >> $UN 2>&1
  echo "UNREG_RC $c rc=$? $(date '+%F_%T')" >> $UN
  python $R/build_rows_r2.py "$c" "$LABEL" $UN $R/rows/${c}_${LABEL}.json >> $UN 2>&1 || { echo "ROWS_FAIL $c" >> $UN; continue; }
  CUDA_VISIBLE_DEVICES=0 SCORE_STEP=4 flock /tmp/claude-gpu0.lock \
    python $R/score_registered_extm_r2.py "$REG" $R/rows/${c}_${LABEL}.json "$c" >> $R/score_reg_${LABEL}.log 2>&1
  echo "REG_RC $c rc=$? $(date '+%F_%T')" >> $R/score_reg_${LABEL}.log
done
echo "CHAIN_END $(date '+%F_%T')" >> $UN
