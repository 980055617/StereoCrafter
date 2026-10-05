#!/bin/bash
# DIAGNOSTIC: UNREG scores of the colour-removed dev renders (color_remove_v1.py), one call per label, each under the GPU-1 lock.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
O=outputs/deep_20261004/scale_gt/diag_colorremoved_v1
export CUDA_VISIBLE_DEVICES=1
for lab in main_v1_s500_s8 main_v1_s1000_s8 contrast8_v1_s500_s8; do
  F=$L/score_dev_${lab}_colrm.txt; [ -e $F ] && { echo "SKIP $F exists"; continue; }
  ARGS=""; for c in 0040 0082 0091 0184 0245 0268; do ARGS="$ARGS $c=$O/${c}_${lab}_colrm/${c}_inpainting_results_sbs.mkv"; done
  SCORE_STEP=4 flock /tmp/claude-gpu1.lock python scripts/distill/runs/fulldata_v2/beyond4/score_clip_ll.py $ARGS > $F 2>&1 < /dev/null
  echo "COLRM_SCORED $lab $(grep -c '^ROW' $F) ROWs $(date +%F_%T)"
done
