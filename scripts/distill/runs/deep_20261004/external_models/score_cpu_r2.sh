#!/bin/bash
# CPU scoring of M2SVid renders (deviation D3): per clip UNREG (score_clip_ll_cpu_r2.py, origin re-scored in the same call),
# REGISTERED from the published shifts (score_reg_from_published_r2.py, origin + deliverable re-scored), NR (pyiqa, CPU),
# temporal (temporal_cpu_r2.py).  No GPU, no lock.  usage: bash score_cpu_r2.sh <clip>...
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export CUDA_VISIBLE_DEVICES="" SCORE_STEP=4 NTHREADS=${NTHREADS:-8}
R=scripts/distill/runs/deep_20261004/external_models
O=outputs/deep_20261004/external_models
TAG=m2svid_fa_w16; LABEL=${TAG}_ll
for c in "$@"; do
  M=$O/clips/${c}_${LABEL}/${c}_inpainting_results_sbs.mkv
  [ -s "$M" ] || { echo "MISSING $c" >> $R/score_cpu_r2.log; continue; }
  echo "CLIP_START $c $(date '+%F_%T')" >> $R/score_cpu_r2.log
  python $R/score_clip_ll_cpu_r2.py "$c=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv" "$c=$M" \
    > $R/unreg_cpu/${c}.txt 2>&1 &
  [ -e $O/score_reg_cpu_${TAG}_r2/$c.json ] || python $R/score_reg_from_published_r2.py $O/score_reg_cpu_${TAG}_r2 $c ${LABEL}=$M \
    > $R/reg_cpu/${c}.log 2>&1 &
  [ -e $O/nr_cpu_${TAG}_r2/$c.json ] || PYTHONPATH=/mnt/ssd_data/deep_20261004/external_models/pylib_iqa \
    TORCH_HOME=/mnt/ssd_data/deep_20261004/external_models/torch_home HF_HUB_OFFLINE=1 \
    python $R/nr_metrics_r2.py $O/nr_cpu_${TAG}_r2 $c origin_ll=outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv \
    deliv=outputs/beyond_distil_mamba_scaled/clips/${c}_mstudent2_step800_deliv_ll/${c}_inpainting_results_sbs.mkv \
    m2svid=$M > $R/nr_cpu/${c}.log 2>&1 &
  python $R/temporal_cpu_r2.py $O/temporal_cpu_${TAG}_r2 $TAG $c > $R/temporal_cpu/${c}.log 2>&1 &
  wait
  echo "CLIP_END $c $(date '+%F_%T')" >> $R/score_cpu_r2.log
done
