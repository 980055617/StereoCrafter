#!/bin/bash
# PHASE A for all 291 GT-valid train clips (CPU only, no GPU): 6 shards x 4 threads.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
echo "PREP_START $(date +%F_%T)"
for sh in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES='' PREP_THREADS=4 python $L/prep_crops_v3.py /mnt/ssd_data/deep_20261004/scale_gt/cache_v1 $sh 6 $L/train_clips_v1.json > $L/prep_v3_sh$sh.log 2>&1 &
done
wait
echo "PREP_DONE $(date +%F_%T)"
