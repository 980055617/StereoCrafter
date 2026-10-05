#!/bin/bash
# scale_gt SMOKE chain: encode 8 windows (2 clips) -> train 20 steps -> render dev 0184 plain origin + step0 swap (md5 control)
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "SMOKE_START $(date +%F_%T)"
flock /tmp/claude-gpu1.lock python $L/encode_cache_v1.py /mnt/ssd_data/deep_20261004/scale_gt/smoke_v2/crops /mnt/ssd_data/deep_20261004/scale_gt/smoke_v2/latents 0160,0094 > $L/smoke/encode_smoke_v1.log 2>&1
echo "ENCODE rc=$? $(date +%F_%T)"
flock /tmp/claude-gpu1.lock python $L/train_scale_gt_v1.py $L/smoke/spec_smoke_v1.json > $L/smoke/train_smoke_v1.log 2>&1
echo "TRAIN rc=$? $(date +%F_%T)"
bash $L/run_render_v1.sh $L/smoke/jobs_smoke_v1.txt
echo "RENDER done $(date +%F_%T)"
echo "SMOKE_DONE $(date +%F_%T)"
