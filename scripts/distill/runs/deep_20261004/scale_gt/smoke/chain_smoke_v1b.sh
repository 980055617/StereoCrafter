#!/bin/bash
# scale_gt SMOKE re-run (v1b: fixed window ordering): train 20 steps on 2 clips -> step0-swap render of dev 0184 (control C1)
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "SMOKE_B_START $(date +%F_%T)"
flock /tmp/claude-gpu1.lock python $L/train_scale_gt_v1.py $L/smoke/spec_smoke_v1b.json > $L/smoke/train_smoke_v1b.log 2>&1 < /dev/null
echo "TRAIN rc=$? $(date +%F_%T)"
bash $L/run_render_v1.sh $L/smoke/jobs_smoke_v1b.txt
echo "SMOKE_B_DONE $(date +%F_%T)"
