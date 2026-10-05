#!/bin/bash
# DIAGNOSTIC: encode the dev windows (dev_cache_v1) and evaluate the held-out masked v-MSE of origin (step0) and the MAIN /
# CONTRAST checkpoints at the matched steps.  Each GPU job under the GPU-1 lock.
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=scripts/distill/runs/deep_20261004/scale_gt
D=/mnt/ssd_data/deep_20261004/scale_gt
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "HELDOUT_CHAIN_START $(date +%F_%T)"
flock /tmp/claude-gpu1.lock python $L/encode_cache_v1.py $D/dev_cache_v1/crops $D/dev_cache_v1/latents 0040,0082,0091,0184,0245,0268 > $L/encode_dev_v1.log 2>&1 < /dev/null
echo "ENCODE_DEV rc=$? $(date +%F_%T)"
CKS="$D/ck/main_v1/step0.pt"
for s in 250 500 1000 2000 3000; do CKS="$CKS $D/ck/main_v1/step$s.pt"; done
for s in 250 500 1000; do CKS="$CKS $D/ck/contrast8_v1/step$s.pt"; done
flock /tmp/claude-gpu1.lock python $L/eval_heldout_loss_v1.py $L/heldout_loss_v1.json $CKS > $L/heldout_loss_v1.log 2>&1 < /dev/null
echo "HELDOUT rc=$? $(date +%F_%T)"
