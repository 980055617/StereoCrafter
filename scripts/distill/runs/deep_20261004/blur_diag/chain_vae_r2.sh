#!/bin/bash
# blur_diag VAE round trips for the 11 non-smoke clips, GPU 0, one flock acquisition per clip (~6.5 min each).
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/blur_diag
O=outputs/deep_20261004/blur_diag
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "CHAIN_VAE_START $(date +%F_%T)"
for c in 0052 0125 0128 0141 0147 0225 0251 0259 0301 0170 0042; do
  if [ -e $O/vae_rt_r2/$c/meta_vae_rt.json ] && grep -q total_seconds $O/vae_rt_r2/$c/meta_vae_rt.json; then
    echo "VAE $c already complete -> skip"; continue; fi
  flock /tmp/claude-gpu0.lock python $R/vae_roundtrip_r2.py $c > $O/vae_rt_r2/$c.log 2>&1
  echo "VAE $c rc=$? $(grep -c '] wrote ' $O/vae_rt_r2/$c.log) rows $(date +%F_%T)"
done
echo "CHAIN_VAE_DONE $(date +%F_%T)"
