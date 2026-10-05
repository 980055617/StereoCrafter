#!/bin/bash
# blur_diag PREREG ADDENDUM 5 chain: RS_GTx_L for 0204 (CPU), then -- after the VAE batch -- ONE acquisition of
# /tmp/claude-gpu0.lock for VAE_GTx_L (0170 0204 0042 0052) + their add-on scoring (GPU LPIPS/NR, CPU detail).
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/blur_diag
O=outputs/deep_20261004/blur_diag
L=/mnt/ssd_data/deep_20261004/blur_diag
echo "ADDON_START $(date +%F_%T)"
CUDA_VISIBLE_DEVICES='' python $R/make_lanczos_rows_r2.py --rs 0204 > $R/make_lanczos_rs_0204_r2.log 2>&1
echo "RS_GTx_L 0204 rc=$? $(date +%F_%T)"
until grep -q VAE_BATCH_DONE $R/chain_vae_batch_r2.log 2>/dev/null; do sleep 30; done
echo "VAE batch done -> queue for lock $(date +%F_%T)"
flock /tmp/claude-gpu0.lock bash -c "
  export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo ADDON_LOCKED \$(date +%F_%T)
  python $R/vae_gtx_lanczos_r2.py 0170 0204 0042 0052 > $O/lanczos_r2/vae_gtx_lanczos.log 2>&1; echo VAE_GTX_L rc=\$? \$(date +%F_%T)
  for c in 0170 0204 0042 0052; do
    ROWS=VAE_GTx_L; [ \$c = 0204 ] && ROWS=VAE_GTx_L,RS_GTx_L
    env PYTHONPATH=$L/pylib TORCH_HOME=$L/torch_home HF_HOME=$L/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python $R/score_addon_r2.py $O/scores_r2/addon \$c \$ROWS > $O/scores_r2/logs/addon_\$c.log 2>&1
    echo ADDON_SCORE \$c rc=\$? \$(date +%F_%T)
  done
  echo ADDON_UNLOCK \$(date +%F_%T)"
echo "ADDON_DONE $(date +%F_%T)"
