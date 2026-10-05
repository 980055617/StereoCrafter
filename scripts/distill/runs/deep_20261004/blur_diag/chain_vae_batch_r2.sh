#!/bin/bash
# blur_diag VAE batch (PREREG ADDENDUM 3): waits until the per-clip chain_vae_r2.sh has finished its running clip, stops
# that chain only while it has no python in flight (a waiting flock holds nothing), then runs ALL remaining clips inside
# ONE acquisition of /tmp/claude-gpu0.lock.  VAE_GT32 only on 0301 0170 0042 (+0204, 0052 done earlier).
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/blur_diag
O=outputs/deep_20261004/blur_diag
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "VAE_BATCH_START $(date +%F_%T)"
CG=$(systemctl --user show -p ControlGroup --value blurdiag-vae-chain 2>/dev/null)
while systemctl --user is-active --quiet blurdiag-vae-chain; do
  busy=0
  if [ -n "$CG" ] && [ -r /sys/fs/cgroup$CG/cgroup.procs ]; then
    for p in $(cat /sys/fs/cgroup$CG/cgroup.procs); do
      grep -q "vae_roundtrip_r2.py" /proc/$p/cmdline 2>/dev/null && ! grep -q "^flock" /proc/$p/cmdline 2>/dev/null && busy=1
    done
  else
    busy=1
  fi
  if [ $busy = 0 ]; then
    systemctl --user stop blurdiag-vae-chain
    echo "STOPPED_BY_OPERATOR $(date +%F_%T): no python in flight (only a waiting flock); remaining clips -> chain_vae_batch_r2.sh" >> $R/chain_vae_r2.log
    echo "stopped per-clip chain $(date +%F_%T)"
    break
  fi
  sleep 5
done
flock /tmp/claude-gpu0.lock bash -c '
  R=scripts/distill/runs/deep_20261004/blur_diag; O=outputs/deep_20261004/blur_diag
  echo "VAE_BATCH_LOCKED $(date +%F_%T)"
  for c in 0125 0128 0141 0147 0225 0251 0259 0301 0170 0042; do
    if [ -e $O/vae_rt_r2/$c/meta_vae_rt.json ] && grep -q total_seconds $O/vae_rt_r2/$c/meta_vae_rt.json; then
      echo "VAE $c already complete -> skip"; continue; fi
    case $c in 0301|0170|0042) ROWS=VAE_GT,VAE_BR,VAE_GT32,VAE_GTx,RS_GTx;; *) ROWS=VAE_GT,VAE_BR,VAE_GTx,RS_GTx;; esac
    python $R/vae_roundtrip_r2.py $c $ROWS > $O/vae_rt_r2/${c}_batch.log 2>&1
    echo "VAE $c rc=$? rows=$ROWS $(grep -c "] wrote " $O/vae_rt_r2/${c}_batch.log) written $(date +%F_%T)"
  done
  echo "VAE_BATCH_UNLOCK $(date +%F_%T)"'
echo "VAE_BATCH_DONE $(date +%F_%T)"
