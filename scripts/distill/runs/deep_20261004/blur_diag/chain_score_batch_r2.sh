#!/bin/bash
# blur_diag scoring batch (PREREG ADDENDUM 3).  Detail statistics on CPU (no lock, 3 in parallel) as soon as a clip's
# rows exist; LPIPS + NR of the 11 non-smoke clips inside ONE acquisition of /tmp/claude-gpu0.lock once every VAE row
# and both HIRES_B (+ LANCZOS) rows exist.  Scorers refuse to overwrite.
set -u
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
R=scripts/distill/runs/deep_20261004/blur_diag
O=outputs/deep_20261004/blur_diag
S=$O/scores_r2
L=/mnt/ssd_data/deep_20261004/blur_diag
mkdir -p $S/lpips $S/detail $S/nr $S/logs
CL="0052 0125 0128 0141 0147 0225 0251 0259 0301 0170 0042"
vae_done() { [ -e $O/vae_rt_r2/$1/meta_vae_rt.json ] && grep -q total_seconds $O/vae_rt_r2/$1/meta_vae_rt.json; }
renders_done() { grep -q RENDER_BATCH_DONE $R/chain_render_batch_r2.log 2>/dev/null; }
all_vae_done() { for c in $CL; do vae_done $c || return 1; done; return 0; }
echo "SCORE_BATCH_START $(date +%F_%T)"
# ---- detail (CPU) as rows become available
pending="$CL"
while [ -n "$pending" ]; do
  left=""
  for c in $pending; do
    ready=0
    if vae_done $c; then
      case $c in 0170|0042) renders_done && ready=1;; *) ready=1;; esac
    fi
    if [ $ready = 1 ] && [ ! -e $S/detail/$c.json ]; then
      while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 5; done
      ( CUDA_VISIBLE_DEVICES='' python $R/score_detail_r2.py $S/detail $c > $S/logs/detail_$c.log 2>&1
        echo "DETAIL $c rc=$? $(date +%F_%T)" ) &
    elif [ $ready = 0 ]; then left="$left $c"; fi
  done
  pending=$(echo $left)
  [ -n "$pending" ] && sleep 30
done
# ---- LPIPS + NR (GPU, one acquisition) once everything exists
until renders_done && all_vae_done; do sleep 30; done
echo "ALL_ROWS_READY $(date +%F_%T)"
flock /tmp/claude-gpu0.lock bash -c "
  export CUDA_VISIBLE_DEVICES=0
  echo SCORE_BATCH_LOCKED \$(date +%F_%T)
  for c in $CL; do
    python $R/score_lpips_r2.py $S/lpips \$c > $S/logs/lpips_\$c.log 2>&1; a=\$?
    env PYTHONPATH=$L/pylib TORCH_HOME=$L/torch_home HF_HOME=$L/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python $R/score_nr_r2.py $S/nr \$c > $S/logs/nr_\$c.log 2>&1; b=\$?
    echo \"SCORE \$c lpips_rc=\$a nr_rc=\$b \$(grep -o 'G0 [A-Z]*' $S/logs/lpips_\$c.log | tail -1) \$(date +%F_%T)\"
  done
  echo SCORE_BATCH_UNLOCK \$(date +%F_%T)"
wait
echo "SCORE_BATCH_DONE $(date +%F_%T)"
