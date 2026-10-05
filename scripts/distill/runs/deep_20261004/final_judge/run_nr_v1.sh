#!/bin/bash
# final_judge NR pass: per clip, wait for its judge score JSON, then judge_nr_v1.py under the GPU lock.
# usage: run_nr_v1.sh <GPU> <clip> [<clip> ...]
set -u
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
L=/mnt/ssd_data/deep_20261004/blur_diag
J=scripts/distill/runs/deep_20261004/final_judge
SD=outputs/deep_20261004/final_judge/score_v1; ND=outputs/deep_20261004/final_judge/nr_v1
GPU=$1; shift
LOG=$J/nr_v1_gpu$GPU.log
echo "START $(date '+%F_%T') gpu=$GPU clips=$*" >> $LOG
for c in "$@"; do
  until [ -e $SD/$c.json ]; do sleep 10; done
  if [ -e $ND/$c.json ]; then echo "SKIP $c" >> $LOG; continue; fi
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$L/pylib TORCH_HOME=$L/torch_home HF_HOME=$L/hf_home HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 flock /tmp/claude-gpu$GPU.lock python $J/judge_nr_v1.py $SD $ND $c >> $LOG 2>&1
  echo "CLIP_RC $c rc=$? $(date '+%F_%T')" >> $LOG
done
echo "END $(date '+%F_%T')" >> $LOG
