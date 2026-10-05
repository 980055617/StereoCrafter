#!/bin/bash
# vae_20261005 / decoder_swap -- run one python command of this lane on GPU $1 under that GPU's lock, conda env active.
# usage: bash run_gpu_v1.sh <gpu> <log> <script.py> [args...]
set -u
GPU=$1; LOG=$2; shift 2
cd /home/kawa/master_project/StereoCrafter
set +u; source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate stereocrafter; set -u
export PYTHONPATH=/mnt/ssd_data/vae_20261005/decoder_swap/pylib
export TORCH_HOME=/mnt/ssd_data/vae_20261005/decoder_swap/torch_home
export HF_HOME=/mnt/ssd_data/vae_20261005/decoder_swap/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "START $(date +%F_%T) gpu=$GPU $*" >> $LOG
CUDA_VISIBLE_DEVICES=$GPU flock /tmp/claude-gpu$GPU.lock python "$@" >> $LOG 2>&1 < /dev/null
RC=$?
echo "END rc=$RC $(date +%F_%T)" >> $LOG
exit $RC
