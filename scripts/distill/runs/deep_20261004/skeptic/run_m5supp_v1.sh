#!/bin/bash
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 SK_THREADS=3
export TORCH_HOME=/mnt/ssd_data/deep_20261004/skeptic/torch_home PYTHONPATH=/mnt/ssd_data/deep_20261004/skeptic/pylib
python scripts/distill/runs/deep_20261004/skeptic/job_m5supp_v1.py outputs/deep_20261004/skeptic/m5supp_v1.json 2>&1 | grep -v -i -E "[w]arning|load_state_dict"
echo DRIVER_DONE
