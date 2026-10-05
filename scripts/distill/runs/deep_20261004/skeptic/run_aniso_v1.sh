#!/bin/bash
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/distill/runs/deep_20261004/skeptic/job_aniso_v1.py outputs/deep_20261004/skeptic/aniso_v1.json 0042 0052 0125 0128 0141 0147 0170 0204 0225 0251 0259 0301 2>&1 | grep -v -i -E "[w]arning"
echo DRIVER_DONE
