#!/bin/bash
# CPU-only driver for the skeptic lane.  usage: run_cpu_v1.sh <threads> <script.py> <out_dir> <clip> [<clip> ...]
set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate stereocrafter; set -u
cd /home/kawa/master_project/StereoCrafter
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" SK_THREADS="$1"
export TORCH_HOME=/mnt/ssd_data/deep_20261004/skeptic/torch_home PYTHONPATH=/mnt/ssd_data/deep_20261004/skeptic/pylib
TH="$1"; SCRIPT="$2"; OUT="$3"; shift 3
for c in "$@"; do
  if [ -f "$OUT/$c.json" ]; then echo "[skip] $c exists"; continue; fi
  python "$SCRIPT" "$OUT" "$c" 2>&1 | grep -v -i -E "[w]arning|load_state_dict" || echo "[FAIL] $c"
done
echo "DRIVER_DONE $SCRIPT $*"
