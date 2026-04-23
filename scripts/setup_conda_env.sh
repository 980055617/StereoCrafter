#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-stereocrafter}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment-conda.yml" --prune
else
  conda env create -n "$ENV_NAME" -f "$PROJECT_DIR/environment-conda.yml"
fi

set +u
conda activate "$ENV_NAME"
set -u

FORWARD_WARP_ROOT="$PROJECT_DIR/dependency/Forward-Warp"
FORWARD_WARP_CUDA_LIB="$(find "$FORWARD_WARP_ROOT/Forward_Warp/cuda/build" -maxdepth 1 -type d -name 'lib.*' 2>/dev/null | head -n 1 || true)"
STEREO_PYTHONPATH="$PROJECT_DIR:$FORWARD_WARP_ROOT"
if [[ -n "$FORWARD_WARP_CUDA_LIB" ]]; then
  STEREO_PYTHONPATH="$STEREO_PYTHONPATH:$FORWARD_WARP_CUDA_LIB"
fi
export PYTHONPATH="$STEREO_PYTHONPATH:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export TRITON_DISABLE_LINE_INFO="${TRITON_DISABLE_LINE_INFO:-1}"
export CC="${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"

python -m pip install -U pip setuptools wheel ninja packaging hatchling hatch-vcs pybind11
python -m pip install --no-cache-dir -r "$PROJECT_DIR/requirements.txt"

# Keep Triton aligned with PyTorch 2.4.0. The old Docker image pinned 2.3.1 for
# Turing, but this conda environment runs on RTX 4090 and torch inductor expects 3.0.
python -m pip uninstall -y triton triton-nightly >/dev/null 2>&1 || true
python -m pip install --no-cache-dir "triton==3.0.0"

TORCH_LIB_DIR="$(python - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).parent / "lib")
PY
)"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

# mamba-ssm has CUDA extensions. Build it after torch/triton are fixed in place.
python -m pip uninstall -y mamba-ssm >/dev/null 2>&1 || true
rm -rf "${HOME}/.cache/torch_extensions" "${HOME}/.cache/pip"
python -m pip install --no-cache-dir --no-binary=:all: --no-build-isolation "mamba-ssm[causal-conv1d]"

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/stereocrafter.sh" <<EOF
export PYTHONPATH="$STEREO_PYTHONPATH:\${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="\${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="\${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="\${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export TRITON_DISABLE_LINE_INFO="\${TRITON_DISABLE_LINE_INFO:-1}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB_DIR:\${LD_LIBRARY_PATH:-}"
export CC="\${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc}"
export CXX="\${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
export CUDAHOSTCXX="\${CUDAHOSTCXX:-\$CXX}"
EOF

python "$PROJECT_DIR/scripts/verify_conda_env.py"
