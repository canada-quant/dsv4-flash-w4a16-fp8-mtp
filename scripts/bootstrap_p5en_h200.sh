#!/usr/bin/env bash
# Bootstrap an AWS p5en.48xlarge DLAMI box (8× H200, SM 9.0) for the DSv4 MTP re-quant.
# Idempotent — safe to re-run after partial completion or instance restart.
#
# Why H200 (and not B300): the predecessor canada-quant/DeepSeek-V4-Flash-W4A16-FP8
# was successfully calibrated on H200, and our B300 attempt hit multi-rank NCCL
# friction. Same recipe, proven hardware, plus the MTP-preservation deltas we
# developed on the B300 box.
#
# What the DLAMI gives us (verified on ami-0bae40837d7422a24, 2026-05-20):
#   * /opt/pytorch       — Python 3.13.13 venv, torch 2.11.0+cu130, CUDA 13.0
#                          bundled (runtime only — source builds need full
#                          /usr/local/cuda)
#   * /opt/dlami/nvme    — 27.6 TB LVM RAID0 over 8× 3.5 TB NVMe (instance
#                          store; wiped on stop)
#   * /dev/shm           — 1.0 TB tmpfs
#   * /                  — 4.9 TB root EBS (single disk; no separate /data EBS)
#
# What this script adds:
#   * /scratch -> /opt/dlami/nvme symlink (ephemeral by design)
#   * cuda-toolkit-13-0 at /usr/local/cuda (full toolkit for source builds)
#   * HF_HOME, CUDA_HOME, PATH in ~/.bashrc
#   * ~/venv-calib       (transformers 5.8.1 + llm-compressor f2aa32e2 + compressed-tensors 0.15.1a20260515)
#   * ~/venv-serve       (jasl/vllm @ 3424fba5)
#   * MTP-preservation patches applied to venv-calib
#   * packed_modules_mapping patches applied to venv-serve
#
# Layout note: unlike the B300 setup, there is no second EBS to mount as /data.
# Root is 4.9 TB so venvs and build trees live under $HOME directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLMC_SHA="f2aa32e2bde1941182d8f8a348837574969335e6"
VLLM_SHA="3424fba51301504262c3d8355e2560469f18c9c4"
TRANSFORMERS_VER="5.8.1"
# 0.15.1a20260515 ships compressed_tensors.distributed, which llm-compressor
# f2aa32e2 imports unconditionally. See memory:project_dsv4_mtp_requant.
CT_VER="0.15.1a20260515"

# ---------- CUDA toolchain ----------
# /opt/pytorch/cuda is runtime-only on the DLAMI; install the apt toolkit for
# source builds. (Same gotcha as B300 — see memory:dlami_cuda_toolkit_incomplete.)
if [[ ! -d /usr/local/cuda/lib64 ]]; then
    echo "[cuda] installing cuda-toolkit-13-0 (one-time, ~3 GB download)"
    if [[ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]]; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
    fi
    sudo apt-get -qq update
    sudo apt-get install -y cuda-toolkit-13-0
fi
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[cuda] $($CUDA_HOME/bin/nvcc --version | grep 'release' | head -1)"

add_to_bashrc() {
    local line="$1"
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}
add_to_bashrc "export CUDA_HOME=$CUDA_HOME"
add_to_bashrc "export PATH=\$CUDA_HOME/bin:\$PATH"
add_to_bashrc "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"

# ---------- /scratch -> instance store ----------
sudo ln -sfn /opt/dlami/nvme /scratch
sudo chown -h "$USER:$USER" /scratch
mkdir -p /scratch/hf-cache
add_to_bashrc "export HF_HOME=/scratch/hf-cache"
export HF_HOME=/scratch/hf-cache
echo "[ok] /scratch -> $(readlink /scratch)"

# ---------- venv-calib ----------
VENV_CALIB="$HOME/venv-calib"
if [[ ! -f "$VENV_CALIB/bin/python" ]]; then
    /opt/pytorch/bin/python3 -m venv "$VENV_CALIB"
fi
# shellcheck source=/dev/null
source "$VENV_CALIB/bin/activate"
pip install --quiet --upgrade pip wheel setuptools
# torch from PyTorch's cu130 index — DLAMI's pre-installed 2.11.0 lives in
# /opt/pytorch and we don't inherit it (no --system-site-packages, see
# memory:dlami_python_gotcha for why).
pip install --quiet --index-url https://download.pytorch.org/whl/cu130 torch
pip install --quiet --no-deps "transformers==$TRANSFORMERS_VER"
pip install --quiet accelerate datasets safetensors "compressed-tensors==$CT_VER" \
    huggingface_hub hf-transfer tqdm regex tokenizers
pip install --quiet --no-deps "git+https://github.com/vllm-project/llm-compressor.git@$LLMC_SHA"
deactivate
echo "[ok] venv-calib ready at $VENV_CALIB"

# ---------- venv-serve ----------
VENV_SERVE="$HOME/venv-serve"
if [[ ! -f "$VENV_SERVE/bin/python" ]]; then
    /opt/pytorch/bin/python3 -m venv "$VENV_SERVE"
fi
# shellcheck source=/dev/null
source "$VENV_SERVE/bin/activate"
pip install --quiet --upgrade pip wheel
# Pin setuptools <78 — jasl/vllm's pyproject.toml uses both project.license.file
# AND project.license.text, which setuptools 78+ rejects (PEP 639 strict).
pip install --quiet "setuptools<78"
# jasl/vllm @ 3424fba5 pins torch==2.11.0 in pyproject. If we pre-install
# torch 2.12 (the default from the cu130 index), pip will downgrade *after*
# building the C++ extensions — the .so links against 2.12 symbols then
# loads against 2.11 at import time → ImportError on libtorch ABI. Pin 2.11.0
# explicitly here to keep the symbols stable through the whole build.
pip install --quiet --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0"
pip install --quiet ninja cmake "numpy<3" pybind11 packaging setuptools-scm

# Hopper SM 9.0a — drop B300's 10.0a; predecessor calibration arch.
export TORCH_CUDA_ARCH_LIST="9.0a"
export MAX_JOBS=${MAX_JOBS:-32}
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"

mkdir -p "$HOME/src"
[[ -d "$HOME/src/vllm/.git" ]] || git clone https://github.com/jasl/vllm.git "$HOME/src/vllm"
cd "$HOME/src/vllm"
git fetch --depth=1 origin "$VLLM_SHA"
git checkout "$VLLM_SHA"

# --no-deps prevents pip from touching torch during the install; we already
# have the exact version vLLM's pyproject demands.
pip install --no-build-isolation -v --no-deps -e "$HOME/src/vllm" 2>&1 | tee /tmp/vllm_build.log
# Resolve the rest of the runtime deps separately, now that the extension is built.
pip install --quiet -r "$HOME/src/vllm/requirements/common.txt" 2>/dev/null || true
deactivate
echo "[ok] venv-serve ready at $VENV_SERVE"

# ---------- apply patches to venv-calib ----------
# Diff prefixes (verified 2026-05-20):
#   modeling_deepseek_v4.py.diff has `a/transformers/...` → -p1 with cwd at site-packages
#   helpers.py.diff             has `a/src/llmcompressor/...` → -p2 with cwd at site-packages
source "$VENV_CALIB/bin/activate"
TR_DIR=$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')
SITE_PACKAGES=$(dirname "$TR_DIR")
if ! grep -q "paul/dsv4" "$TR_DIR/models/deepseek_v4/modeling_deepseek_v4.py"; then
    patch -p1 -d "$SITE_PACKAGES" < "$REPO_ROOT/patches/modeling_deepseek_v4.py.diff"
fi
LLMC_DIR=$(python -c 'import llmcompressor, os; print(os.path.dirname(os.path.dirname(llmcompressor.__file__)))')
if ! grep -q "paul/dsv4" "$LLMC_DIR/llmcompressor/pipelines/sequential/helpers.py"; then
    patch -p2 -d "$LLMC_DIR" < "$REPO_ROOT/patches/helpers.py.diff"
fi
echo "[ok] venv-calib patches applied"
deactivate

# ---------- apply patches to venv-serve ----------
source "$VENV_SERVE/bin/activate"
VLLM_DIR=$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
python "$REPO_ROOT/scripts/patch_v4_forcausal_packed_mapping.py" "$VLLM_DIR"
python "$REPO_ROOT/scripts/patch_mtp_packed_mapping.py" "$VLLM_DIR"
echo "[ok] venv-serve patches applied"
deactivate

echo
echo "BOOTSTRAP_DONE"
echo "  HF_HOME=$HF_HOME"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  venv-calib=$VENV_CALIB (Python $($VENV_CALIB/bin/python --version 2>&1 | awk '{print $2}'))"
echo "  venv-serve=$VENV_SERVE (Python $($VENV_SERVE/bin/python --version 2>&1 | awk '{print $2}'))"
