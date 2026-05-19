#!/usr/bin/env bash
# Bootstrap an AWS p6-b300.48xlarge DLAMI box for the DSv4 MTP re-quant.
# Idempotent — safe to re-run after partial completion or instance restart.
#
# What the DLAMI gives us out of the box (verified on ami-02e9fc7da15a197f9,
# 2026-05-19):
#   * /opt/pytorch — Python 3.13.13 venv with torch 2.11.0+cu130 (CUDA arch
#     list includes sm_100/B300), bundled nvidia/cu13 toolkit at
#     /opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/{bin,include,lib,nvvm}
#     and a parallel /opt/pytorch/cuda symlink to the same toolkit.
#   * /opt/dlami/nvme — 27.6 TB RAID0/LVM instance store (wiped on stop).
#   * /dev/shm — 2.0 TB tmpfs.
#   * /dev/nvme1n1 — 300 GB raw EBS, unmounted.
#
# What this script adds:
#   * mkfs.ext4 + persistent mount /dev/nvme1n1 -> /data (300 GB survives stop)
#   * /scratch -> /opt/dlami/nvme symlink (ephemeral by design)
#   * HF_HOME, CUDA_HOME, PATH in ~/.bashrc
#   * /data/venv-calib (calibration: transformers 5.8.1 + llm-compressor f2aa32e2 + compressed-tensors 0.15.0.1)
#   * /data/venv-serve (serving: jasl/vllm @ 3424fba5)
#   * applied patches to the two venvs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLMC_SHA="f2aa32e2bde1941182d8f8a348837574969335e6"
VLLM_SHA="3424fba51301504262c3d8355e2560469f18c9c4"
TRANSFORMERS_VER="5.8.1"
# 0.15.1a20260515 is the alpha that ships compressed_tensors.distributed, which
# llm-compressor f2aa32e2 imports unconditionally. Predecessor's 0.15.0.1 pin
# predates this alpha (no .distributed submodule -> ModuleNotFoundError on import).
CT_VER="0.15.1a20260515"

# ---------- CUDA toolchain ----------
# /opt/pytorch/cuda is runtime-only on the DLAMI (no lib64, no unversioned .so);
# install the full apt-packaged toolkit at /usr/local/cuda for source builds.
# Memory note: dlami_cuda_toolkit_incomplete.md
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

# Idempotent bashrc additions
add_to_bashrc() {
    local line="$1"
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}
add_to_bashrc "export CUDA_HOME=$CUDA_HOME"
add_to_bashrc "export PATH=\$CUDA_HOME/bin:\$PATH"
add_to_bashrc "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"

# ---------- /data (300 GB EBS) ----------
if ! mountpoint -q /data; then
    if [[ "$(sudo file -s /dev/nvme1n1)" == "/dev/nvme1n1: data" ]]; then
        sudo mkfs.ext4 -F -L data /dev/nvme1n1
    fi
    sudo mkdir -p /data
    sudo mount /dev/nvme1n1 /data
    sudo chown "$USER:$USER" /data
    UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)
    if ! grep -q "$UUID" /etc/fstab; then
        echo "UUID=$UUID /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
    fi
fi
echo "[ok] /data mounted: $(df -h /data | tail -1)"

# ---------- /scratch -> instance store ----------
sudo ln -sfn /opt/dlami/nvme /scratch
sudo chown -h "$USER:$USER" /scratch
mkdir -p /scratch/hf-cache
add_to_bashrc "export HF_HOME=/scratch/hf-cache"
export HF_HOME=/scratch/hf-cache
echo "[ok] /scratch -> $(readlink /scratch)"

# ---------- venv-calib ----------
if [[ ! -f /data/venv-calib/bin/python ]]; then
    /opt/pytorch/bin/python3 -m venv /data/venv-calib
fi
# shellcheck source=/dev/null
source /data/venv-calib/bin/activate
pip install --quiet --upgrade pip wheel setuptools
# torch from PyTorch's cu130 index — DLAMI's pre-installed 2.11.0 lives in
# /opt/pytorch and we don't inherit it (no --system-site-packages, see
# memory:dlami_python_gotcha for why). Pin handled by transitive resolution.
pip install --quiet --index-url https://download.pytorch.org/whl/cu130 torch
pip install --quiet --no-deps "transformers==$TRANSFORMERS_VER"
pip install --quiet accelerate datasets safetensors "compressed-tensors==$CT_VER" \
    huggingface_hub hf-transfer tqdm regex tokenizers
pip install --quiet --no-deps "git+https://github.com/vllm-project/llm-compressor.git@$LLMC_SHA"
deactivate
echo "[ok] venv-calib ready"

# ---------- venv-serve ----------
if [[ ! -f /data/venv-serve/bin/python ]]; then
    /opt/pytorch/bin/python3 -m venv /data/venv-serve
fi
# shellcheck source=/dev/null
source /data/venv-serve/bin/activate
pip install --quiet --upgrade pip wheel
# Pin setuptools to <78: jasl/vllm's pyproject.toml uses both
# project.license.file AND project.license.text, which setuptools 78+ rejects
# (PEP 639 strict). Older setuptools accepts the mixed form.
pip install --quiet "setuptools<78"
pip install --quiet --index-url https://download.pytorch.org/whl/cu130 torch
# Build prerequisites (visible to cmake via --no-build-isolation)
pip install --quiet ninja cmake "numpy<3" pybind11 packaging setuptools-scm
#
# NOTE — CUDA toolkit completeness:
# vLLM source build via the DLAMI-bundled CUDA at /opt/pytorch/cuda fails
# because the bundle ships .so.13 files but no unversioned .so symlinks
# (e.g., libnvrtc.so, libcudart.so), and the lib/ dir is not at lib64/.
# Investigated 2026-05-19:
#   1. ld: cannot find -lcudadevrt  ->  fixed with `sudo ln -sfn lib /opt/pytorch/cuda/lib64`
#   2. CUDA_nvrtc_LIBRARY NOTFOUND  ->  needs `libnvrtc.so` symlink to `libnvrtc.so.13`
#   3. probably more missing symlinks beyond that
# Cleanest fix: install the standard NVIDIA CUDA toolkit alongside, which
# provides /usr/local/cuda with full and properly-symlinked libs.
#   sudo apt-get install -y cuda-toolkit-13-0
# Then point CUDA_HOME=/usr/local/cuda for the vLLM build (keep
# /opt/pytorch's torch — it works against the system CUDA at runtime).
#
# Until that step is taken, the block below WILL FAIL. Leaving it staged so
# a future session can flip the CUDA_HOME and rerun.
export TORCH_CUDA_ARCH_LIST="10.0a"
export MAX_JOBS=${MAX_JOBS:-32}
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"

# Clone jasl/vllm into /data/src/vllm if not present, so it survives instance
# stop and can be edited locally if pyproject quirks need patching.
mkdir -p /data/src
[[ -d /data/src/vllm/.git ]] || git clone https://github.com/jasl/vllm.git /data/src/vllm
cd /data/src/vllm
git fetch --depth=1 origin "$VLLM_SHA"
git checkout "$VLLM_SHA"

pip install --no-build-isolation -v -e /data/src/vllm 2>&1 | tee /tmp/vllm_build.log
deactivate
echo "[ok] venv-serve ready"

# ---------- apply patches to venv-calib ----------
source /data/venv-calib/bin/activate
TR_DIR=$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')
if ! grep -q "paul/dsv4 calibration" "$TR_DIR/models/deepseek_v4/modeling_deepseek_v4.py"; then
    patch -p1 -d "$(dirname "$(dirname "$TR_DIR")")" < "$REPO_ROOT/patches/modeling_deepseek_v4.py.diff"
fi
LLMC_DIR=$(python -c 'import llmcompressor, os; print(os.path.dirname(os.path.dirname(llmcompressor.__file__)))')
if ! grep -q "paul/dsv4" "$LLMC_DIR/llmcompressor/pipelines/sequential/helpers.py"; then
    patch -p1 -d "$LLMC_DIR/.." < "$REPO_ROOT/patches/helpers.py.diff"
fi
echo "[ok] venv-calib patches applied"
deactivate

# ---------- apply patches to venv-serve ----------
source /data/venv-serve/bin/activate
VLLM_DIR=$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
python "$REPO_ROOT/scripts/patch_v4_forcausal_packed_mapping.py" "$VLLM_DIR"
python "$REPO_ROOT/scripts/patch_mtp_packed_mapping.py" "$VLLM_DIR"
echo "[ok] venv-serve patches applied"
deactivate

echo
echo "BOOTSTRAP_DONE"
echo "  HF_HOME=$HF_HOME"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  venv-calib=/data/venv-calib (Python $(/data/venv-calib/bin/python --version 2>&1 | awk '{print $2}'))"
echo "  venv-serve=/data/venv-serve (Python $(/data/venv-serve/bin/python --version 2>&1 | awk '{print $2}'))"
