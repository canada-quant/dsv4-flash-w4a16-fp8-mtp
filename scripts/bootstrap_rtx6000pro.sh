#!/usr/bin/env bash
# Bootstrap a Brev RTX PRO 6000 Server Edition box (Blackwell, SM 12.0)
# for serving the W4A16+FP8+MTP artifact.
#
# Hardware target: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB
# HBM3 per GPU), 1 TiB RAM, 256 GiB root disk + 7.6 TiB ephemeral LVM
# at /opt/dlami/nvme. Tested with familiar-teal-worm (NCA-d2e3-84318),
# 4 GPUs × 96 vCPU, Columbus OH AWS region. Driver 580.159, pre-installed
# CUDA 12.9 toolkit at /usr/local/cuda — we install 13.0 alongside.
#
# Key differences vs bootstrap_p5en_h200.sh:
#   * /opt/dlami/nvme is local NVMe (same layout as p5en) — symlink as
#     /scratch and put everything there. Artifact (159 GiB) + venvs +
#     vLLM build all fit with TB to spare.
#   * SM 12.0 (Blackwell consumer/server family — distinct from B300's
#     SM 10.0). Uses jasl/vllm@ds4-sm120-preview-dev — jasl's current
#     SM12-tuned branch, rebased on post-refactor upstream main. The
#     older `ds4-sm120-experimental` branch (May 6) requires six extra
#     patches and forces --enforce-eager; preview-dev needs only the
#     standard packed_modules_mapping plus the dynamo-safe wo_a check.
#   * No DLAMI bundled torch; need full CUDA toolkit for the source build.
#
# Idempotent — safe to re-run after partial completion.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# jasl/vllm SM12-tuned branch (predecessor's serve build).
# Pin to a SHA once we verify on first run; for now use branch head.
VLLM_REMOTE="https://github.com/jasl/vllm.git"
# Use preview-dev (jasl's current SM12 branch, rebased on post-refactor
# upstream main). Has the four H200 cherry-pick fixes baked in. The older
# `ds4-sm120-experimental` branch (May 6) predates the refactor and requires
# 6+ extra patches; preview-dev needs only the standard packed_modules_mapping.
VLLM_BRANCH="ds4-sm120-preview-dev"

# Cherry-pick PRs from the H200 build that fix scheme resolution +
# MTP-from-safetensors auto-detect (format-agnostic, apply to SM12 too).
VLLM_CHERRYPICKS=(
    "43248"   # is_static_input_scheme bool wrap
    "43288"   # scale_fmt default ue8m0
    "43290"   # weight_scale_inv-or-weight_scale fallback (SM90/SM120 shared)
    "43319"   # auto-detect BF16 MTP from safetensors
)

# ---------- workspace ----------
# /scratch -> /opt/dlami/nvme symlink should already exist (one-time
# setup outside this script). 7.6 TiB ephemeral, wiped on stop.
sudo ln -sfn /opt/dlami/nvme /scratch 2>/dev/null || true
sudo chown -h "$USER:$USER" /scratch 2>/dev/null || true
WORKSPACE="/scratch"
mkdir -p "$WORKSPACE/hf-cache"

add_to_bashrc() {
    local line="$1"
    grep -qxF "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
}

# ---------- CUDA toolchain ----------
if [[ ! -d /usr/local/cuda/lib64 ]]; then
    echo "[cuda] installing cuda-toolkit-13-0 (one-time, ~3 GB download)"
    if [[ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]]; then
        UBUNTU_REL=$(lsb_release -rs | tr -d .)
        if [[ "$UBUNTU_REL" == "2204" ]]; then
            UBUNTU_REPO="ubuntu2204"
        else
            UBUNTU_REPO="ubuntu2404"
        fi
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_REPO}/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
    fi
    sudo apt-get -qq update
    sudo apt-get install -y cuda-toolkit-13-0
fi
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[cuda] $($CUDA_HOME/bin/nvcc --version | grep 'release' | head -1)"

add_to_bashrc "export CUDA_HOME=$CUDA_HOME"
add_to_bashrc "export PATH=\$CUDA_HOME/bin:\$PATH"
add_to_bashrc "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
add_to_bashrc "export HF_HOME=$WORKSPACE/hf-cache"

# ---------- system deps for source build ----------
if ! dpkg -s build-essential >/dev/null 2>&1; then
    sudo apt-get install -y build-essential git python3-venv python3-dev libnuma-dev
fi

# ---------- venv-serve ----------
VENV_SERVE="$HOME/venv-serve"
if [[ ! -f "$VENV_SERVE/bin/python" ]]; then
    python3 -m venv "$VENV_SERVE"
fi
# shellcheck source=/dev/null
source "$VENV_SERVE/bin/activate"
pip install --quiet --upgrade pip wheel
# Setuptools <78 to avoid PEP 639 strict-license rejection on jasl/vllm pyproject.
pip install --quiet "setuptools<78"
# torch 2.11.0+cu130 — pyproject of the SM120 branch pins 2.11.
pip install --quiet --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0"
pip install --quiet ninja cmake "numpy<3" pybind11 packaging setuptools-scm

# Blackwell RTX PRO 6000 is SM 12.0. Use 12.0a for architecture-specific
# kernels jasl branch enables. (a-suffix = arch-specific PTX/SASS.)
export TORCH_CUDA_ARCH_LIST="12.0a"
export MAX_JOBS=${MAX_JOBS:-32}
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
add_to_bashrc "export TORCH_CUDA_ARCH_LIST=\"12.0a\""

mkdir -p "$HOME/src"
if [[ ! -d "$HOME/src/vllm/.git" ]]; then
    git clone --branch "$VLLM_BRANCH" "$VLLM_REMOTE" "$HOME/src/vllm"
fi
cd "$HOME/src/vllm"
git fetch origin "$VLLM_BRANCH"
git checkout "$VLLM_BRANCH"
git pull --ff-only origin "$VLLM_BRANCH" || true

# Cherry-pick the H200 fix PRs from upstream vllm-project/vllm.
# Idempotent: skip if already merged.
if ! git remote | grep -q "^upstream$"; then
    git remote add upstream https://github.com/vllm-project/vllm.git
fi
git fetch --quiet upstream main
for pr in "${VLLM_CHERRYPICKS[@]}"; do
    if ! git log --oneline | grep -q "PR #${pr}\b"; then
        # Best-effort cherry-pick — find merge commit on upstream main
        sha=$(git log upstream/main --oneline | grep -i "#${pr}" | head -1 | awk '{print $1}' || true)
        if [[ -n "$sha" ]]; then
            git cherry-pick -X theirs "$sha" || git cherry-pick --abort
        fi
    fi
done

# Source build of vLLM. ~25-30 min on 96 vCPU.
# Non-editable (-e) because the jasl/vllm SM12 branches' build backend
# doesn't implement PEP 660 build_editable hook. Our packed_modules
# patches write into the installed vllm package directly, so editable
# isn't required.
pip install --no-build-isolation -v --no-deps "$HOME/src/vllm" 2>&1 | tee /tmp/vllm_build.log
pip install --quiet -r "$HOME/src/vllm/requirements/common.txt" 2>/dev/null || true
deactivate
echo "[ok] venv-serve ready at $VENV_SERVE"

# ---------- apply packed_modules_mapping patches ----------
source "$VENV_SERVE/bin/activate"
VLLM_DIR=$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
python "$REPO_ROOT/scripts/patch_v4_forcausal_packed_mapping.py" "$VLLM_DIR" || \
    echo "[warn] patch_v4_forcausal failed — check path post-refactor"
python "$REPO_ROOT/scripts/patch_mtp_packed_mapping.py" "$VLLM_DIR" || \
    echo "[warn] patch_mtp failed — check path post-refactor"
deactivate
echo "[ok] venv-serve patches applied"

# ---------- huggingface_hub for download ----------
"$VENV_SERVE/bin/pip" install --quiet huggingface_hub hf-transfer

echo
echo "BOOTSTRAP_DONE"
echo "  WORKSPACE=$WORKSPACE"
echo "  HF_HOME=$WORKSPACE/hf-cache"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  TORCH_CUDA_ARCH_LIST=12.0a"
echo "  venv-serve=$VENV_SERVE (Python $($VENV_SERVE/bin/python --version 2>&1 | awk '{print $2}'))"
