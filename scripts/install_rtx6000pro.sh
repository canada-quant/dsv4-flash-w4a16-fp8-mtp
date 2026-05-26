#!/usr/bin/env bash
# install_rtx6000pro.sh — RTX PRO 6000 install for Card D (W4A16-MTP)
#
# canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP
#
# **CURRENT SHIPPING STATE (2026-05-26)**: this script builds vLLM and
# downloads the artifact correctly, but Card D does NOT serve correctly on
# current upstream vLLM due to a kernel-dispatch bug in the FP8 block-quant
# path (see docs/findings/cardd_deeper_kernel_dispatch_blocker_2026_05_26.md).
# Awaiting vllm-project/vllm#43564 resolution. Card D's published H200 / B300
# benchmarks remain valid on the older jasl/vllm@abad5dc71 build they were
# measured against.
#
# For batched thinking-mode workloads on RTX PRO 6000 today, use the NVFP4
# sibling: canada-quant/dsv4-flash-nvfp4-fp8-mtp (which works cleanly with
# the same patch series).
set -euo pipefail

ARTIFACT="canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP"
VLLM_REPO="https://github.com/jasl/vllm.git"
VLLM_PIN="a02a3778f"
VLLM_BRANCH="ds4-sm120-preview-dev"
VLLM_SRC="${VLLM_SRC:-$HOME/src/vllm}"
VENV="${VENV:-$HOME/venv-serve}"
SCRATCH="${SCRATCH:-/scratch}"

cat <<EOF
================================================================
Card D (W4A16-MTP) RTX PRO 6000 install — $(date)

⚠  CURRENT SHIPPING STATE: vLLM kernel-dispatch issue prevents this
   artifact from running on current upstream vLLM. The build will
   succeed but serve will fail. See:
   docs/findings/cardd_deeper_kernel_dispatch_blocker_2026_05_26.md
   docs/findings/cardd_artifact_weight_scale_naming_blocker_2026_05_25.md

   For batched thinking-mode on RTX PRO 6000, use the NVFP4 sibling:
   canada-quant/dsv4-flash-nvfp4-fp8-mtp

   Card D's H200 / B300 benchmarks remain valid on jasl/vllm@abad5dc71.

================================================================
EOF

# Delegate to the shared install template — the build and patch series are
# identical to Card B. Only the artifact differs.
# Use Card B's script as the canonical implementation; this script is a
# CARD=D wrapper.
SHARED_INSTALL_URL="https://raw.githubusercontent.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/main/scripts/install_rtx6000pro.sh"

ARTIFACT="$ARTIFACT" REPO_ROOT="$(dirname "$(realpath "$0")")/.." curl -sL "$SHARED_INSTALL_URL" | bash
