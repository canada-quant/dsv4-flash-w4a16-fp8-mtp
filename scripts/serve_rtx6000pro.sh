#!/usr/bin/env bash
# Serve the W4A16+FP8+MTP artifact on RTX PRO 6000 Blackwell (SM 12.0).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1   bash scripts/serve_rtx6000pro.sh <model_path> <port> <tp>
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/serve_rtx6000pro.sh <model_path> <port> <tp>
#
# Notes:
#   - Marlin MoE TP>2 bug (vLLM #41511, OPEN as of 2026-05) — TP=4 may
#     fail at load with W4A16 expert scale-sharding errors. If it
#     does, document the failure and stop; that's a known upstream
#     block, not a model issue.
#   - C15 (DeepGemm `next_n` assertion) caps num_speculative_tokens=1
#     on Hopper; SM12 attention.hpp paths also assert next_n==1. k=1
#     is the only viable spec config here.
#   - SM12 = Blackwell consumer/server family. DO use
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (per H200 build);
#     it's the *B300* (SM 10.0 datacenter) family that breaks on this flag.

set -euo pipefail

MODEL_PATH="${1:?usage: $0 <model_path> <port> <tp>}"
PORT="${2:?usage: $0 <model_path> <port> <tp>}"
TP="${3:?usage: $0 <model_path> <port> <tp>}"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="12.0a"
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "$HOME/venv-serve/bin/activate" ]]; then
    source "$HOME/venv-serve/bin/activate"
fi

exec vllm serve "$MODEL_PATH" \
    --served-model-name DSV4-W4A16-FP8-MTP deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash \
    --tensor-parallel-size "$TP" \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 4096 \
    --max-num-seqs 16 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --enforce-eager \
    --trust-remote-code --host 0.0.0.0 --port "$PORT"
