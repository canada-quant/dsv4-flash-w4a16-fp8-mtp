#!/usr/bin/env bash
# Phase 5 — single-instance smoke serve on a single GPU pair (TP=2).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_b300_tp2.sh /scratch/weights/w4a16-fp8-mtp 8000
#
# Notes:
#   - Pinned to ONE GPU pair via CUDA_VISIBLE_DEVICES so the Marlin MoE TP>2
#     bug (vLLM #41511, open as of 2026-05-19) cannot bite us. For a full
#     8-GPU box, run 4 instances on disjoint pairs (Phase 7).
#   - --speculative-config method=mtp num_speculative_tokens=2 enables the
#     MTP draft path that motivates this whole repo.
#   - B300 env: TORCH_CUDA_ARCH_LIST=10.0a, do NOT set
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (breaks Blackwell
#     allreduce), do NOT use VLLM_TRITON_MLA_SPARSE* (SM12x-only).

set -euo pipefail

MODEL_PATH="${1:?usage: $0 <model_path> <port>}"
PORT="${2:?usage: $0 <model_path> <port>}"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="10.0a"
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=0
# DO NOT SET:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   (breaks Blackwell allreduce)
#   VLLM_TRITON_MLA_SPARSE*                             (SM12x-only)

# Source the serve venv if not already in one
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f /data/venv-serve/bin/activate ]]; then
    source /data/venv-serve/bin/activate
fi

exec vllm serve "$MODEL_PATH" \
    --served-model-name DSV4-W4A16-FP8-MTP deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 524288 \
    --max-num-seqs 2 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --trust-remote-code --host 0.0.0.0 --port "$PORT"
