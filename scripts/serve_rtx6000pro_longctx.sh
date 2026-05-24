#!/usr/bin/env bash
# Same as serve_rtx6000pro.sh but with --max-model-len bumped to 32768
# so we can run context-length sweep benchmarks. KV cache budget drops
# proportionally — reduces --max-num-seqs from 16 to 4 to keep total
# token budget similar.

set -uo pipefail

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
    --max-model-len 32768 \
    --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --disable-custom-all-reduce \
    --trust-remote-code --host 0.0.0.0 --port "$PORT"
