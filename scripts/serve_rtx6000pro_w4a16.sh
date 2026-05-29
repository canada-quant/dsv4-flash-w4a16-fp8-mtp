#!/usr/bin/env bash
# serve_rtx6000pro_w4a16.sh — serve the W4A16-FP8-MTP artifact on RTX PRO 6000
# Blackwell Server Edition (SM 12.0), TP=2 or TP=4, env-driven.
#
# Run via the canada-quant/dsv4-rtx6000pro:v3 image. All the SM 12.0-specific
# fixes (jasl/vllm@27fd665b + canada-quant cherry-pick for BF16 MTP + 13-layer
# dep patches + cute.arch.fmin shim) are baked into the image.
#
# Usage inside container:
#   TP=2 MAX_NUM_SEQS=8  MAX_MODEL_LEN=16384 bash scripts/serve_rtx6000pro_w4a16.sh
#   TP=4 MAX_NUM_SEQS=16 MAX_MODEL_LEN=32768 bash scripts/serve_rtx6000pro_w4a16.sh
#
# Important env knobs:
#   TP                       2 or 4 (default 2)
#   MAX_NUM_SEQS             concurrent in-flight sequences (default 8)
#   MAX_MODEL_LEN            context window cap (default 16384)
#   MAX_NUM_BATCHED_TOKENS   per-step token budget (auto: max(4096, 512*MAX_NUM_SEQS))
#   GPU_MEM_UTIL             VRAM fraction (default 0.92)
#   CUDAGRAPH_SIZES          JSON list of batch sizes to capture (auto: matches MAX_NUM_SEQS)
#
# Notes:
#   - VLLM_TEST_FORCE_FP8_MARLIN=1 routes attention block-FP8 layers through
#     Marlin (the only working SM 12.0 path); harmless for W4A16 routed experts
#     which already use Marlin's native INT4 kernel.
#   - VLLM_USE_LAYERNAME=0 avoids the Inductor MoE lowering crash on FakeScriptObject
#     WITHOUT having to fall back to --enforce-eager (CUDA graphs stay enabled).

set -uo pipefail

MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP}"
MODEL_NAME="${MODEL_NAME:-DSV4-W4A16-FP8-MTP}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# Auto-size batched-tokens budget so MAX_NUM_SEQS sequences can run at once
# without being chunk-throttled. floor: 4096 (the prior default).
DEFAULT_BATCHED=$(( MAX_NUM_SEQS * 512 ))
if [ "$DEFAULT_BATCHED" -lt 4096 ]; then DEFAULT_BATCHED=4096; fi
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$DEFAULT_BATCHED}"

# Auto-grow cudagraph capture sizes to cover MAX_NUM_SEQS (powers of 2 up to it).
# Override with CUDAGRAPH_SIZES='[1,2,4,8,16]' if you want custom shape coverage.
if [ -z "${CUDAGRAPH_SIZES:-}" ]; then
    sizes="1"
    s=2
    while [ "$s" -le "$MAX_NUM_SEQS" ]; do
        sizes="$sizes,$s"
        s=$(( s * 2 ))
    done
    CUDAGRAPH_SIZES="[$sizes]"
fi
CUDAGRAPH_MAX="$MAX_NUM_SEQS"

echo "[serve] W4A16-FP8-MTP on ${TP}× RTX PRO 6000 (MTP k=1, CUDA graphs ON)"
echo "[serve]   model:                   $MODEL_ID"
echo "[serve]   tensor_parallel_size:    $TP"
echo "[serve]   max_model_len:           $MAX_MODEL_LEN"
echo "[serve]   max_num_seqs:            $MAX_NUM_SEQS"
echo "[serve]   max_num_batched_tokens:  $MAX_NUM_BATCHED_TOKENS"
echo "[serve]   gpu_memory_utilization:  $GPU_MEM_UTIL"
echo "[serve]   cudagraph_capture_sizes: $CUDAGRAPH_SIZES (max $CUDAGRAPH_MAX)"

export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_USE_LAYERNAME=0
export HF_HUB_ENABLE_HF_TRANSFER=1

vllm serve "$MODEL_ID" \
    --tensor-parallel-size "$TP" \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --disable-custom-all-reduce \
    --compilation-config "{\"cudagraph_capture_sizes\":${CUDAGRAPH_SIZES},\"max_cudagraph_capture_size\":${CUDAGRAPH_MAX}}" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
