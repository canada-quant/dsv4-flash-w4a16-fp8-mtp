#!/usr/bin/env bash
# serve_rtx6000pro_tp2_w4a16.sh — serve the W4A16-FP8-MTP sibling artifact on
# 2× RTX PRO 6000 using the same canada-quant/dsv4-rtx6000pro:v3 image.
#
# Why this script exists: the Docker recipe was built around the NVFP4 model
# but the underlying patches (jasl/vllm + canada-quant MTP cherry-pick + 13
# layer fixes) are all model-card-agnostic. They target SM 12.0 + DeepSeek-V4
# architecture, not a specific quantization. W4A16 is the canonical RTX PRO
# 6000 choice today (~8% smaller footprint, slightly better throughput on
# Marlin INT4 vs Marlin FP4-dequant) — this script lets users pick that path
# with the same image.
#
# Differences from serve_rtx6000pro_tp2.sh (NVFP4 default):
#   - MODEL_ID points at the W4A16 sibling
#   - VLLM_TEST_FORCE_FP8_MARLIN env retained (harmless for W4A16; routes the
#     attention layers' block-FP8 path through Marlin, which is the only
#     working path on SM 12.0 — same as the NVFP4 script)

set -uo pipefail

MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP}"
MODEL_NAME="${MODEL_NAME:-DSV4-W4A16-FP8-MTP}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"

echo "[serve] W4A16-FP8-MTP on 2× RTX PRO 6000 TP=2 (MTP k=1, CUDA graphs ON)"
echo "[serve]   model: $MODEL_ID"
echo "[serve]   max_model_len: $MAX_MODEL_LEN   max_num_seqs: $MAX_NUM_SEQS"

export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_USE_LAYERNAME=0
export HF_HUB_ENABLE_HF_TRANSFER=1

vllm serve "$MODEL_ID" \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.92 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --disable-custom-all-reduce \
    --compilation-config '{"cudagraph_capture_sizes":[1,2],"max_cudagraph_capture_size":2}' \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT"
