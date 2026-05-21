#!/usr/bin/env bash
# Option B serve smoke — validate the calibration artifact end-to-end through vLLM.
#
# Pre-condition: smoke iter 7 (or full Phase 2) wrote an artifact at
# /scratch/weights/w4a16-fp8-mtp-smoke (or /scratch/weights/w4a16-fp8-mtp-gptq
# for full Phase 2). Pass the path as arg 1.
#
# What this test exercises (the world-firsts as well as the routine):
#   1. Our save_pretrained output → vLLM loader (NEVER tested)
#   2. Our config.json (with layer_types truncation) → vLLM model construction
#   3. W4A16 packed weights → Marlin kernels (proven on predecessor, not ours)
#   4. FP8_BLOCK attention → FlashMLA (proven on predecessor, not ours)
#   5. BF16 MTP weights → vLLM's DeepSeekV4MTP class (NEVER done before — value prop)
#   6. --speculative-config method=mtp num_speculative_tokens=2 → spec-decode fires
#
# Acceptance criteria (all must hold):
#   - vLLM serve health endpoint returns 200 within 5 min
#   - 4 of 4 chat-smoke prompts return coherent responses (not garbled int4)
#   - vLLM metrics show NON-ZERO MTP draft tokens generated AND accepted
#     (specifically: `spec_decode_num_draft_tokens` > 0 and acceptance > 50%)
#
# If responses come back coherent BUT spec-decode metrics show 0% acceptance / 0
# drafts, MTP is loading but not being used — that's the subtle failure mode
# where the artifact looks fine on stdout but the value prop is silently broken.

set -euo pipefail
MODEL_PATH="${1:-/scratch/weights/w4a16-fp8-mtp-smoke}"
PORT="${2:-8000}"

source ~/venv-serve/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="9.0a"
export CUDA_VISIBLE_DEVICES=0,1   # TP=2 on GPUs 0,1 — matches predecessor's published config

echo "[option-b] serving $MODEL_PATH on port $PORT (TP=2)..."

# Launch in background so we can run the chat-smoke probe + curl metrics
nohup vllm serve "$MODEL_PATH" \
    --served-model-name DSV4-W4A16-FP8-MTP \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 32768 \
    --max-num-seqs 2 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" \
    > /tmp/vllm_optionb.log 2>&1 &
SERVE_PID=$!
echo "[option-b] vllm PID=$SERVE_PID, log at /tmp/vllm_optionb.log"

# Wait for /health with 15-min timeout (vLLM cold-load of 156GB W4A16+FP8 model
# into TP=2 ran ~7-8 min in the smoke iter 8 dry-run; predecessor docs suggest
# even longer at higher TP. 15 min is comfortable headroom.)
echo "[option-b] waiting for /health (max 15 min)..."
deadline=$(($(date +%s) + 900))
until curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if [ $(date +%s) -gt $deadline ]; then
        echo "[option-b] FAIL — /health never returned 200 in 5 min"
        tail -50 /tmp/vllm_optionb.log
        kill -9 $SERVE_PID 2>/dev/null
        exit 2
    fi
    sleep 5
done
echo "[option-b] /health OK"

# Confirm MTP was loaded — vLLM logs should mention DeepSeekV4MTP construction
if ! grep -qi "DeepSeekV4MTP\|spec.*decode.*method=mtp\|speculative" /tmp/vllm_optionb.log; then
    echo "[option-b] WARN — no MTP/spec-decode mentions in vLLM startup log; check manually"
    grep -iE "MTP|spec|speculative|draft" /tmp/vllm_optionb.log | head
fi

# Chat-smoke quick: 4 prompts
echo "[option-b] running chat-smoke (4 prompts)..."
PROMPTS=(
    "What is the capital of France?"
    "Write a one-sentence definition of recursion."
    "Solve: 17 * 23 = ?"
    "Python one-liner to reverse a string?"
)
for i in "${!PROMPTS[@]}"; do
    P="${PROMPTS[$i]}"
    echo "[option-b]   prompt $((i+1))/4: $P"
    R=$(curl -fsS -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --arg p "$P" '{model: "DSV4-W4A16-FP8-MTP", messages: [{role: "user", content: $p}], max_tokens: 200, temperature: 0.0}')")
    CONTENT=$(echo "$R" | jq -r '.choices[0].message.content // .choices[0].text // "<no content>"')
    echo "[option-b]   reply: ${CONTENT:0:200}"
    echo "---"
done

# Pull spec-decode metrics — this is the gate
echo "[option-b] pulling spec-decode metrics..."
curl -fsS "http://localhost:$PORT/metrics" 2>/dev/null | grep -iE "spec_decode|draft|accept" | head -30

# Cleanup
echo "[option-b] killing server"
kill -TERM $SERVE_PID 2>/dev/null || true
sleep 3
kill -9 $SERVE_PID 2>/dev/null || true

echo "[option-b] DONE — review responses + spec-decode metrics above"
echo "[option-b] success criteria:"
echo "  (a) all 4 responses coherent (not garbled int4 nonsense)"
echo "  (b) spec_decode_num_draft_tokens > 0 in metrics"
echo "  (c) spec_decode_acceptance > 0.5 (50% acceptance is the floor sanity)"
echo "  (a) + (b) + (c) all green → launch Phase 2"
echo "  any red → stop, debug before committing 14h"
