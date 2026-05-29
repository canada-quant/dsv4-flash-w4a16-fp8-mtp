#!/usr/bin/env bash
# bench_matrix.sh — full W4A16 bench suite against a running vllm serve.
#
# Drives every test we need to publish: AIME-2024 thinking at c=1/2/4, GSM8K-50
# at c=8, throughput sweep bs=1/4/8/16 random 256/256 + bs=1/4 random 1024/1024.
# All AIME runs set max_tokens to the full max_model_len budget (minus prompt
# room) so reasoning is never truncated — quality scores represent the model's
# true capability, not the cap we picked.
#
# Usage (inside container):
#   TAG=tp2_32k bash /workspace/scripts/bench_matrix.sh
#   TAG=tp4_32k bash /workspace/scripts/bench_matrix.sh
#
# Env:
#   TAG               output filename suffix (required)
#   BASE_URL          vllm endpoint (default http://127.0.0.1:8000)
#   MODEL_NAME        served model name (default DSV4-W4A16-FP8-MTP)
#   MODEL_ID          repo id for tokenizer in bench serve (default canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP)
#   MAX_MODEL_LEN     window cap, used to size AIME max_tokens (default 32768)
#   OUT_DIR           bench-out subdir (default /workspace/bench-out)
#   AIME_CONCS        space-separated concurrencies for AIME (default "1 2 4")
#   BS_RANDOM_256     space-separated batch sizes for 256/256 sweep (default "1 4 8 16")
#   BS_RANDOM_1024    space-separated batch sizes for 1024/1024 sweep (default "1 4")
#   SKIP_AIME / SKIP_GSM / SKIP_THROUGHPUT — set to 1 to skip a phase

set -uo pipefail

TAG="${TAG:?TAG required (e.g. tp2_32k or tp4_32k)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_NAME="${MODEL_NAME:-DSV4-W4A16-FP8-MTP}"
MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OUT_DIR="${OUT_DIR:-/workspace/bench-out}"
AIME_CONCS="${AIME_CONCS:-1 2 4}"
BS_RANDOM_256="${BS_RANDOM_256:-1 4 8 16}"
BS_RANDOM_1024="${BS_RANDOM_1024:-1 4}"

# AIME max_tokens budget: leave ~500 tokens for prompt overhead
AIME_MAX_TOKENS=$(( MAX_MODEL_LEN - 500 ))

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/${TAG}_SUMMARY.md"
echo "# Bench matrix — $TAG ($(date -u +%Y-%m-%dT%H:%M:%SZ))" > "$SUMMARY"
echo "Endpoint: $BASE_URL  Model: $MODEL_NAME  max_model_len=$MAX_MODEL_LEN  AIME max_tokens=$AIME_MAX_TOKENS" >> "$SUMMARY"
echo >> "$SUMMARY"

log() { echo "[bench $(date -u +%H:%M:%S)] $*" | tee -a "$OUT_DIR/${TAG}.runlog"; }
log "TAG=$TAG MAX_MODEL_LEN=$MAX_MODEL_LEN AIME_MAX_TOKENS=$AIME_MAX_TOKENS"

# ---- Phase 1: AIME-30 thinking at c=1, c=2, c=4 ----
if [ -z "${SKIP_AIME:-}" ]; then
    echo "## AIME-2024 thinking-mode (max_tokens=$AIME_MAX_TOKENS, n=30)" >> "$SUMMARY"
    echo "| c | correct/30 | errors | stop | length | MTP accept | wall s |" >> "$SUMMARY"
    echo "|---|---|---|---|---|---|---|" >> "$SUMMARY"
    for C in $AIME_CONCS; do
        OUT="$OUT_DIR/${TAG}_aime30_c${C}_thinking.json"
        LOG="$OUT_DIR/${TAG}_aime30_c${C}_thinking.log"
        log "AIME-30 c=$C thinking → $OUT"
        python3 /workspace/scripts/aime_thinking_bench.py \
            --base-url "$BASE_URL" --model "$MODEL_NAME" \
            --concurrency "$C" --max-tokens "$AIME_MAX_TOKENS" \
            --out "$OUT" > "$LOG" 2>&1 || log "  c=$C aime failed (see $LOG)"
        if [ -f "$OUT" ]; then
            python3 - "$OUT" "$C" >> "$SUMMARY" <<'PYEOF'
import json, sys
p, c = sys.argv[1], sys.argv[2]
d = json.load(open(p))
t = d["totals"]; m = d["mtp_delta"]; fr = t["finish_reasons"]
acc = m.get("acceptance_rate")
acc_s = f"{acc*100:.2f}%" if acc is not None else "n/a"
print(f"| {c} | {t['correct']}/30 | {t['errors']} | {fr.get('stop',0)} | {fr.get('length',0)} | {acc_s} | {t['wallclock_s']:.0f} |")
PYEOF
        else
            echo "| $C | FAIL | — | — | — | — | — |" >> "$SUMMARY"
        fi
    done
    echo >> "$SUMMARY"
fi

# ---- Phase 2: GSM8K-50 c=8 ----
if [ -z "${SKIP_GSM:-}" ]; then
    OUT="$OUT_DIR/${TAG}_gsm8k50_c8.json"
    LOG="$OUT_DIR/${TAG}_gsm8k50_c8.log"
    log "GSM8K-50 c=8 → $OUT"
    lm_eval --model local-completions \
        --tasks gsm8k \
        --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
        --num_fewshot 8 \
        --limit 50 \
        --output_path "$OUT" > "$LOG" 2>&1 || log "  GSM8K failed"
    echo "## GSM8K-50 c=8 (strict-match)" >> "$SUMMARY"
    if grep -q "strict-match" "$LOG" 2>/dev/null; then
        grep -A1 "strict-match" "$LOG" | tail -2 | sed 's/^/    /' >> "$SUMMARY"
    else
        echo "    see $LOG" >> "$SUMMARY"
    fi
    echo >> "$SUMMARY"
fi

# ---- Phase 3: throughput sweep (random 256/256) ----
if [ -z "${SKIP_THROUGHPUT:-}" ]; then
    echo "## Throughput sweep — random 256/256 (MTP-on)" >> "$SUMMARY"
    echo "| bs | output tok/s | TPOT median ms | TPOT p99 ms | wall s |" >> "$SUMMARY"
    echo "|---|---|---|---|---|" >> "$SUMMARY"
    for BS in $BS_RANDOM_256; do
        NREQ=$((BS * 4))
        OUT="$OUT_DIR/${TAG}_bench_random256_bs${BS}.json"
        LOG="$OUT_DIR/${TAG}_bench_random256_bs${BS}.log"
        log "throughput random 256/256 bs=$BS n=$NREQ"
        vllm bench serve \
            --base-url "$BASE_URL" \
            --model "$MODEL_NAME" \
            --tokenizer "$MODEL_ID" \
            --trust-remote-code \
            --dataset-name random --random-input-len 256 --random-output-len 256 \
            --num-prompts "$NREQ" --max-concurrency "$BS" \
            --save-result --result-dir "$OUT_DIR" \
            --result-filename "$(basename $OUT)" \
            > "$LOG" 2>&1 || log "  bs=$BS failed"
        if [ -f "$OUT" ]; then
            python3 - "$OUT" "$BS" >> "$SUMMARY" <<'PYEOF'
import json, sys
p, bs = sys.argv[1], sys.argv[2]
j = json.load(open(p))
ot = j.get("output_throughput", 0)
mtpot = 1000 * j.get("median_tpot_ms", 0) / 1000
ptpot = 1000 * j.get("p99_tpot_ms", 0) / 1000
dur = j.get("duration", 0)
print(f"| {bs} | {ot:.1f} | {mtpot:.2f} | {ptpot:.2f} | {dur:.1f} |")
PYEOF
        else
            echo "| $BS | FAIL | — | — | — |" >> "$SUMMARY"
        fi
    done
    echo >> "$SUMMARY"

    echo "## Throughput sweep — random 1024/1024 (MTP-on)" >> "$SUMMARY"
    echo "| bs | output tok/s | TPOT median ms | TPOT p99 ms | wall s |" >> "$SUMMARY"
    echo "|---|---|---|---|---|" >> "$SUMMARY"
    for BS in $BS_RANDOM_1024; do
        NREQ=$((BS * 4))
        OUT="$OUT_DIR/${TAG}_bench_random1024_bs${BS}.json"
        LOG="$OUT_DIR/${TAG}_bench_random1024_bs${BS}.log"
        log "throughput random 1024/1024 bs=$BS n=$NREQ"
        vllm bench serve \
            --base-url "$BASE_URL" \
            --model "$MODEL_NAME" \
            --tokenizer "$MODEL_ID" \
            --trust-remote-code \
            --dataset-name random --random-input-len 1024 --random-output-len 1024 \
            --num-prompts "$NREQ" --max-concurrency "$BS" \
            --save-result --result-dir "$OUT_DIR" \
            --result-filename "$(basename $OUT)" \
            > "$LOG" 2>&1 || log "  bs=$BS failed"
        if [ -f "$OUT" ]; then
            python3 - "$OUT" "$BS" >> "$SUMMARY" <<'PYEOF'
import json, sys
p, bs = sys.argv[1], sys.argv[2]
j = json.load(open(p))
ot = j.get("output_throughput", 0)
mtpot = 1000 * j.get("median_tpot_ms", 0) / 1000
ptpot = 1000 * j.get("p99_tpot_ms", 0) / 1000
dur = j.get("duration", 0)
print(f"| {bs} | {ot:.1f} | {mtpot:.2f} | {ptpot:.2f} | {dur:.1f} |")
PYEOF
        else
            echo "| $BS | FAIL | — | — | — |" >> "$SUMMARY"
        fi
    done
    echo >> "$SUMMARY"
fi

log "DONE — summary at $SUMMARY"
cat "$SUMMARY"
