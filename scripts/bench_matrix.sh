#!/usr/bin/env bash
# bench_matrix.sh — full W4A16 bench suite against a running vllm serve.
#
# AIME-2024 phase covers two axes:
#   - thinking-mode comparison at c=4 across {chat, high, max} (3 runs)
#   - thinking=high single-shot reference at c=1 (1 run)
# All AIME runs set max_tokens to max_model_len-500 so reasoning runs to natural
# stop. DS-V4 reasoning parser aliases low/medium → high, so chat/high/max are
# the only effective distinct modes.
#
# Throughput: vllm bench serve, random 256/256 at bs=1/4/8/16 and 1024/1024 at
# bs=1/4. GSM8K-50 c=8 strict-match.
#
# Usage (inside container, after vllm is up):
#   TAG=tp2_64k MAX_MODEL_LEN=65536 bash /workspace/scripts/bench_matrix.sh
#   TAG=tp4_64k MAX_MODEL_LEN=65536 bash /workspace/scripts/bench_matrix.sh
#
# Env knobs:
#   TAG               output filename suffix (required)
#   BASE_URL          vllm endpoint (default http://127.0.0.1:8000)
#   MODEL_NAME        served name (default DSV4-W4A16-FP8-MTP)
#   MODEL_ID          tokenizer ref (default canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP)
#   MAX_MODEL_LEN     window cap (default 65536) — drives AIME max_tokens
#   OUT_DIR           bench-out subdir (default /workspace/bench-out)
#   AIME_MODES        space-separated modes for AIME c=4 sweep (default "chat high max")
#   AIME_C1_MODE      mode for the c=1 single-shot reference run (default high)
#   BS_RANDOM_256     batch sizes for 256/256 sweep (default "1 4 8 16")
#   BS_RANDOM_1024    batch sizes for 1024/1024 sweep (default "1 4")
#   SKIP_AIME / SKIP_GSM / SKIP_THROUGHPUT — set to 1 to skip a phase

set -uo pipefail

TAG="${TAG:?TAG required (e.g. tp2_64k or tp4_64k)}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_NAME="${MODEL_NAME:-DSV4-W4A16-FP8-MTP}"
MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
OUT_DIR="${OUT_DIR:-/workspace/bench-out}"
AIME_MODES="${AIME_MODES:-chat high max}"
AIME_C1_MODE="${AIME_C1_MODE:-high}"
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

# ---- Phase 1A: AIME-30 thinking-mode sweep at c=4 ----
if [ -z "${SKIP_AIME:-}" ]; then
    echo "## AIME-2024 thinking-mode sweep at c=4 (max_tokens=$AIME_MAX_TOKENS, n=30)" >> "$SUMMARY"
    echo "| mode | correct/30 | errors | stop | length | MTP accept | wall s |" >> "$SUMMARY"
    echo "|---|---|---|---|---|---|---|" >> "$SUMMARY"
    for MODE in $AIME_MODES; do
        OUT="$OUT_DIR/${TAG}_aime30_c4_${MODE}.json"
        LOG="$OUT_DIR/${TAG}_aime30_c4_${MODE}.log"
        log "AIME-30 c=4 mode=$MODE → $OUT"
        python3 /workspace/scripts/aime_thinking_bench.py \
            --base-url "$BASE_URL" --model "$MODEL_NAME" \
            --concurrency 4 --max-tokens "$AIME_MAX_TOKENS" \
            --reasoning-effort "$MODE" \
            --out "$OUT" > "$LOG" 2>&1 || log "  mode=$MODE aime failed (see $LOG)"
        if [ -f "$OUT" ]; then
            python3 - "$OUT" "$MODE" >> "$SUMMARY" <<'PYEOF'
import json, sys
p, mode = sys.argv[1], sys.argv[2]
d = json.load(open(p))
t = d["totals"]; m = d["mtp_delta"]; fr = t["finish_reasons"]
acc = m.get("acceptance_rate")
acc_s = f"{acc*100:.2f}%" if acc is not None else "n/a"
print(f"| {mode} | {t['correct']}/30 | {t['errors']} | {fr.get('stop',0)} | {fr.get('length',0)} | {acc_s} | {t['wallclock_s']:.0f} |")
PYEOF
        else
            echo "| $MODE | FAIL | — | — | — | — | — |" >> "$SUMMARY"
        fi
    done
    echo >> "$SUMMARY"

    # ---- Phase 1B: AIME-30 c=1 reference (single-shot quality) ----
    OUT="$OUT_DIR/${TAG}_aime30_c1_${AIME_C1_MODE}.json"
    LOG="$OUT_DIR/${TAG}_aime30_c1_${AIME_C1_MODE}.log"
    log "AIME-30 c=1 single-shot mode=$AIME_C1_MODE → $OUT"
    python3 /workspace/scripts/aime_thinking_bench.py \
        --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --concurrency 1 --max-tokens "$AIME_MAX_TOKENS" \
        --reasoning-effort "$AIME_C1_MODE" \
        --out "$OUT" > "$LOG" 2>&1 || log "  c=1 aime failed"
    echo "## AIME-2024 c=1 single-shot (mode=$AIME_C1_MODE, max_tokens=$AIME_MAX_TOKENS)" >> "$SUMMARY"
    echo "| c | correct/30 | errors | stop | length | MTP accept | wall s |" >> "$SUMMARY"
    echo "|---|---|---|---|---|---|---|" >> "$SUMMARY"
    if [ -f "$OUT" ]; then
        python3 - "$OUT" >> "$SUMMARY" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d["totals"]; m = d["mtp_delta"]; fr = t["finish_reasons"]
acc = m.get("acceptance_rate")
acc_s = f"{acc*100:.2f}%" if acc is not None else "n/a"
print(f"| 1 | {t['correct']}/30 | {t['errors']} | {fr.get('stop',0)} | {fr.get('length',0)} | {acc_s} | {t['wallclock_s']:.0f} |")
PYEOF
    else
        echo "| 1 | FAIL | — | — | — | — | — |" >> "$SUMMARY"
    fi
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
    echo "## GSM8K-50 c=8 (strict-match, 8-shot)" >> "$SUMMARY"
    if grep -q "strict-match" "$LOG" 2>/dev/null; then
        grep -A1 "strict-match\|exact_match" "$LOG" | tail -6 | sed 's/^/    /' >> "$SUMMARY"
    else
        echo "    see $LOG" >> "$SUMMARY"
    fi
    echo >> "$SUMMARY"
fi

# ---- Phase 3: throughput sweep ----
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
