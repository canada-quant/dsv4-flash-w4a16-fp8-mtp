#!/usr/bin/env bash
# Comprehensive Card B benchmark suite, driven from inside the canada-quant/dsv4-rtx6000pro
# Docker image. Brings up vllm serve, runs throughput + reasoning + concurrency + reasoning-
# budget sweeps, tears down. Writes everything to a single timestamped output dir.
#
# Usage (on RTX PRO 6000 host with the image built):
#
#   docker run --gpus all --rm -it \
#     --shm-size=16g --ipc=host \
#     --network host \
#     -v /opt/dlami/nvme/hf-cache:/root/.cache/huggingface \
#     -v $(pwd)/bench-out:/workspace/bench-out \
#     -v $(pwd)/scripts:/workspace/scripts:ro \
#     canada-quant/dsv4-rtx6000pro:v3 \
#     bash /workspace/scripts/bench_docker_full.sh
#
# Outputs land in /workspace/bench-out/<timestamp>/.

set -uo pipefail

MODEL_ID="${MODEL_ID:-canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP}"
MODEL_NAME="${MODEL_NAME:-DSV4-NVFP4-FP8-MTP}"
TP="${TP:-4}"
MTP_K="${MTP_K:-1}"  # SM 12.0 caps spec at k=1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

TS="$(date -u +%Y-%m-%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-/workspace/bench-out/${TS}_cardb_rtxpro6000_tp${TP}}"
mkdir -p "$OUT_DIR"
echo "[bench] writing to $OUT_DIR"

log() { echo "[bench $(date -u +%H:%M:%S)] $*" | tee -a "$OUT_DIR/run.log" ; }

cleanup() {
    if [ -n "${SERVE_PID:-}" ] && kill -0 "$SERVE_PID" 2>/dev/null; then
        log "tearing down vllm serve PID=$SERVE_PID"
        kill -TERM "$SERVE_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$SERVE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---- environment snapshot ----
{
    echo "=== nvidia-smi ==="; nvidia-smi -L
    echo "=== python ==="; python3 --version
    echo "=== torch ==="; python3 -c "import torch; print(torch.__version__, 'cuda=', torch.cuda.is_available())"
    echo "=== vllm ==="; python3 -c "import vllm; print(vllm.__version__, vllm.__file__)"
    echo "=== env ==="; env | grep -E "VLLM|HF_|CUDA" | sort
} > "$OUT_DIR/env.txt" 2>&1
log "env snapshot saved to $OUT_DIR/env.txt"

# ---- start vllm serve in background ----
log "starting vllm serve TP=$TP, mtp k=$MTP_K, max_model_len=$MAX_MODEL_LEN"
VLLM_TEST_FORCE_FP8_MARLIN=1 \
nohup vllm serve "$MODEL_ID" \
    --tensor-parallel-size "$TP" \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_K}" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --disable-custom-all-reduce \
    --enforce-eager \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --host 0.0.0.0 --port "$PORT" \
    > "$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!
log "vllm serve PID=$SERVE_PID, logs at $OUT_DIR/serve.log"

# ---- wait for /v1/models ----
log "waiting for $BASE_URL/v1/models to respond (timeout 1200s)"
for i in $(seq 1 240); do
    if curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
        log "serve ready after ${i}x5s = $((i*5))s"
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        log "FATAL: vllm serve died during startup. Tail of serve.log:"
        tail -50 "$OUT_DIR/serve.log" | tee -a "$OUT_DIR/run.log"
        exit 1
    fi
    sleep 5
done
if ! curl -sf "$BASE_URL/v1/models" >/dev/null 2>&1; then
    log "FATAL: /v1/models never came up. Tail of serve.log:"
    tail -50 "$OUT_DIR/serve.log" | tee -a "$OUT_DIR/run.log"
    exit 1
fi

curl -s "$BASE_URL/v1/models" | python3 -m json.tool > "$OUT_DIR/models.json"
log "model endpoint up; bench starting"

# ---- chat smoke ----
log "phase 1: chat smoke (4 prompts)"
python3 - "$BASE_URL" "$MODEL_NAME" >"$OUT_DIR/01_chat_smoke.log" 2>&1 <<'PYEOF' || log "chat smoke had errors"
import sys, json, urllib.request
base, model = sys.argv[1], sys.argv[2]
prompts = [
    ("simple_math", "What is 17 * 23? Answer with just the number."),
    ("code", "Write a Python function to reverse a string."),
    ("explain", "Explain quantum entanglement in 2 sentences."),
    ("chain", "If a train travels 60 mph for 2.5 hours, how far does it go? Show work."),
]
for tag, p in prompts:
    payload = {"model": model, "messages": [{"role":"user","content":p}], "max_tokens": 200, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", method="POST",
                                 headers={"Content-Type":"application/json"},
                                 data=json.dumps(payload).encode())
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        content = r["choices"][0]["message"].get("content","") or r["choices"][0]["message"].get("reasoning_content","")
        usage = r.get("usage", {})
        print(f"[{tag}] {len(content)}c, completion_tokens={usage.get('completion_tokens')}, finish={r['choices'][0].get('finish_reason')}")
        print(f"    -> {content[:200]!r}")
    except Exception as e:
        print(f"[{tag}] FAIL: {e}")
PYEOF
cat "$OUT_DIR/01_chat_smoke.log" | tee -a "$OUT_DIR/run.log"

# ---- throughput sweep (vllm bench serve, random 256/256) ----
log "phase 2: throughput sweep (vllm bench serve, random 256/256)"
for BS in 1 4 16 32; do
    NREQ=$((BS * 4))
    log "  bs=$BS (n=$NREQ)"
    vllm bench serve \
        --base-url "$BASE_URL" \
        --model "$MODEL_NAME" \
        --tokenizer "$MODEL_ID" \
        --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 256 \
        --num-prompts "$NREQ" --max-concurrency "$BS" \
        --save-result --result-dir "$OUT_DIR" \
        --result-filename "02_bench_random256_bs${BS}.json" \
        > "$OUT_DIR/02_bench_random256_bs${BS}.log" 2>&1 \
        || log "  bs=$BS failed (see 02_bench_random256_bs${BS}.log)"
done

# ---- throughput long context (1024/1024) ----
log "phase 3: throughput long (1024/1024) at bs=1,4"
for BS in 1 4; do
    NREQ=$((BS * 4))
    log "  bs=$BS"
    vllm bench serve \
        --base-url "$BASE_URL" \
        --model "$MODEL_NAME" \
        --tokenizer "$MODEL_ID" \
        --trust-remote-code \
        --dataset-name random --random-input-len 1024 --random-output-len 1024 \
        --num-prompts "$NREQ" --max-concurrency "$BS" \
        --save-result --result-dir "$OUT_DIR" \
        --result-filename "03_bench_random1024_bs${BS}.json" \
        > "$OUT_DIR/03_bench_random1024_bs${BS}.log" 2>&1 \
        || log "  bs=$BS failed"
done

# ---- AIME concurrency sweep (the marquee test) ----
log "phase 4: AIME-2024 thinking-mode concurrency sweep (c=1/2/4, max_tokens=16384)"
for C in 1 2 4; do
    log "  c=$C"
    python3 /workspace/scripts/aime_thinking_bench.py \
        --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --concurrency "$C" --max-tokens 16384 \
        --out "$OUT_DIR/04_aime30_c${C}_max16k.json" \
        > "$OUT_DIR/04_aime30_c${C}_max16k.log" 2>&1 \
        || log "  c=$C failed (see log)"
done

# ---- AIME reasoning-budget sweep (think_max) ----
log "phase 5: AIME-2024 thinking-mode reasoning-budget sweep (c=1, max_tokens variants)"
for MT in 8192 16384 32768; do
    if [ "$MT" -gt "$MAX_MODEL_LEN" ]; then
        log "  max_tokens=$MT exceeds MAX_MODEL_LEN=$MAX_MODEL_LEN — skipping"
        continue
    fi
    log "  max_tokens=$MT"
    python3 /workspace/scripts/aime_thinking_bench.py \
        --base-url "$BASE_URL" --model "$MODEL_NAME" \
        --concurrency 1 --max-tokens "$MT" \
        --out "$OUT_DIR/05_aime30_c1_max${MT}.json" \
        > "$OUT_DIR/05_aime30_c1_max${MT}.log" 2>&1 \
        || log "  max_tokens=$MT failed"
done

# ---- GSM8K-50 ----
log "phase 6: GSM8K-50 strict-match"
lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False" \
    --num_fewshot 8 \
    --limit 50 \
    --output_path "$OUT_DIR/06_gsm8k50.json" \
    > "$OUT_DIR/06_gsm8k50.log" 2>&1 \
    || log "  GSM8K-50 failed"

# ---- HumanEval ----
log "phase 7: HumanEval pass@1"
lm_eval --model local-completions \
    --tasks humaneval_instruct \
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False" \
    --confirm_run_unsafe_code \
    --output_path "$OUT_DIR/07_humaneval.json" \
    > "$OUT_DIR/07_humaneval.log" 2>&1 \
    || log "  HumanEval failed"

# ---- IFEval ----
log "phase 8: IFEval prompt-strict"
lm_eval --model local-completions \
    --tasks ifeval \
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False" \
    --output_path "$OUT_DIR/08_ifeval.json" \
    > "$OUT_DIR/08_ifeval.log" 2>&1 \
    || log "  IFEval failed"

# ---- summary ----
log "all phases complete; building summary"
python3 - "$OUT_DIR" > "$OUT_DIR/SUMMARY.md" 2>"$OUT_DIR/summary.err" <<'PYEOF' || log "summary build failed"
import json, os, sys, glob
from pathlib import Path
d = Path(sys.argv[1])
print(f"# Card B (NVFP4-FP8-MTP) benchmark suite — {d.name}\n")
print(f"Output dir: `{d}`\n")
print("## Env\n```")
print(open(d/"env.txt").read())
print("```\n")
print("## Throughput sweep — random 256/256, MTP-on\n")
print("| bs | concurrency | throughput tok/s | TPOT median ms | TPOT p99 ms | wall s |")
print("|---|---|---|---|---|---|")
for bs in (1, 4, 16, 32):
    f = d / f"02_bench_random256_bs{bs}.json"
    if not f.exists(): continue
    j = json.load(open(f))
    print(f"| {bs} | {j.get('num_prompts','?')} | {j.get('output_throughput',0):.1f} | {1000*j.get('median_tpot_ms',0)/1000:.2f} | {1000*j.get('p99_tpot_ms',0)/1000:.2f} | {j.get('duration',0):.1f} |")
print("\n## Throughput long-context — random 1024/1024, MTP-on\n")
print("| bs | throughput tok/s | TPOT median ms |")
print("|---|---|---|")
for bs in (1, 4):
    f = d / f"03_bench_random1024_bs{bs}.json"
    if not f.exists(): continue
    j = json.load(open(f))
    print(f"| {bs} | {j.get('output_throughput',0):.1f} | {1000*j.get('median_tpot_ms',0):.2f} |")
print("\n## AIME-2024 thinking-mode concurrency sweep (max_tokens=16384)\n")
print("| c | correct/30 | errors | stop | length | MTP accept | wall (s) |")
print("|---|---|---|---|---|---|---|")
for c in (1, 2, 4):
    f = d / f"04_aime30_c{c}_max16k.json"
    if not f.exists(): continue
    j = json.load(open(f))
    t, m = j["totals"], j["mtp_delta"]
    acc = m.get("acceptance_rate")
    print(f"| {c} | {t.get('correct','?')}/30 | {t.get('errors','?')} | {t['finish_reasons'].get('stop',0)} | {t['finish_reasons'].get('length',0)} | {acc*100:.2f}% | {t.get('wallclock_s',0):.0f} |" if acc is not None else f"| {c} | {t.get('correct')}/30 | {t.get('errors')} | {t['finish_reasons'].get('stop',0)} | {t['finish_reasons'].get('length',0)} | n/a | {t.get('wallclock_s',0):.0f} |")
print("\n## AIME reasoning-budget sweep (c=1, vary max_tokens)\n")
print("| max_tokens | correct/30 | stop | length | MTP accept | wall (s) |")
print("|---|---|---|---|---|---|")
for mt in (8192, 16384, 32768):
    f = d / f"05_aime30_c1_max{mt}.json"
    if not f.exists(): continue
    j = json.load(open(f))
    t, m = j["totals"], j["mtp_delta"]
    acc = m.get("acceptance_rate")
    acc_s = f"{acc*100:.2f}%" if acc is not None else "n/a"
    print(f"| {mt} | {t['correct']}/30 | {t['finish_reasons'].get('stop',0)} | {t['finish_reasons'].get('length',0)} | {acc_s} | {t['wallclock_s']:.0f} |")
print("\n## Knowledge bench\n")
for tag, fn in [("GSM8K-50", "06_gsm8k50.json"), ("HumanEval", "07_humaneval.json"), ("IFEval", "08_ifeval.json")]:
    f = d / fn
    if not f.exists():
        print(f"- **{tag}**: missing")
        continue
    try:
        j = json.load(open(f))
        res = j.get("results", {}) or {}
        for k, v in res.items():
            scores = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
            print(f"- **{tag}** [{k}]: {json.dumps(scores)}")
    except Exception as e:
        print(f"- **{tag}**: parse error {e}")
PYEOF
cat "$OUT_DIR/SUMMARY.md"
log "bench suite COMPLETE. Output dir: $OUT_DIR"
