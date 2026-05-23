#!/usr/bin/env bash
# Run the full benchmark suite against a running vllm serve instance.
# Drops outputs under benchmarks/rtx6000pro/<TS>/ — one dir per run.
#
# Usage:
#   bash scripts/bench_rtx6000pro_suite.sh <base_url> <tp> [skip_long]
#   skip_long=1 → skip MMLU + MMLU-Pro + AIME (the multi-hour items)
#
# Assumes:
#   - vllm serve is up at $1 (model name DSV4-W4A16-FP8-MTP)
#   - venv-serve activated OR vllm available on PATH
#   - lm-eval-harness installed (`pip install lm-eval[api]`)

set -uo pipefail

# Activate venv-serve so vllm + lm_eval are on PATH
if [[ -f "$HOME/venv-serve/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/venv-serve/bin/activate"
fi

BASE_URL="${1:?usage: $0 <base_url> <tp> [skip_long]}"
TP="${2:?usage: $0 <base_url> <tp> [skip_long]}"
SKIP_LONG="${3:-0}"
TS=$(date -u +%Y-%m-%dT%H%M%SZ)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/benchmarks/rtx6000pro/tp${TP}_${TS}"
mkdir -p "$OUT_DIR"
echo "[bench] writing to $OUT_DIR"

# --- chat-smoke quick (~1 min) ---
echo "[bench] chat-smoke quick"
bash "$REPO_ROOT/scripts/chat_smoke.sh" "$BASE_URL" 2>&1 | tee "$OUT_DIR/chat_smoke_quick.log"

# --- MTP acceptance @ 200 (~5-10 min) ---
echo "[bench] MTP acceptance @ 200"
python - "$BASE_URL" "$OUT_DIR/acceptance_200.json" 2>&1 | tee "$OUT_DIR/acceptance_200.log" <<'PYEOF'
import sys, json, time
from urllib.request import urlopen, Request
import re
base, outpath = sys.argv[1], sys.argv[2]
PROMPTS = []
import random
random.seed(42)
words = "the quick brown fox jumps over the lazy dog while pondering deep questions about life universe everything code math science art".split()
for _ in range(200):
    PROMPTS.append(" ".join(random.choices(words, k=64)))

def get_metrics():
    try:
        text = urlopen(base + "/metrics", timeout=10).read().decode()
    except Exception as e:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        if "spec_decode" in line or "draft_acceptance" in line or "num_accepted" in line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[-1])
                except Exception:
                    pass
    return out

m0 = get_metrics()
print("[mtp] baseline metrics keys:", sorted(m0.keys())[:5])

t0 = time.time()
for i, p in enumerate(PROMPTS):
    payload = {
        "model": "DSV4-W4A16-FP8-MTP",
        "prompt": p,
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }
    req = Request(base + "/v1/completions", method="POST",
                  headers={"Content-Type": "application/json"},
                  data=json.dumps(payload).encode())
    try:
        urlopen(req, timeout=300).read()
    except Exception as e:
        print(f"  prompt {i}: FAIL {e}")
    if (i+1) % 20 == 0:
        print(f"  {i+1}/200 done  ({time.time()-t0:.0f}s)")

dt = time.time() - t0
m1 = get_metrics()
print(f"[mtp] total: {dt:.0f}s")

result = {
    "n_prompts": 200,
    "wall_s": dt,
    "before": m0,
    "after": m1,
}

# Try to compute acceptance delta
acc_delta = {}
for k in m1:
    if k in m0:
        delta = m1[k] - m0[k]
        if delta != 0:
            acc_delta[k] = delta
result["delta"] = acc_delta

# Approximate acceptance: spec_decode_num_accepted_tokens / spec_decode_num_draft_tokens
for accepted_k in m1:
    if "num_accepted" in accepted_k:
        for draft_k in m1:
            if "num_draft" in draft_k or "num_proposed" in draft_k:
                d_acc = m1[accepted_k] - m0.get(accepted_k, 0)
                d_drf = m1[draft_k] - m0.get(draft_k, 0)
                if d_drf > 0:
                    result["acceptance_rate"] = d_acc / d_drf
                    result["num_accepted"] = d_acc
                    result["num_drafted"] = d_drf
                    print(f"[mtp] acceptance: {d_acc:.0f} / {d_drf:.0f} = {d_acc/d_drf*100:.2f}%")
                    break
        if "acceptance_rate" in result:
            break

with open(outpath, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"[mtp] wrote {outpath}")
PYEOF

# --- Throughput TPOT (vllm bench serve, MTP vs no-spec) ---
# Note: with MTP spec already enabled in serve config, "no-spec" requires
# a separate server. For this RTX 6000 Pro run we record MTP-on numbers
# only (one serve config). Predecessor H200 BENCHMARKS.md has the
# no-spec baseline.
echo "[bench] throughput TPOT (MTP-on, bs=1/4/16) via vllm bench serve"
for BS in 1 4 16; do
    NREQ=$((BS * 8))
    vllm bench serve \
        --base-url "$BASE_URL" \
        --model DSV4-W4A16-FP8-MTP \
        --tokenizer /scratch/weights/w4a16-fp8-mtp-gptq \
        --trust-remote-code \
        --dataset-name random --random-input-len 256 --random-output-len 256 \
        --num-prompts $NREQ --max-concurrency $BS \
        --save-result --result-dir "$OUT_DIR" \
        --result-filename "bench_mtp_bs${BS}.json" 2>&1 | tee "$OUT_DIR/bench_mtp_bs${BS}.log" || \
        echo "[warn] vllm bench bs=$BS failed"
done

# --- GSM8K 8-shot (~30 min) ---
echo "[bench] GSM8K 8-shot"
lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args "model=DSV4-W4A16-FP8-MTP,base_url=${BASE_URL}/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
    --num_fewshot 8 \
    --output_path "$OUT_DIR/gsm8k.json" \
    --log_samples 2>&1 | tee "$OUT_DIR/gsm8k.log" || echo "[warn] GSM8K failed"

# --- HumanEval (~5 min) ---
echo "[bench] HumanEval pass@1"
lm_eval --model local-completions \
    --tasks humaneval_instruct \
    --model_args "model=DSV4-W4A16-FP8-MTP,base_url=${BASE_URL}/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
    --confirm_run_unsafe_code \
    --output_path "$OUT_DIR/humaneval.json" \
    --log_samples 2>&1 | tee "$OUT_DIR/humaneval.log" || echo "[warn] HumanEval failed"

if [[ "$SKIP_LONG" != "1" ]]; then
    # --- MMLU 5-shot (~60 min) ---
    echo "[bench] MMLU 5-shot"
    lm_eval --model local-completions \
        --tasks mmlu \
        --model_args "model=DSV4-W4A16-FP8-MTP,base_url=${BASE_URL}/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
        --num_fewshot 5 \
        --output_path "$OUT_DIR/mmlu.json" 2>&1 | tee "$OUT_DIR/mmlu.log" || echo "[warn] MMLU failed"

    # --- AIME 24 (~10 min) ---
    echo "[bench] AIME 24"
    lm_eval --model local-completions \
        --tasks aime24 \
        --model_args "model=DSV4-W4A16-FP8-MTP,base_url=${BASE_URL}/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
        --output_path "$OUT_DIR/aime24.json" 2>&1 | tee "$OUT_DIR/aime24.log" || echo "[warn] AIME failed"
fi

echo "[bench] DONE. Output in $OUT_DIR"
ls -la "$OUT_DIR"
