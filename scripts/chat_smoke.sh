#!/usr/bin/env bash
# Phase 5/6 — minimal chat smoke against a running vllm serve instance.
# Tests:
#   - /health 200
#   - 4 chat completions with simple prompts → finite text response
#
# Usage:
#   bash scripts/chat_smoke.sh http://localhost:8000
set -euo pipefail

BASE="${1:?usage: $0 <base_url>  e.g. http://localhost:8000}"

echo "=== /health ==="
curl -fsS "$BASE/health" || { echo "FAIL: /health"; exit 1; }
echo "OK"

echo
echo "=== 4 chat completions ==="
pass=0
for prompt in \
    "Hello! What is 1+1?" \
    "Write a haiku about Blackwell GPUs." \
    "What is the capital of France?" \
    "Briefly explain Mixture of Experts."
do
    echo "--- $prompt ---"
    response=$(curl -fsS "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"DSV4-W4A16-FP8-MTP\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"max_tokens\": 100,
            \"temperature\": 1.0,
            \"top_p\": 1.0
        }") || { echo "FAIL: completion request"; continue; }

    content=$(echo "$response" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'][:200])" 2>/dev/null || echo "(parse failed)")
    if [[ -n "$content" && "$content" != "(parse failed)" ]]; then
        echo "  $content"
        pass=$((pass + 1))
    else
        echo "  EMPTY"
    fi
done

echo
echo "chat-smoke: $pass/4 PASS"
[ "$pass" -eq 4 ]
