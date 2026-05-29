#!/bin/bash
OUT="${OUT:-/opt/dlami/nvme/bench-tp2}"
BASE="${BASE:-http://127.0.0.1:8000}"
M="${MODEL_NAME:-DSV4-W4A16-FP8-MTP}"

echo "=== SMOKE 1: math ==="
time curl -s -X POST $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 47 * 89?\"}],\"max_tokens\":120,\"temperature\":0}" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(\"text:\", r[\"choices\"][0][\"message\"][\"content\"][:300]); print(\"tokens:\", r[\"usage\"][\"completion_tokens\"])"

echo
echo "=== SMOKE 2: code ==="
time curl -s -X POST $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function to compute the nth Fibonacci number iteratively.\"}],\"max_tokens\":200,\"temperature\":0}" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(\"text:\", r[\"choices\"][0][\"message\"][\"content\"][:400]); print(\"tokens:\", r[\"usage\"][\"completion_tokens\"])"

echo
echo "=== SMOKE 3: reasoning chain ==="
time curl -s -X POST $BASE/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"If a train leaves city A at 60 mph and another leaves city B at 80 mph toward each other, with cities 280 miles apart, when do they meet? Show your work.\"}],\"max_tokens\":250,\"temperature\":0}" \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print(\"text:\", r[\"choices\"][0][\"message\"][\"content\"][:500]); print(\"tokens:\", r[\"usage\"][\"completion_tokens\"])"

echo
echo "=== SMOKE 4: MTP metrics ==="
curl -s $BASE/metrics 2>&1 | grep -E "spec_decode|num_accepted|num_draft|num_emitted|acceptance" | head -20
