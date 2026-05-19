#!/usr/bin/env bash
# Phase 7 — 4× TP=2 serve, pinned to disjoint GPU pairs on a p6-b300.48xlarge.
# Frontended by a caller-supplied LB (nginx/Caddy round-robin) for throughput.
#
# Usage:
#   bash scripts/serve_b300_full_node.sh /scratch/weights/w4a16-fp8-mtp
#
# Each instance listens on a different port (8000-8003). Logs go to
# /tmp/vllm-serve-<port>.log.
set -euo pipefail

MODEL="${1:?usage: $0 <model_path>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$MODEL" ]]; then
    echo "FATAL: model path not found: $MODEL" >&2
    exit 1
fi

declare -a PAIRS=("0,1:8000" "2,3:8001" "4,5:8002" "6,7:8003")

PIDS=()
for spec in "${PAIRS[@]}"; do
    gpus="${spec%%:*}"
    port="${spec##*:}"
    log="/tmp/vllm-serve-${port}.log"
    echo "==> launching CUDA_VISIBLE_DEVICES=$gpus on :$port  (log: $log)"
    CUDA_VISIBLE_DEVICES="$gpus" nohup bash "$HERE/serve_b300_tp2.sh" "$MODEL" "$port" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo
echo "launched 4 instances (PIDs: ${PIDS[*]})"
echo "tail logs:  tail -f /tmp/vllm-serve-8000.log /tmp/vllm-serve-8001.log ..."
echo
echo "wait for /health on each before sending traffic:"
for spec in "${PAIRS[@]}"; do
    port="${spec##*:}"
    echo "  while ! curl -fsS http://localhost:$port/health; do sleep 5; done"
done

wait
