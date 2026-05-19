#!/usr/bin/env python3
"""Phase 6 — minimal decode throughput + MTP acceptance benchmark.

Connects to a running ``vllm serve`` instance and measures:
  * tokens/sec on a long-context decode workload (default ~64K prompt, 2K
    decode tokens; lower than the 524K target in PLAN.md to fit the smoke
    pass; rerun with --prompt-tokens 524288 for the headline number)
  * MTP draft acceptance rate (reported by vLLM when --speculative-config
    method=mtp is set — pulled from the /metrics endpoint)

Not a full eval harness (GSM8K / HumanEval / NIAH / toolcall15 live in the
external jasl/vllm-ds4-sm120-harness). This script is a quick sanity check
that the served model is healthy and the MTP path is firing.

Usage::

    python scripts/bench_decode.py http://localhost:8000 \\
        --prompt-tokens 65536 --decode-tokens 2048 --requests 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.request import urlopen, Request


def _post(url: str, payload: dict) -> dict:
    req = Request(url, method="POST", headers={"Content-Type": "application/json"},
                  data=json.dumps(payload).encode("utf-8"))
    with urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())


def _get(url: str) -> str:
    with urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def measure_one(base: str, prompt_tokens: int, decode_tokens: int) -> dict:
    """Issue a single completion request and time it."""
    # Build a long prompt by repeating a phrase. The /v1/completions endpoint
    # avoids chat-template overhead — important for clean throughput numbers.
    placeholder = "The quick brown fox jumps over the lazy dog. "
    prompt = (placeholder * ((prompt_tokens // 10) + 1))[: prompt_tokens * 5]

    payload = {
        "model": "DSV4-W4A16-FP8-MTP",
        "prompt": prompt,
        "max_tokens": decode_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False,
    }
    t0 = time.time()
    out = _post(base + "/v1/completions", payload)
    dt = time.time() - t0
    completion_tokens = out.get("usage", {}).get("completion_tokens", decode_tokens)
    return {
        "wall_s": dt,
        "completion_tokens": completion_tokens,
        "tok_per_s": completion_tokens / max(dt, 1e-6),
    }


def fetch_mtp_acceptance(base: str) -> dict | None:
    """Parse vLLM's /metrics endpoint for spec_decode_efficiency / acceptance_rate."""
    try:
        text = _get(base + "/metrics")
    except Exception as exc:
        print(f"[mtp] /metrics fetch failed: {exc}", file=sys.stderr)
        return None
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        if "spec_decode" in line or "speculative" in line or "draft_acceptance" in line:
            parts = line.split()
            if len(parts) >= 2:
                metrics[parts[0]] = parts[-1]
    return metrics or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url", help="e.g. http://localhost:8000")
    ap.add_argument("--prompt-tokens", type=int, default=65536)
    ap.add_argument("--decode-tokens", type=int, default=2048)
    ap.add_argument("--requests", type=int, default=4)
    args = ap.parse_args()

    print(f"=== bench_decode  prompt={args.prompt_tokens}  decode={args.decode_tokens}  "
          f"requests={args.requests} ===")

    # warm-up
    print("[warmup] 1 short request")
    measure_one(args.base_url, prompt_tokens=512, decode_tokens=64)

    results = []
    for i in range(args.requests):
        r = measure_one(args.base_url, args.prompt_tokens, args.decode_tokens)
        results.append(r)
        print(f"  req {i+1}/{args.requests}: {r['completion_tokens']:5d} tok in "
              f"{r['wall_s']:6.2f}s  ->  {r['tok_per_s']:6.2f} tok/s")

    avg = sum(r["tok_per_s"] for r in results) / len(results)
    p50 = sorted(r["tok_per_s"] for r in results)[len(results) // 2]
    print()
    print(f"avg: {avg:.2f} tok/s   median: {p50:.2f} tok/s")
    print(f"Acti reference: 85.52 tok/s @ 524K  (predecessor target)")

    print()
    metrics = fetch_mtp_acceptance(args.base_url)
    if metrics:
        print("=== MTP / speculative-decoding metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("(no MTP metrics scraped; check /metrics endpoint or check spec config)")


if __name__ == "__main__":
    main()
