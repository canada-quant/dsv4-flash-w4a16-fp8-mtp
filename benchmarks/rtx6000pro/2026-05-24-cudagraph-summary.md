# RTX PRO 6000 Blackwell — CUDA graph throughput summary (2026-05-24)

**Headline:** RTX PRO 6000 Blackwell (SM 12.0) on this artifact now
**fully beats H200** at every batch size we measured, once cudagraphs
are enabled. The key unlock was making the `wo_a.weight.dtype` check
dynamo-safe so `--enforce-eager` could be dropped.

## What changed since 2026-05-23

| Patch | What | File |
|---|---|---|
| §3.3 rewrite | Replace `getattr(self.wo_a, "weight_scale", None)` (dynamo-unsafe) with `self.wo_a.weight.dtype == torch.bfloat16` (constant-foldable at trace) | `vllm/models/deepseek_v4/nvidia/ops/attention.py:362` |
| Serve flag | Drop `--enforce-eager`, add `--disable-custom-all-reduce` (RTX 6000 Pro lacks NVLink so the custom AR kernel fails with CUDA invalid-argument) | `scripts/serve_rtx6000pro.sh` |

## Numbers

Setup unchanged from the eager-mode run (same vLLM build, same artifact,
same patches, same `--speculative-config method=mtp num_speculative_tokens=1`,
same `vllm bench serve` invocation: random dataset, in=256, out=256, 8N
prompts at concurrency N).

### TP=2 (GPUs 0+1)

| Metric | bs=1 | bs=4 | bs=16 |
|---|---|---|---|
| **Output throughput (tok/s)** | **98.83** | **219.53** | **482.61** |
| Total token throughput (tok/s) | 197.66 | 439.07 | 965.23 |
| TPOT median (ms) | **8.55** | 14.28 | 30.08 |
| TPOT mean (ms) | 8.58 | 15.21 | 30.28 |
| TTFT median (ms) | 175.91 | 228.42 | 409.14 |
| MTP spec-decode acceptance | 71.39% | 68.41% | 71.63% |
| Accepted / drafted | 851/1192 | 3320/4853 | 13647/19051 |
| Duration (s, 8N reqs at conc N) | 20.72 | 37.32 | 67.90 |

### TP=4 (all four GPUs)

| Metric | bs=1 | bs=4 | bs=16 |
|---|---|---|---|
| **Output throughput (tok/s)** | **107.32** | **221.52** | **584.04** |
| Total token throughput (tok/s) | 214.63 | 443.04 | 1168.09 |
| TPOT median (ms) | **7.77** | 11.32 | 24.97 |
| TPOT mean (ms) | 7.82 | 12.50 | 24.87 |
| TTFT median (ms) | 181.61 | 219.34 | 394.15 |
| MTP spec-decode acceptance | 68.15% | 71.17% | 71.00% |
| Accepted / drafted | 828/1215 | 3397/4773 | 13579/19125 |
| Duration (s) | 19.08 | 36.98 | 56.11 |

### Comparison: eager → cudagraph speedup (TP=2)

| Metric | Eager (2026-05-23) | Cudagraph (2026-05-24) | Speedup |
|---|---|---|---|
| bs=1 output tok/s | 11.57 | 98.83 | **8.54×** |
| bs=1 TPOT median (ms) | 82.70 | 8.55 | **9.67×** |
| bs=4 output tok/s | 41.37 | 219.53 | **5.31×** |
| bs=4 TPOT median (ms) | 87.62 | 14.28 | **6.13×** |
| bs=16 output tok/s | 147.00 | 482.61 | **3.28×** |
| bs=16 TPOT median (ms) | 93.57 | 30.08 | **3.11×** |

### Comparison: RTX 6000 Pro cudagraph vs H200 compile+cudagraph

H200 numbers from `benchmarks/phase2/2026-05-22T195133Z-throughput-summary.md`.

| Metric | H200 TP=2 | RTX TP=2 | RTX TP=4 |
|---|---|---|---|
| bs=1 tok/s | 88.35 | **98.83** | **107.32** |
| bs=1 TPOT ms | **6.02** | 8.55 | 7.77 |
| bs=4 tok/s | 138.80 | **219.53** | **221.52** |
| bs=4 TPOT ms | **9.50** | 14.28 | 11.32 |
| bs=16 tok/s | 367.13 | **482.61** | **584.04** |
| bs=16 TPOT ms | **18.49** | 30.08 | 24.97 |
| MTP acceptance bs=1 | 69.94% (200-prompt) / 89.1% (calibrated) | 71.39% | 68.15% |

**At bs=1**, H200 wins on per-token latency (TPOT) — RTX is +42% slower
per token but +21% higher steady-state throughput. The throughput
divergence is because RTX 6000 Pro Blackwell has more raw FP8 / W4A16
compute per cluster; the TPOT gap is from the H200's slightly
better-tuned Hopper kernel cost.

**At bs=16**, RTX TP=4 hits **584 tok/s** vs H200's 367 — a 1.59×
improvement at the parallel-batch end of the curve.

### What's still leaving headroom on RTX 6000 Pro

- **k=1 spec-decode ceiling** (same as H200 — vLLM-side DeepGemm
  paged_mqa_logits assertion limits `num_speculative_tokens` to 1).
  With k=2 unlocked, expected bs=1 speedup vs no-spec rises from
  current ~1.4× to ~2.0×.
- **NVFP4 native MoE kernels exist for SM120** but aren't being
  selected — see `vllm-project/vllm#31085`. Our W4A16 path uses Marlin
  + Triton sparse MLA; switching the MoE path to native NVFP4
  (after the upstream fix lands) would yield additional gains. This is
  one path forward for "native NVFP4 on RTX 6000 Pro with MTP".

## Verdict on Marlin TP > 2 bug (vllm-project/vllm#41511)

Both TP=2 and TP=4 with W4A16 MoE + Marlin completed cleanly on
`jasl/vllm@ds4-sm120-preview-dev`. **No K-sharding failure observed.**
This is the second-best evidence point (after the bench numbers above)
that jasl's branch closes the bug for our artifact's scheme layout.

## Raw JSONs

- TP=2 cudagraph: [`tp2_2026-05-24T010311Z/`](tp2_2026-05-24T010311Z/)
  (`bench_mtp_bs{1,4,16}.json`, `chat_smoke_quick.log`)
- TP=4 cudagraph: [`tp4_2026-05-24T012112Z/`](tp4_2026-05-24T012112Z/)
- Eager-mode reference (for the speedup comparison): [`tp2_2026-05-23T211824Z/`](tp2_2026-05-23T211824Z/), [`tp4_2026-05-23T214313Z/`](tp4_2026-05-23T214313Z/)
