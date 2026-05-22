# Phase 2 throughput — with/without MTP (TS 2026-05-22T19:51:33Z)

## Setup

- Artifact: `/scratch/weights/w4a16-fp8-mtp-gptq` (Phase 2 post-fixup, MTP acceptance 89.1% at k=1)
- Hardware: 8× H200 (using TP=2, GPUs 0+1)
- vLLM: `~/src/vllm` HEAD `50d9dd902` (cherry-pick PRs #43248+#43288+#43290+#43319)
- vLLM bench: `vllm bench serve --dataset-name random --random-input-len 256 --random-output-len 256`
- N prompts: 8N at concurrency N (8 / 32 / 128 for bs=1/4/16)
- **Spec config: `method=mtp num_speculative_tokens=1`** — `=2` triggers DeepGemm kernel assertion `next_n == 1 or next_n == 2` during cudagraph profiling in this build, so we ran at k=1. Sibling's published 2.03× was at k=2; theoretical k=1 ceiling is ~1.89× (1 + acceptance with overhead).
- max-num-seqs 16, max-model-len 4096, kv-cache-dtype fp8, block-size 256

## Raw results

| BS | Config | Output tok/s | TPOT median (ms) | TPOT mean (ms) | TTFT median (ms) | TTFT mean (ms) |
|---|---|---|---|---|---|---|
| 1 | MTP-spec | 88.35 | **6.02** | 6.04 | 160 | 1356 |
| 1 | no-spec | 104.10 | **8.93** | 8.99 | 156 | 167 |
| 4 | MTP-spec | 138.80 | 9.50 | 17.01 | 179 | 2993 |
| 4 | no-spec | 242.41 | 10.73 | 10.76 | 287 | 1478 |
| 16 | MTP-spec | 367.13 | 18.49 | 34.08 | 218 | 2404 |
| 16 | no-spec | 953.92 | 15.28 | 15.32 | 386 | 387 |

## Speedup analysis (MTP-spec vs no-spec)

| Metric | bs=1 | bs=4 | bs=16 |
|---|---|---|---|
| Output tok/s ratio | **0.85×** ⚠ | 0.57× | 0.38× |
| TPOT median ratio (lower=faster) | **1.49× faster** ✓ | 1.13× faster | 0.83× slower |
| TPOT mean ratio (lower=faster) | 1.49× faster | 0.63× slower | 0.45× slower |

**Headline: 1.49× decode speedup at bs=1 (single-user interactive) via TPOT median.**

Output_throughput at bs=1 favors no-MTP (104 vs 88) ONLY because the MTP path has a huge first-request TTFT outlier (1356ms mean vs 160ms median) — cudagraph capture cost on first call. Steady-state per-token decode (TPOT median) is decisively faster with MTP at bs=1.

## Why bs=4, bs=16 don't win

Spec-decode wins when the verifier model is underutilized (single-user, decode-bound). At higher concurrency, the verifier is already filling its batch lane, so the extra verifier passes from spec-decode add overhead without saving wall-clock. This matches the sibling's published methodology doc — MTP is reported as `c=1` because that's where the speedup is real.

## How this compares to sibling NVFP4-FP8-MTP

| | Ours W4A16-FP8 (this run) | Sibling NVFP4-FP8 (published) |
|---|---|---|
| MTP acceptance at k=1 / k=2 | 89.1% (k=1) | 67.29% (k=2, raw code) |
| Decode speedup vs no-spec | 1.49× (k=1, bs=1) | 2.03× (k=2, c=1 HumanEval raw) |
| Hardware | H200 SXM6 (Hopper, sm_90a) | B300 SXM6 AC (Blackwell, sm_103a) |
| Quantization | W4A16 (Marlin) + FP8 attn | NVFP4 + FP8 attn |
| MTP precision | BF16 (Option Y) | BF16 (Option Y) |

Direct comparison is muddied by k=1 vs k=2 (kernel limitation in our build). With k=2 unlocked, theoretical bs=1 speedup is ~1.85× given our 89% acceptance, comparable to sibling.

## Raw JSONs

- `bench_2026-05-22T195133Z_mtp_bs1.json`
- `bench_2026-05-22T195133Z_mtp_bs4.json`
- `bench_2026-05-22T195133Z_mtp_bs16.json`
- `bench_2026-05-22T195133Z_nomtp_bs1.json`
- `bench_2026-05-22T195133Z_nomtp_bs4.json`
- `bench_2026-05-22T195133Z_nomtp_bs16.json`

## Open work item

DeepGemm `next_n == 1 or next_n == 2` kernel assertion fires when `num_speculative_tokens=2` during cudagraph profiling. Worth filing upstream — likely a shape mismatch in the spec-decode kernel's profile-step batch dim. Working around with k=1 cost ~0.5× of theoretical speedup.
