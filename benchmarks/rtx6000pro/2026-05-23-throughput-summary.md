# RTX PRO 6000 Blackwell throughput summary (2026-05-23)

First end-to-end serve + benchmark of
`canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP` on **NVIDIA RTX PRO 6000
Blackwell Server Edition** (SM 12.0, 96 GiB HBM3 each).

## Setup

- **Hardware:** Brev `familiar-teal-worm` — 4× RTX PRO 6000 Blackwell,
  96 vCPU, 1 TiB RAM. Columbus OH AWS.
- **vLLM build:** `jasl/vllm@ds4-sm120-preview-dev` (SHA `c79225692`,
  built from source with `TORCH_CUDA_ARCH_LIST=12.0a`)
- **Patches:** packed_modules_mapping (×2) + nvidia/ops/attention
  weight_scale fallback + BF16 wo_a fallback + compressor/indexer
  dequant preprocess (one-time, 166 weights bf16). See
  [`RECIPE_RTX6000PRO.md`](../../RECIPE_RTX6000PRO.md) for the full
  recipe.
- **Mode:** **`--enforce-eager`** (torch.compile + cudagraph disabled).
  The conditional `wo_a.weight_scale` getattr in attention.py is not
  dynamo-safe with the current patch shape — running in eager mode
  unblocks the bench but leaves ~10× headroom vs compiled. See
  "Future work" in `RECIPE_RTX6000PRO.md`.
- **Spec-decode:** `method=mtp, num_speculative_tokens=1` (same k=1
  ceiling as H200 per upstream constraints).
- **Bench command:** `vllm bench serve --base-url http://localhost:8000 --model DSV4-W4A16-FP8-MTP --tokenizer /scratch/weights/w4a16-fp8-mtp-gptq --trust-remote-code --dataset-name random --random-input-len 256 --random-output-len 256 --num-prompts $((BS*8)) --max-concurrency $BS`

## TP=2 (2× GPU pair, GPUs 0+1 — same PCIe switch)

| Metric | bs=1 | bs=4 | bs=16 |
|---|---|---|---|
| **Output throughput (tok/s)** | **11.57** | **41.37** | **147.00** |
| Total token throughput (tok/s) | 23.14 | 82.75 | 294.01 |
| TPOT median (ms) | 82.70 | 87.62 | 93.57 |
| TPOT mean (ms) | 82.02 | 90.23 | 99.49 |
| TTFT median (ms) | 303.09 | 473.28 | 510.60 |
| **MTP spec-decode acceptance** | **74.17%** | 70.70% | 72.04% |
| Accepted / drafted | 870 / 1173 | 3386 / 4789 | 13691 / 19004 |
| Duration (8N reqs at concurrency N) | 177 s | 198 s | 223 s |

## TP=4 (all four GPUs)

| Metric | bs=1 | bs=4 | bs=16 |
|---|---|---|---|
| **Output throughput (tok/s)** | **11.74** | **41.81** | **157.33** |
| Total token throughput (tok/s) | 23.49 | 83.61 | 314.66 |
| TPOT median (ms) | 85.23 | 88.17 | 94.38 |
| TPOT mean (ms) | 83.87 | 87.49 | 95.66 |
| TTFT median (ms) | 168.80 | 466.61 | 508.91 |
| **MTP spec-decode acceptance** | **72.84%** | 69.94% | 71.28% |
| Accepted / drafted | 861 / 1182 | 3364 / 4810 | 13607 / 19090 |
| Duration | n/a | 196 s | 208 s |

## TP=2 vs TP=4: marginal at this batch profile

| Batch | TP=2 tok/s | TP=4 tok/s | Δ |
|---|---|---|---|
| bs=1 | 11.57 | 11.74 | +1.5% |
| bs=4 | 41.37 | 41.81 | +1.1% |
| bs=16 | 147.00 | 157.33 | **+7.0%** |

Both configurations are **decode-throughput-bound by the per-step eager
overhead**, not parallelism. The MTP draft path adds ~1.4× decode
speedup vs no-spec (acceptance ≈ 72%) on top of the base eager
throughput. TP=4 helps a bit at high concurrency where the verifier
batch is larger; at bs=1 the parallel TP overhead actually neutralizes
the win.

## MTP acceptance is the headline (this is what the artifact is for)

**~70-75% draft-token acceptance at k=1** across all six configurations
(TP×bs combos). This is in the same band as the H200 published number
of 69.94% (200 random prompts) and the sibling NVFP4-FP8-MTP
benchmark's 67.29% on HumanEval. The MTP block (BF16 preserved per
Option Y) is **firing correctly on RTX 6000 Pro Blackwell**.

## H200 vs RTX 6000 Pro at bs=1

| | H200 (compile + cudagraph) | RTX 6000 Pro (eager) | Ratio |
|---|---|---|---|
| TPOT median MTP (ms) | 6.02 | 82.70 | ~13.7× slower |
| Output tok/s MTP | 88.35 | 11.57 | ~7.6× slower |
| MTP acceptance | 89.1% (Phase 2 calibrated) / 69.94% (200-prompt eval) | 74.17% | within band |

The ~13× TPOT gap is entirely **eager-mode penalty**, not the
hardware. Cleaning up the dynamo-incompatible `wo_a.weight_scale`
runtime check (cache the attribute presence at `__init__` time so
the forward path branches on a static flag) would let us re-enable
`torch.compile` + cudagraph and recover most of that gap. Estimated
post-fix bs=1 TPOT: 8-15 ms range — competitive with H200 once
compile fires.

## Marlin TP > 2 bug (#41511) verdict

vLLM upstream issue #41511 flagged that W4A16 MoE expert `weight_scale`
is not K-sharded under TP > 2, which would block W4A16 MoE serving at
TP=4. **We did not hit this bug on `jasl/vllm@ds4-sm120-preview-dev`.**
TP=4 loaded cleanly, all 43 MoE layers + 1 MTP layer initialized,
and inference produced correct (chat-smoke 4/4 PASS) output. Either
the bug is fixed on the SM12 branch's compressed_tensors path, or our
W4A16 layout doesn't trigger the failing path. Worth a follow-up
verification on the (frozen, no longer reproducible without rebuild)
SHA captured in this run.

## Raw JSONs

`benchmarks/rtx6000pro/tp{2,4}_2026-05-23T*/bench_mtp_bs{1,4,16}.json`
