# Card D deeper kernel-dispatch blocker discovered — 2026-05-26

## Summary

Yesterday (2026-05-25) we documented that Card D's safetensors uses `.weight_scale` rather than the canonical `.weight_scale_inv` naming, and identified two unblock paths:

1. Rename keys in safetensors `.weight_scale` → `.weight_scale_inv` (artifact-side)
2. Comprehensive defensive vLLM patch upstream

Today we tried both. **Neither alone unblocks Card D.** The artifact has a *third* issue: the FP8 block-quant scheme is being routed to a per-channel kernel at forward time, failing the shape check `b_scales dim 1 = 32 is not size_n = 1536`.

## What we attempted

### Attempt 1: Rename `.weight_scale` → `.weight_scale_inv` in safetensors

Wrote a header-only rename script (`scripts/rename_weight_scale_to_inv.py`) that rewrites only the JSON header of each shard (no tensor data rewrite). 21 minutes, 215 keys renamed across 4 shards. Result: load **failed earlier** with `KeyError: 'layers.0.attn.fused_wqa_wkv.weight_scale_inv'` — the model's fused layer (`MergedColumnParallelLinear`) registers `weight_scale` as the params_dict key, not `weight_scale_inv`, so the renamed safetensors keys couldn't be matched.

**Conclusion**: the rename direction was wrong. The model expects `.weight_scale` in safetensors (matching its params_dict naming), and the canonical `weight_scale_inv` naming convention applies only at one specific call site in `scaled_mm/marlin.py:73`.

### Attempt 2: Defensive `getattr` patch on `scaled_mm/marlin.py`

Patched `vllm/model_executor/kernels/linear/scaled_mm/marlin.py:71` to accept either `weight_scale_inv` or `weight_scale` and to always register the result under `weight_scale_inv` (so downstream code finds it).

Result: model **loaded past `process_weights_after_loading`** for the first time. But forward-pass kernel call failed:

```
RuntimeError: b_scales dim 1 = 32 is not size_n = 1536
```

The kernel being called expects a **per-channel scale** of shape `(1536,)` (one per output channel after TP=4 partitioning of 6144), but receives the **block-shaped scale** `(4, 32)` from FP8 block 128×128 quantization. Some forward-path code is routing the FP8 block layer to the wrong kernel (`marlin_gemm` per-channel instead of `marlin_block_gemm` or equivalent).

## Hypothesis on the deeper bug

The FP8 block-quant scheme registers `weight_scale` as the layer's param (verified at `process_weights_after_loading`). Some downstream kernel-dispatch code distinguishes block vs per-channel by looking at `weight_scale_inv` (present) vs `weight_scale` (per-channel). Because the artifact's scale is named `weight_scale`, the dispatch falls into the per-channel branch, which then fails the shape check.

`marlin_utils_fp8.py` already has `hasattr(layer, "weight_scale_inv")` defensive logic at lines 138-141 and 190-193, but the kernel-dispatch site that's failing is elsewhere. We didn't find it in 30 minutes of tracing.

## Path forward

This is **not** the kind of bug a downstream artifact can patch around. It needs an upstream fix where:

- Either the FP8 block-quant scheme always registers its scale param as `weight_scale_inv` regardless of the source artifact's naming, OR
- Or all downstream dispatch code is hardened to handle both names via `hasattr` checks.

Filed upstream as part of [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564) — extended that issue today with the additional kernel-dispatch finding.

## What this means for Card D's published H200 / B300 benchmarks

The H200 + B300 benchmark numbers are still valid for the older `jasl/vllm@ds4-sm120-experimental@abad5dc71` build (the build they were measured on). The artifact weights are mathematically correct. Only modern (post-c79225692) upstream vLLM builds reject the artifact, due to evolving kernel-dispatch expectations. The model itself works on the version of vLLM at which it was calibrated and measured.

## What this means for the Marlin c_tmp + workspace 4× patches

The patches (in [`vllm-patches/0002`](../../vllm-patches/0002_marlin_moe_workspace_4x.patch) and [`vllm-patches/0003`](../../vllm-patches/0003_marlin_moe_c_tmp_36889.patch)) remain **built and ready** in `_moe_C.abi3.so`, but cannot be functionally verified against this artifact until the kernel-dispatch issue is fixed upstream. They were verified to apply cleanly and compile correctly, and they don't introduce any new layer-registration logic — they're purely Marlin MoE kernel changes (workspace sizing + c_tmp reduce buffer clamp).

## Recommended path for next session

1. Wait for upstream movement on [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564) (extended today).
2. If urgent: build vLLM against `jasl/vllm@abad5dc71` (the old build that worked for this artifact) and verify Marlin patches there. This is a real time investment (~30-45 min build) but does have a known-working baseline.
3. If publishing Card D fresh: re-calibrate against `llmcompressor>=0.x.x` (whichever version emits canonical `weight_scale_inv` naming directly) — but per yesterday's analysis, this isn't strictly necessary for the model to be mathematically correct, just for shippability on current vLLM.
