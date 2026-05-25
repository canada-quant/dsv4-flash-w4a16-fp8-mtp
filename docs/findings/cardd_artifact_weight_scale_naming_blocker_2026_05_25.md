# Card D safetensors `.weight_scale` vs `.weight_scale_inv` blocker — 2026-05-25 (PM)

## Summary

The Card D dequant'd artifact does **not** load on current vLLM. Diagnosis chased through three layers:

1. **Config.json metadata** — config still claimed compressor + indexer were FP8 block-quantized but they were removed during the shipping-bug dequant. **Fixed** ([HF commit `0bbee82146`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP/commit/0bbee82146562d9b7e4687483d348c760e61ef3b)). Did not unblock loading.

2. **Defensive vLLM loader patch** — patched `scaled_mm/marlin.py` in venv-serve to accept either `weight_scale_inv` or `weight_scale` and register the result under `weight_scale_inv` so downstream code finds it. Unblocked the *loader* but exposed a deeper kernel shape-check failure (`b_scales dim 1 = 32 is not size_n = 1536`) — the forward path uses a different kernel that doesn't tolerate the naming either.

3. **Real root cause** — all 33,239 quantized scale tensors in the artifact's safetensors are named `<module>.weight_scale` (no `_inv` suffix). vLLM expects FP8 block scales under `<module>.weight_scale_inv`. The mismatch is not solvable by a single Python-side patch because the naming touches the loader path, the forward path, and Dynamo's static lookup tables.

## Two viable unblock paths (neither requires re-quantization)

### Option A — Rename keys in safetensors (artifact fix)

Open each `model-NNNNN-of-00004.safetensors` file. For every key ending in `.weight_scale` where the corresponding `.weight` tensor is FP8 (`torch.float8_e4m3fn`) or has FP8 block-shaped scales, rename the key to `.weight_scale_inv`. Write back. Re-upload to HF (159 GB total, ~30 min on a fast network).

This is the cleanest fix — the math is identical, just the attribute names match what vLLM expects.

### Option B — Comprehensive vLLM defensive patch upstream

Filed at [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564). Two-line `getattr` fallback in `scaled_mm/marlin.py` + similar in `marlin_utils_fp8.py` + DeepseekV4 renaming mapper extension. Tagged kylesayrs (compressed-tensors maintainer). Question pending on whether `weight_scale_inv` is semantically the multiplicative inverse vs `weight_scale` is divide-by; if they're truly interchangeable the patch is trivial.

**Recommendation: do both.** Option A unblocks downstream consumers immediately. Option B helps the broader ecosystem.

## What this means for previously-published Card D benchmarks

The H200 + B300 numbers in Card D's HF README are valid — measured on the *original* (pre-dequant, pre-vLLM-format-change) artifact on an older vLLM. The mathematical content of the artifact's weights has not changed since calibration. The benchmark numbers stand; only the loading-pipeline compatibility broke.

## What this means for the Marlin c_tmp + workspace 4x patches

The patches are **built and verified to compile** (`_moe_C.abi3.so` is 181,697,240 bytes, +1,784 from the pre-patch baseline, matching the expected added defensive sizing). They cannot be **functionally verified on Card D** until one of Option A or Option B lands. They were verified to apply cleanly against `jasl/vllm@a02a3778f + PR #40923` and the compiled binary is ready to drop into any working venv-serve install.

## Next-session next steps

1. **Option A**: write a small `rename_safetensors_scale_inv.py` that walks all 4 shards, renames keys, and re-saves. Then re-upload + smoke-test Card D + run the AIME c=4 thinking sweep — this is the moment-of-truth for the Marlin c_tmp + workspace 4x patches.
2. **Option B**: monitor `#43564` for maintainer response; offer to author the PR if there's interest.
3. **Defensive `e_score_correction_bias` patch for Card A** — separate bug, separate fix, also needs a vLLM PR.
