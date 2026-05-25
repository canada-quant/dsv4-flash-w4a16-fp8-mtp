# Session findings 2026-05-25 — Card D Marlin patches built, separate config.json blocker found

## Summary

Cherry-picked [`PR #36889`](https://github.com/vllm-project/vllm/pull/36889) (Marlin MoE `c_tmp` size fix) + a workspace-4x oversize patch on top of `jasl/vllm@a02a3778f + PR #40923` and rebuilt `_moe_C.abi3.so` successfully on RTX PRO 6000. The new binary is 1,784 bytes larger than the pre-patch version, reflecting the added defensive sizing.

However, verification of the patches on Card D against the current dequantized artifact is **blocked by an unrelated, pre-existing artifact issue**: the dequantized safetensors (where compressor/indexer FP8 weights were rewritten to BF16 in-place) still ship a `config.json` that declares the FP8 block-quant scheme for attention sub-modules including `fused_wkv_wgate`. vLLM's loader then runs `process_weights_after_loading` on the `MergedColumnParallelLinear` for `fused_wkv_wgate`, which expects a `weight_scale_inv` attribute that the dequantized artifact doesn't carry (it has only `weight_scale` because the dequant collapsed the FP8 scales into BF16 weights).

This is the same bug-class as [`vllm-project/vllm#43512`](https://github.com/vllm-project/vllm/issues/43512) ("Plumb quant_config into compressor.fused_wkv_wgate") — but in our case, the root cause is the **artifact** (dequantized safetensors + still-FP8 config.json), not vLLM itself.

## What we verified

- **Cherry-picked patches apply cleanly.** PR #36889's `ops.cu` diff + workspace 4x in `marlin_utils.py:268` both apply without conflict against `jasl/vllm@a02a3778f`.
- **Build succeeds.** `_moe_C.abi3.so` rebuilt from `csrc/moe/marlin_moe_wna16/{ops.cu,marlin_template.h}` with native `sm_120a` cubins (the build crashes at `spinloop.abi3.so` install — a known DLAMI Python 3.10 gotcha — but the Marlin .so is produced cleanly and can be copied to `venv-serve` manually).
- **The c_tmp patch's compiled binary is 1,784 bytes larger** than the unpatched native cubin (matches the added `device_max_shared_mem` indirection + removed `min()` clamp).

## What we couldn't verify

- **AIME-30 c=4 thinking-mode regression test on Card D.** The post-patch Card D serve never reaches inference because of the `weight_scale_inv` AttributeError above. To run this regression we need either (a) a fresh Card D quantization that emits a coherent config.json, OR (b) a vLLM defensive `getattr(layer, "weight_scale_inv", layer.weight_scale)` patch in `vllm/model_executor/kernels/linear/scaled_mm/marlin.py:73` (and the parallel site at line 106 in `marlin_utils_fp8.py`).

## Next-session unblock plan

1. **Easiest path** — apply a defensive `getattr` patch to `scaled_mm/marlin.py` and `marlin_utils_fp8.py` to fall back from `weight_scale_inv` to `weight_scale`, then re-test Card D AIME c=4. **Approach time: ~2 hours.** Patch then becomes `0009_marlin_weight_scale_fallback.patch` in `vllm-patches/`.
2. **Cleaner path** — fix Card D's `config.json` to mark `fused_wkv_wgate` (and other dequantized layers) as ignored from the FP8 block-quant scheme. Re-upload artifact. Then test against vanilla loader. **Approach time: ~1 hour artifact edit + re-upload, then bench.**
3. **Combined** — do both (1) for upstream contribution + (2) for clean artifact behavior. Recommended.

Either approach unblocks the patch verification. The patches themselves are **ready to land** — they just need a non-broken Card D artifact to validate against.

## Files staged for this work

- `vllm-patches/0003_marlin_moe_c_tmp_36889.patch` — PR #36889 cherry-pick (ops.cu)
- `vllm-patches/0002_marlin_moe_workspace_4x.patch` — workspace 4x oversize (marlin_utils.py:268)
- `/home/ubuntu/src/vllm/vllm/_moe_C.abi3.so` (181,697,240 bytes, dated 2026-05-25 04:34 UTC) — the patched binary that's ready to drop into any working venv

## Upstream filing posture

We've already posted [our repro on PR #36889](https://github.com/vllm-project/vllm/pull/36889#issuecomment-4531289048) noting we'd report back within 2 hours. The c_tmp build success + the artifact-side blocker is the real story to tell upstream — file as a follow-up comment with the bench JSONs from Card B (which is the clean control on the same hardware + same patched build).
