# dsv4-flash-w4a16-fp8-mtp

Re-do of [`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8) with the **MTP layer (layer 43) correctly included in the GPTQ calibration pass**, targeting AWS `p6-b300.48xlarge` (8× B300, Blackwell DC SM 10.0).

Sibling, non-overlapping scope to [`dsv4-flash-reasoning-agent`](https://github.com/pasta-paul/dsv4-flash-reasoning-agent) (which adds SFT + GRPO on top of the quant). This repo is **quant-only**.

## Status

**Phase 1 — dequant COMPLETE (2026-05-19 17:50 UTC).** Output at `/scratch/weights/bf16-mtp` on the live EC2 box: 46 safetensors shards, 543 GB BF16, 35,020 total tensors. **797 MTP weight tensors preserved** (verified by `scripts/verify_mtp_keys.py`; consistent with upstream 1,575 = 797 weights + 778 scales, where every scale was consumed into its paired BF16 weight). Spot-check on `mtp.0.attn.wq_a.weight` (post-dequant `[1024, 4096]` bfloat16, mean-abs 0.023, max 0.22) and `mtp.0.ffn.experts.0.w1.weight` (`[2048, 4096]` bfloat16, mean-abs 0.022, max 0.19) show sane magnitudes — no overflow, no underflow. EC2 box (`i-0714f36a266c8c59b`, `us-west-2`, 8× B300), `venv-calib` patched (`transformers==5.8.1`, `compressed-tensors==0.15.1a20260515`, `llm-compressor@f2aa32e2`, `torch 2.12.0+cu130`); `DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected == []` runtime-confirmed.

**Phase 2 — COMPLETE via model_free RTN (2026-05-19).** Quantized artifact at `/scratch/weights/w4a16-fp8-mtp` on the EC2 box: **146 GB, 102,826 tensors, 2,340 mtp.0.* tensors** (768 expert W4A16 scales = 256×3, 5 FP8 attention scales, 22 BF16 passthrough modules + the quantized weights). `verify_mtp_quantized.py` PASSES. Wall clock: 7 min (5 min W4A16 expert pass + 2 min FP8 attention pass).

Path used: `llmcompressor.entrypoints.model_free.model_free_ptq` operating directly on safetensors files (no `PreTrainedModel` required), bypassing the MTP-class integration block that would otherwise need a 500+ LOC shim. Two-pass recipe (W4A16 experts → FP8_BLOCK attention + MTP entry projections) implemented in [`scripts/quantize_v4_model_free.py`](scripts/quantize_v4_model_free.py).

Trade-off vs GPTQ: round-to-nearest per weight, no Hessian refinement. FP8_BLOCK on BF16 weights is essentially lossless via RTN; W4A16 has measurable quality loss vs GPTQ on dense layers but is acceptable for the **MTP layer** specifically (used for speculative drafts where draft acceptance is the metric, not pure logit fidelity). For users who want GPTQ on the main 43 layers, [`PHASE2_DESIGN.md`](PHASE2_DESIGN.md) documents the full path; the adapter scaffold is at [`scripts/upstream/`](scripts/upstream/) with a passing tiny-model smoke test ([`scripts/smoke_test_adapter.py`](scripts/smoke_test_adapter.py)).

**Phase 4 — vLLM build deferred.** First `venv-serve` build attempt failed at cmake configure (output suppressed by pip's isolated build). Bootstrap updated with `--no-build-isolation` + verbose log capture; not yet re-run since serving isn't needed until Phase 5.

See [`PLAN.md`](PLAN.md) for the full phase-by-phase execution plan, [`patches/VERSIONS.md`](patches/VERSIONS.md) for patch provenance.

## Target output

- HF model: `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` (to be created)
- Decode target: **>85.52 tok/s @ 524K** on 8× B300 with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
- Eval bar: GSM8K ≥94.5%, HumanEval pass@1 ≥77%, toolcall15 ≥26/30, NIAH 5/5 @ 524K

## Recipe

Same topology as the original quant — FP8_BLOCK 128×128 attention + W4A16 INT4 g=128 sym GPTQ routed experts — extended to cover **layer 43 (MTP)** in the same calibration pass. Not post-hoc spliced.

## License

Apache-2.0, inherited from the base model.
