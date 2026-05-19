# dsv4-flash-w4a16-fp8-mtp

Re-do of [`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8) with the **MTP layer (layer 43) correctly included in the GPTQ calibration pass**, targeting AWS `p6-b300.48xlarge` (8× B300, Blackwell DC SM 10.0).

Sibling, non-overlapping scope to [`dsv4-flash-reasoning-agent`](https://github.com/pasta-paul/dsv4-flash-reasoning-agent) (which adds SFT + GRPO on top of the quant). This repo is **quant-only**.

## Status

**Phase 1 — dequant COMPLETE (2026-05-19 17:50 UTC).** Output at `/scratch/weights/bf16-mtp` on the live EC2 box: 46 safetensors shards, 543 GB BF16, 35,020 total tensors. **797 MTP weight tensors preserved** (verified by `scripts/verify_mtp_keys.py`; consistent with upstream 1,575 = 797 weights + 778 scales, where every scale was consumed into its paired BF16 weight). Spot-check on `mtp.0.attn.wq_a.weight` (post-dequant `[1024, 4096]` bfloat16, mean-abs 0.023, max 0.22) and `mtp.0.ffn.experts.0.w1.weight` (`[2048, 4096]` bfloat16, mean-abs 0.022, max 0.19) show sane magnitudes — no overflow, no underflow. EC2 box (`i-0714f36a266c8c59b`, `us-west-2`, 8× B300), `venv-calib` patched (`transformers==5.8.1`, `compressed-tensors==0.15.1a20260515`, `llm-compressor@f2aa32e2`, `torch 2.12.0+cu130`); `DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected == []` runtime-confirmed.

**Phase 2 — running via model_free RTN.** Discovered `llmcompressor.entrypoints.model_free.model_free_ptq` operates directly on safetensors files (no `PreTrainedModel` required), bypassing the MTP-class integration block that would otherwise need a 500+ LOC shim. Trade-off: round-to-nearest instead of GPTQ — FP8_BLOCK on BF16 is essentially lossless via RTN; W4A16 has measurable quality loss vs GPTQ but is acceptable for the MTP layer (used for speculative drafts). Recipe applied in two passes (W4A16 experts, then FP8_BLOCK attention + MTP entry projections) — script at [`scripts/quantize_v4_model_free.py`](scripts/quantize_v4_model_free.py).

For the GPTQ path (next session), the full adapter is staged at [`scripts/upstream/`](scripts/upstream/) — vendor model.py is loaded with `Linear` rebound to an `nn.Linear` subclass, `kernel.py` shimmed with PyTorch reference implementations of `sparse_attn` and `hc_split_sinkhorn`. The tiny-model smoke test passes (`scripts/smoke_test_adapter.py`). What remains is wiring the loaded BF16 model into `llmcompressor.oneshot` as a `PreTrainedModel` subclass. See [`PHASE2_DESIGN.md`](PHASE2_DESIGN.md).

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
