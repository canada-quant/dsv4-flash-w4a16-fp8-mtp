# dsv4-flash-w4a16-fp8-mtp

Re-do of [`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8) with the **MTP layer (layer 43) correctly included in the GPTQ calibration pass**, targeting AWS `p6-b300.48xlarge` (8× B300, Blackwell DC SM 10.0).

Sibling, non-overlapping scope to [`dsv4-flash-reasoning-agent`](https://github.com/pasta-paul/dsv4-flash-reasoning-agent) (which adds SFT + GRPO on top of the quant). This repo is **quant-only**.

## Status

**Phase 1 — dequant in progress.** EC2 box (`i-0714f36a266c8c59b`, `us-west-2`, 8× B300) bootstrapped; `/data` mounted (300 GB EBS, persistent), `/scratch` symlinked to instance-store NVMe (27.6 TB). `venv-calib` built and patched: `transformers==5.8.1`, `compressed-tensors==0.15.1a20260515` (alpha — the predecessor's "phantom" pin shipped on 2026-05-15), `llm-compressor@f2aa32e2`, `torch 2.12.0+cu130`. Both transformers hunks (mtp-retention + calibration cache) and the llm-compressor helpers hunk are applied; runtime asserts that `DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected == []`. Upstream weights (160 GB) downloading from `deepseek-ai/DeepSeek-V4-Flash`.

See [`PLAN.md`](PLAN.md) for the full phase-by-phase execution plan, [`patches/VERSIONS.md`](patches/VERSIONS.md) for patch provenance.

## Target output

- HF model: `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` (to be created)
- Decode target: **>85.52 tok/s @ 524K** on 8× B300 with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
- Eval bar: GSM8K ≥94.5%, HumanEval pass@1 ≥77%, toolcall15 ≥26/30, NIAH 5/5 @ 524K

## Recipe

Same topology as the original quant — FP8_BLOCK 128×128 attention + W4A16 INT4 g=128 sym GPTQ routed experts — extended to cover **layer 43 (MTP)** in the same calibration pass. Not post-hoc spliced.

## License

Apache-2.0, inherited from the base model.
