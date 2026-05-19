# dsv4-flash-w4a16-fp8-mtp

Re-do of [`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8) with the **MTP layer (layer 43) correctly included in the GPTQ calibration pass**, targeting AWS `p6-b300.48xlarge` (8× B300, Blackwell DC SM 10.0).

Sibling, non-overlapping scope to [`dsv4-flash-reasoning-agent`](https://github.com/pasta-paul/dsv4-flash-reasoning-agent) (which adds SFT + GRPO on top of the quant). This repo is **quant-only**.

## Status

**Phase 0 — instance bring-up.** EC2 launched (`i-0714f36a266c8c59b`, `us-west-2`), drivers + HBM verified, repo skeleton in place. Next: format/mount `/data`, set up the two venvs, apply the four local patches, kick off dequant.

See [`PLAN.md`](PLAN.md) for the full phase-by-phase execution plan.

## Target output

- HF model: `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` (to be created)
- Decode target: **>85.52 tok/s @ 524K** on 8× B300 with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
- Eval bar: GSM8K ≥94.5%, HumanEval pass@1 ≥77%, toolcall15 ≥26/30, NIAH 5/5 @ 524K

## Recipe

Same topology as the original quant — FP8_BLOCK 128×128 attention + W4A16 INT4 g=128 sym GPTQ routed experts — extended to cover **layer 43 (MTP)** in the same calibration pass. Not post-hoc spliced.

## License

Apache-2.0, inherited from the base model.
