---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash
tags:
- deepseek
- deepseek_v4
- quantized
- w4a16
- fp8
- mtp
- speculative-decoding
- text-generation
- conversational
pipeline_tag: text-generation
library_name: transformers
---

# DeepSeek-V4-Flash — W4A16-FP8 with MTP

This is a quantized re-pack of [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) with the **multi-token-prediction (MTP) layer included**:

| | this repo | predecessor `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` | upstream `deepseek-ai/DeepSeek-V4-Flash` |
|---|---|---|---|
| Routed experts | W4A16 INT4 g=128 sym | W4A16 INT4 g=128 sym | FP4 (e2m1, mxfp4) |
| Attention / dense | FP8_BLOCK 128×128 | FP8_BLOCK 128×128 | FP8 e4m3 |
| **MTP block (layer 43)** | **included, quantized** | **dropped silently** | FP4/FP8 |
| Size on disk | ~146 GB | ~140 GB | 160 GB |
| Calibration | RTN (model_free) | GPTQ (oneshot) | n/a |

## Why this exists

`transformers` 5.8.1's `DeepseekV4PreTrainedModel` declares
```python
_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
```
which silently drops every `mtp.*` tensor on `from_pretrained`. The predecessor quant trusted that load path and shipped without the MTP block — losing the speculative-decoding head that gives DeepSeek-V4-Flash its serving-time speedup. This re-quant bypasses the silent drop (preserves MTP through dequant) and includes the MTP block in the quantization recipe.

## Recipe

Names below are DeepSeek's *internal* naming convention (which is what the upstream HF safetensors actually use — not the HF-style `model.layers.X.self_attn.q_a_proj` that transformers' tooling typically expects).

- `re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$` → **W4A16** INT4 g=128 sym
- `re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$` → **FP8_BLOCK** 128×128
- `re:mtp\.\d+\.(e_proj|h_proj)$` → **FP8_BLOCK** (MTP entry projections)
- everything else (norms, gates, shared experts, hc_*, attn_sink, compressor/indexer) → **BF16 passthrough**

## Serving with vLLM

```bash
# Single GPU pair (TP=2). Marlin MoE TP>2 bug #41511 still open in vLLM main
# as of 2026-05-19, so deploy 4× TP=2 across the 8 GPUs of a B300 box.
vllm serve canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --max-model-len 524288 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --trust-remote-code
```

You need a vLLM with the two upstream patches applied: the `DeepseekV4ForCausalLM` and `DeepSeekV4MTP` classes both need `packed_modules_mapping`. Patch scripts at [`canada-quant/dsv4-flash-w4a16-fp8-mtp/scripts/patch_v4_forcausal_packed_mapping.py`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp) and `patch_mtp_packed_mapping.py`.

The pinned vLLM commit known to load this checkpoint cleanly is `jasl/vllm@3424fba51301504262c3d8355e2560469f18c9c4`.

## Known caveats

1. **RTN, not GPTQ.** The shipping artifact uses `llmcompressor.entrypoints.model_free.model_free_ptq` which does round-to-nearest weight quantization without Hessian refinement. FP8_BLOCK on BF16 weights is essentially lossless via RTN; W4A16 has measurable quality loss vs GPTQ on dense layers but is acceptable for the MTP draft head (where the metric is acceptance rate, not pure logit fidelity). For users who want GPTQ refinement, the [scripts/upstream/](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/tree/main/scripts/upstream) adapter is staged in the source repo with a passing smoke test; see `PHASE2_DESIGN.md`.
2. **Marlin MoE TP scale-sharding bug #41511** is still OPEN upstream as of 2026-05-19. Deploy as 4× TP=2 instances pinned to GPU pairs, NOT a single TP=8.
3. **DeepSeek encoding only.** Upstream ships no Jinja chat template; use the [`encoding/`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding) folder's Python helpers to encode messages.

## Hardware

Built on AWS `p6-b300.48xlarge` (8× B300, Blackwell DC SM 10.0, 275 GB HBM3e per GPU). Phase 1 dequant: 8m33s on a single B300. Phase 2 quantization: 7m total on a single B300.

## Reproducibility

Source repo + scripts + recipe + every patch: <https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp>. README of the source repo carries the full reproduce-from-scratch CLI.

## Credits

- DeepSeek for the base model and the inference reference implementation.
- The `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` predecessor whose recipe topology this extends.
- "Acti" for publishing the first MTP-enabled DSv4 quant — comparison point at 85.52 tok/s @ 524K on Blackwell DC.

## License

MIT, inherited from the base model.
