# DeepSeek-V4-Flash-W4A16-FP8-MTP — model card (draft)

> Status: draft, populated alongside the calibration run. Numbers marked
> `<pending>` are filled in after Phase 2/3/5/6 complete. Treat this as a
> working document; it will be the published model card at HF release.

## Summary

The first DeepSeek-V4-Flash quantization that **preserves the
Multi-Token-Prediction (MTP) draft head**, enabling vLLM's
`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`
out of the box.

| Component | Format | Why |
|---|---|---|
| Main 43 decoder layers, routed experts (256 × 3 projections) | **W4A16 INT4 group=128 sym, GPTQ** | Compress the bulk of the parameters |
| Main 43 decoder layers, attention (q_a/q_b/kv/o_a/o_b + compressor + indexer) | **FP8_BLOCK 128×128** | Predecessor's validated kernel path |
| Main 43 decoder layers, shared experts + norms + gates + hyper-connection | **BF16** (passthrough) | Predecessor's recipe; safety-critical paths |
| **MTP draft block** (`mtp.0.*`) | **BF16** (deliberate, no quantization) | Preserve speculative-decoding acceptance rate |
| `lm_head` | BF16 | Output head untouched |

### Why MTP stays BF16

MTP is the speculative-decoding draft head. Decode throughput speedup from
MTP depends on **token-acceptance-rate** by the verifier model — what
fraction of the draft's proposed tokens the verifier accepts. Quantization
noise in the draft block directly degrades acceptance, which kills the
speedup.

- DeepSeek's own native release leaves the MTP block at higher precision
  than the MXFP4 experts.
- RedHat's `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` dropped MTP entirely.
- Predecessor `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` shipped without MTP
  (silently — see `huggingface/transformers#46129`).

We preserve MTP at full BF16 precision while still quantizing the main
43 layers' routed experts to W4A16 (where the bulk of the parameters
live). Cost: ~10 GB more on disk (`MTP_BF16 ≈ 13.2 GB` vs
`MTP_W4A16 ≈ 3.3 GB`, a ~7% overhead on a ~146 GB → ~156 GB artifact).
Benefit: full MTP token-acceptance rate, expected ~1.8× decode speedup
at `num_speculative_tokens=2`, against a worst-case ~30% throughput hit
from degraded acceptance if MTP were quantized aggressively.

## Calibration

| Field | Value |
|---|---|
| Hardware | 8× H200 (Hopper SM 9.0, p5en.48xlarge, us-east-2) |
| Dataset | `HuggingFaceH4/ultrachat_200k`, split `train_sft`, seed 42 |
| Samples | 768 |
| Max seq length | 512 |
| Per-rank batch size | 4 |
| Wall clock | `<pending Phase 2 completion>` |
| Recipe entry | `scripts/quantize_v4_w4a16_mtp.py` |

The recipe targets the same FP8_BLOCK attention + W4A16 routed-expert
topology as the predecessor `canada-quant/DeepSeek-V4-Flash-W4A16-FP8`,
extended to deliberately exclude the MTP block from quantization via
`ignore=["lm_head", r"re:.*mtp\\..*"]`.

## Serving

Validated configuration (after Phase 5 smoke serve):

| Field | Value |
|---|---|
| vLLM build | `jasl/vllm@3424fba5` + canada-quant patches (`patches/UPSTREAM_PR_DRAFTS.md`) |
| Tensor parallelism | TP=2 (per predecessor — TP≥4 hits `vllm-project/vllm#41511`) |
| Speculative decoding | `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` |
| Reasoning parser / tool-call parser | `deepseek_v4` (predecessor compatible) |
| kv-cache dtype | `fp8` |

vLLM `vllm.models.deepseek_v4.nvidia.mtp.DeepSeekV4MTP` handles the MTP
draft head; it does not depend on a transformers-side MTP class.

## Quality

`<pending Phase 6 benchmark suite>`

Reference (predecessor, no MTP):

| Benchmark | Predecessor on H200 (no MTP) | This artifact (with MTP) |
|---|---|---|
| chat-smoke quick | 4/4 | `<pending>` |
| chat-smoke quality | 4/4 | `<pending>` |
| chat-smoke coding | 2/2 | `<pending>` |
| GSM8K 8-shot flexible | 92.87% ±0.71% | `<pending>` |
| MMLU 5-shot | 87.27% ±0.27% | `<pending>` |
| HumanEval pass@1 | 54.27% ±3.9% (regex extraction) | `<pending>` |
| MTP token-acceptance rate (pos-0) | N/A | `<pending>` |
| Decode throughput @ 524K (MTP=2) | N/A | `<pending>` (target: >85.52 tok/s "Acti" reference) |

## Upstream contributions

This artifact's release is paired with upstream contributions that
make MTP-preserving DSv4 quantization tractable for anyone else:

| # | Upstream | Status | What it fixes |
|---|---|---|---|
| #2734 | `vllm-project/llm-compressor` | issue | GPTQModifier hangs on multi-rank disjoint-shard MoE |
| #2735 | `vllm-project/llm-compressor` | issue | DSv4 example silently drops MTP layer |
| #2736 | `vllm-project/llm-compressor` | issue | compress_module_list line 304 cudaStreamSynchronize stall |
| #46127 | `huggingface/transformers` | PR | Add `DeepseekV4NextNPredictor` class so MTP keys load |
| #46129 | `huggingface/transformers` | issue | conversion_mapping doesn't cover mtp.* paths |
| #2739 | `vllm-project/llm-compressor` | PR | Extend ARCH_TO_2D_MAPPINGS for MTP |

All filed during the same calibration attempt as this artifact. Full
diagnostic write-ups in `CONTRIBUTIONS_QUEUE.md` and
`FINDINGS_FOR_SIBLING.md`.

## Reproducibility

Single-file bootstrap in `scripts/bootstrap_p5en_h200.sh`. Run
`torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16_mtp.py
--input /scratch/weights/bf16-mtp --output <out>
--samples 768 --batch-size 4 --max-seq-len 512`. Calibration takes
~14h on 8× H200 (predecessor's measured wall clock).

## License

Apache-2.0, inherited from the base model (which is MIT). Companion
vendored upstream files under `vendor/` retain their original licenses.

## Citation

`<pending HF release>`
