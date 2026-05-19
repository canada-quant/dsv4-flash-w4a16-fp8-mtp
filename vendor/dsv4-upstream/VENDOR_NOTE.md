# Vendor note (this directory)

These files are verbatim copies from
[`deepseek-ai/DeepSeek-V4-Flash/tree/main/inference`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference)
fetched on 2026-05-19, used here as the reference implementation for the
Phase 2 MTP shim (see `/PHASE2_DESIGN.md`).

| file | LOC | role |
|---|---|---|
| `model.py` | 827 | `Transformer`, `Block`, `MTPBlock`, `Attention`, `MoE`, `Expert`, `Gate`, `Compressor`, `Indexer`, `RMSNorm`, `Linear`, `ModelArgs` |
| `kernel.py` | 536 | custom GPU kernels: `sparse_attn`, `hc_split_sinkhorn`, `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm` |
| `config.json` | 34 | upstream ModelArgs override example |
| `requirements.txt` | 4 | upstream pip pins |
| `README.md` | 26 | upstream's own usage doc (NOT this file) |

**Not modified here.** Adaptations for our calibration use case will live in
a sibling `scripts/upstream/` tree when Phase 2 lands.

Upstream license is MIT (see the model card). These files inherit that
license.
