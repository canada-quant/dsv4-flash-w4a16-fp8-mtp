# Upstream PR drafts — canada-quant brand

Two patches drafted during the W4A16+MTP calibration work (2026-05-20).
Both are upstream-PR candidates that address the same gap from different
sides: adding first-class support for the DSv4 MTP (multi-token-prediction)
draft head in the standard `transformers` + `llm-compressor` calibration
pipeline.

## P1 — `huggingface/transformers`: add `DeepseekV4NextNPredictor` class

**File:** `patches/transformers_dsv4_mtp.py.diff`
**Runtime shim:** `scripts/transformers_mtp_shim.py` (`install_mtp_shim()`)
**Status:** draft, locally verified — class instantiates cleanly with
the real DSv4-Flash config and exposes all expected
submodules + parameters via `named_modules()` / `named_parameters()`.

### What it does

Adds `DeepseekV4NextNPredictor` (a `DeepseekV4DecoderLayer` subclass with
`e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`, and `hc_head_*` parameters)
to `transformers.models.deepseek_v4.modeling_deepseek_v4`. Modifies
`DeepseekV4Model.__init__` to instantiate `self.mtp` as a
`nn.ModuleList` when `config.num_nextn_predict_layers > 0`. Also
auto-extends `config.layer_types` and `config.mlp_layer_types` to cover
the MTP layer's index.

### Why it's needed

Without this class, `from_pretrained` either:
- Silently drops every `mtp.*` key (current behavior with
  `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`), or
- After our patch hunk that empties that regex (which canada-quant
  carries in `patches/modeling_deepseek_v4.py.diff`), loads the
  keys into the state dict but they have no `nn.Module` to attach
  to — so `named_modules()` never enumerates them and any
  downstream `linearize_moe_model` / `oneshot` calibration silently
  skips them.

### Verified locally

```
$ python -c "
from scripts.transformers_mtp_shim import install_mtp_shim
install_mtp_shim()
import transformers.models.deepseek_v4.modeling_deepseek_v4 as m
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/scratch/weights/bf16-mtp')
block = m.DeepseekV4NextNPredictor(cfg, layer_idx=cfg.num_hidden_layers)
print('params:', sum(p.numel() for p in block.parameters())/1e9, 'B')
"
[mtp-shim] installed: DeepseekV4NextNPredictor + Model.mtp
OK: MTP block instantiated
params: 6.629190907 B
missing expected children: none
hc_head_* params: ['hc_head_fn', 'hc_head_base', 'hc_head_scale']
```

### Open work for the upstream PR

1. Implement `DeepseekV4NextNPredictor.forward()` for inference-time MTP
   draft (the shim deliberately omits this — for calibration we only
   need the weights to land in `nn.Modules`). Reference implementation is
   `DeepSeek-V4-Flash/inference/model.py:MTPBlock.forward` (vendored in
   our repo at `vendor/dsv4-upstream/model.py`).
2. Add tests under `tests/models/deepseek_v4/` for:
   - `from_pretrained` round-trip preserves `mtp.0.*` weights
   - `model.mtp[0]` is iterable via `named_modules()`
   - `num_nextn_predict_layers = 0` path (no MTP) still works
3. Wire `DeepseekV4Model.forward` to optionally invoke the MTP draft
   head (gated on a kwarg; current main forward is unchanged).
4. Decide naming: `DeepseekV4NextNPredictor` mirrors the config field
   `num_nextn_predict_layers`; `DeepseekV4MTP` is more readable but
   diverges from the config namespace.

## P2 — `vllm-project/llm-compressor`: extend `ARCH_TO_2D_MAPPINGS` for MTP

**File:** `patches/llmc_dsv4_mtp_conversion_mappings.diff`
**Status:** draft, regex tested locally against the upstream
checkpoint's `mtp.0.ffn.experts.<E>.{w1,w2,w3}` keys.

### What it does

Extends `ARCH_TO_2D_MAPPINGS["deepseek_v4"]` in
`src/llmcompressor/modeling/moe/conversion_mappings.py` with three
additional `WeightRenaming` entries that mirror the existing
`^layers\.(\d+)\.mlp\.experts\.(\d+)\.{w1,w2,w3}\.` patterns but
anchored at `^mtp\.(\d+)\.` instead. Once `transformers` has the MTP
class (P1), this lets `linearize_moe`'s `named_modules()` walk find
the MTP block's `FusedExpertsProtocol` instance and apply the same 2D
expert reshape.

### Why it's needed

`linearize_moe` is the canonical entry point for DSv4 calibration in
the current `kylesayrs/transformers-v5` branch (HEAD `8c533c21f`,
2026-05-20). Its regex anchor `^layers\.` excludes `mtp.*` paths, so
even with `DeepseekV4NextNPredictor` instantiated (P1), the MTP
block's MoE would be missed.

### Acceptance criteria

- Run a single-rank dry calibration via `load_quantizable_moe()` on
  `canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP` BF16 checkpoint.
- After `save_pretrained`, assert the resulting safetensors index
  contains `mtp.0.mlp.experts.*` keys with the expected `*_packed` /
  `*_scale` suffixes.

## Filing order

1. File P1 against `huggingface/transformers` first — it's the
   structural prerequisite for P2.
2. Once P1 is open (does not need to be merged), file P2 against
   `vllm-project/llm-compressor` cross-referencing P1.
3. Both PRs reference issue
   [vllm-project/llm-compressor#2735](https://github.com/vllm-project/llm-compressor/issues/2735)
   as the problem statement.

## How this lands canada-quant brand

Four upstream contributions filed during the same calibration attempt,
in chronological order:

| # | Issue / PR | Status |
|---|---|---|
| 1 | [llm-compressor#2734](https://github.com/vllm-project/llm-compressor/issues/2734) — GPTQModifier multi-rank disjoint-shard hang | 🔴 filed |
| 2 | [llm-compressor#2735](https://github.com/vllm-project/llm-compressor/issues/2735) — DSv4 example drops MTP layer | 🔴 filed |
| 3 | [llm-compressor#2736](https://github.com/vllm-project/llm-compressor/issues/2736) — compress_module_list line 304 sync stall | 🔴 filed |
| 4 | This file (P1 + P2) | ⏳ drafted; pending PR submission |

Filed during the week `@kylesayrs` committed "tested and validated
fallback pathway" (commit `8c533c21f`, 2026-05-20). Real
canada-quant brand-building during active upstream development.

## Authorship + filing process

User decides timing of upstream submission. Drafts are committed to
this repo under `patches/` so they're reviewable + harvestable. When
ready, file via `gh pr create` from a fresh fork of each upstream repo
referencing this README for the rationale + the per-file diffs as the
PR content.
