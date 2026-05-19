# Patch provenance

Generated 2026-05-19.

## Upstream pins

### transformers (HuggingFace)
- pinned: `transformers==5.8.1`
- DSv4 architecture landed but `DeepSeekV4MTP` class is missing — the
  `from_pretrained` path silently drops every `mtp.*` weight tensor.
- file patched: `transformers/models/deepseek_v4/modeling_deepseek_v4.py`

### llm-compressor (vllm-project)
- repo: https://github.com/vllm-project/llm-compressor.git
- branch: `kylesayrs/transformers-v5` (PR #2647, still open)
- SHA: `f2aa32e2bde1941182d8f8a348837574969335e6`
- HEAD subject: `solid implementation`
- file patched: `src/llmcompressor/pipelines/sequential/helpers.py`
- Predecessor pin was `a308bc0e02181a46567a54fcfd082c9fb89e0337` (same branch,
  earlier commit). The patch was re-rebased here against the new SHA: the
  surrounding context grew comments (kylesayrs left a `# if isinstance(a, )`
  placeholder), so the predecessor's literal diff hunks no longer apply.

### compressed-tensors (vllm-project)
- pinned: `compressed-tensors==0.15.0.1`
- the predecessor used `>=0.15.1a2` which was a phantom alpha pin; 0.15.0.1
  is the actual released version with the needed APIs.

### vLLM (jasl/codex/ds4-sm120-min-enable)
- repo: https://github.com/jasl/vllm.git
- SHA: `3424fba51301504262c3d8355e2560469f18c9c4`
- HEAD subject: `Fix DeepSeek V4 MTP small-batch graph hangs`
- includes: upstream main through 2026-05-17 rebase, PR #42930 (CUDA MTP),
  workspace pre-reservation (was a separate patch in the predecessor).
- post-refactor layout (PR #43004–#43077, 2026-05-19):
  - `vllm/model_executor/models/deepseek_v4.py` ->
    `vllm/models/deepseek_v4/nvidia/model.py`
  - `vllm/model_executor/models/deepseek_v4_mtp.py` ->
    `vllm/models/deepseek_v4/nvidia/mtp.py`
  - The reasoning-agent's `patch_mtp_mapping.py` targeted the pre-refactor
    path; this repo's `scripts/patch_mtp_packed_mapping.py` targets the new
    layout and drops the obsolete weight_scale_inv -> weight_scale global
    replacement (post-refactor mtp.py already chooses the right suffix per
    expert dtype at mtp.py:357-389).

## Patches in this directory

### `modeling_deepseek_v4.py.diff`
Skips `DynamicCache` auto-construction when `past_key_values is None`. With
the V4-Flash config (`layer_types=None`), `DynamicCache(config=...)` falls
back to generic `DynamicLayer` which lacks `store_compression_weights`; the
V4 compressor then crashes calling that method during calibration. Leaving
`past_key_values=None` takes the cache_layer-is-None branch of
`compressor.forward`, which is the right path for GPTQ (no decode-style
accumulation needed).

Same content as the predecessor's patch; the touched lines (~1177 in
modeling_deepseek_v4.py) are stable across the 5.8.x line.

### `helpers.py.diff`
Adds Cache-class handling to `SequentialTracer.create_arg`. Without it, fx
tracing of DSv4 raises
`NotImplementedError: argument of type: <class transformers.cache_utils.DynamicCache>`.
Implementation mirrors the existing `PretrainedConfig` branch: emit a
fresh empty-constructor call so the traced graph constructs a real Cache at
runtime. GPTQ hooks Linear inputs/outputs only, so it does not need real
cache state.
