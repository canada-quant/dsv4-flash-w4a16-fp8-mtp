# PHASE2_GPTQ_DELTA — what's already scaffolded vs what's missing

**Context:** the previous session pivoted to `model_free_ptq` (RTN) to avoid writing
the MTP adapter shim. That pivot is reversed. This document is the step-3 gate
in the user's redirect: a concrete delta between the existing scaffold under
`scripts/upstream/` and what still needs to be built to run the original
**GPTQ-with-MTP** plan end-to-end. **Phase 2 code work is paused until this delta
is approved.**

The current RTN artifact has been renamed `/scratch/weights/w4a16-fp8-mtp-rtn-fallback`
so nothing serves it as canonical.

---

## What's scaffolded today

### `scripts/upstream/__init__.py` (181 LOC)

Side-effect-import shim that makes `vendor/dsv4-upstream/model.py` (the upstream
inference reference, 827 LOC) importable in our calibration venv:

- `sys.modules["kernel"] = scripts.upstream.kernel_shim` set before the vendored
  `model.py` is loaded so its `from kernel import act_quant, sparse_attn, ...`
  resolves to our PyTorch reference impls instead of tilelang.
- `sys.modules["fast_hadamard_transform"]` stubbed to identity (used only on the
  FP8 QAT path, not exercised by W4A16 calibration).
- `vendor/dsv4-upstream/model.py` loaded via `importlib.util.spec_from_file_location`
  (no `__init__.py` needed in vendor/).
- Upstream's custom `Linear` class (FP4/FP8/BF16-aware) is rebound in the vendored
  module's namespace to **`GPTQLinear(nn.Linear)`** — an `nn.Linear` subclass with
  upstream's `(in, out, bias=False, dtype=None)` signature, defaulting to BF16.
  This makes `isinstance(m, nn.Linear)` true for every Linear in the model, which
  is what `llmcompressor`'s GPTQ matcher uses to find calibration targets.
- `ColumnParallelLinear` / `RowParallelLinear` also rebound to `GPTQLinear`
  (behaviorally identical when `world_size == 1`).
- `world_size = 1`, `rank = 0` set on the vendored module — **hardcoded**, see
  Missing #4.
- Re-exports `Transformer`, `Block`, `MTPBlock`, `Attention`, `MoE`, etc., and a
  `build_model_args(upstream_config_path)` that reads the upstream config.json
  and forces `dtype="bf16"`, drops `expert_dtype="fp4"`/`scale_fmt`, sets small
  `max_batch_size`/`max_seq_len`.

### `scripts/upstream/kernel_shim.py` (173 LOC)

PyTorch reference impls of the six tilelang kernels the vendored model imports:

- `sparse_attn(q, kv, attn_sink, topk_idxs, scale)` — gathers topk KV rows per
  query position (clamping `topk_idxs == -1` to 0 and masking those positions to
  `-inf` in scores so they contribute 0 after softmax), appends `attn_sink` as an
  extra logit column, softmaxes, weighted sum. Score+softmax in fp32 for
  numerical stability with `-inf` masking, then cast back to `q.dtype`.
- `hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)` —
  splits the `(2+hc_mult)*hc_mult` mix into `pre` (softmax), `post` (softmax),
  and `comb` (Sinkhorn iterations alternating row/col normalization on
  `comb_logits.exp()`).
- `act_quant`, `fp4_act_quant` — identity passthroughs when `inplace=True`
  (W4A16 doesn't quantize activations); raise `NotImplementedError` for
  non-inplace mode.
- `fp8_gemm`, `fp4_gemm` — defensive `NotImplementedError` (calibration uses
  `nn.Linear.forward` directly via the rebound `Linear` class, so the upstream
  free-function dispatcher never reaches these).

### `scripts/load_bf16_into_transformer.py` (204 LOC)

CLI:
```
python scripts/load_bf16_into_transformer.py \
    --weights /scratch/weights/bf16-mtp \
    --config  vendor/dsv4-upstream/config.json \
    [--probe-forward]
```

- Builds `ModelArgs` from upstream `config.json` via `build_model_args(...)`.
- Instantiates the (shimmed) upstream `Transformer` on CPU (~568 GB BF16).
- Walks every safetensors key in `/scratch/weights/bf16-mtp`, applies a tiny
  `RENAME_RULES` map for embed/head/norm aliases, and copies into
  `transformer.state_dict()[name]`. Records 2 known `PARAM_ALIASES`
  (`mtp.0.embed.weight` and `mtp.0.head.weight` share Parameter objects with
  the main model's `embed`/`head`).
- Reports unmatched-in-safetensors and missing-in-model keys.
- Verified on **meta-device dry run** during the previous session: 35,022 model
  keys vs 35,020 safetensors keys, 0 unmatched, 2 expected aliases missing.
- **Never actually executed end-to-end against real weights** — the previous
  session aborted load mid-RAM-allocation at ~173 GB. See Missing #5.

### `scripts/smoke_test_adapter.py` (112 LOC)

Tiny-model smoke test on a randomly-initialized 2-layer / 8-expert / dim-256
shimmed `Transformer`. **PASSES** as of 2026-05-19 with:

```
forward (main):  logits.shape=[1, 1024] finite=True min=-1.000 max=0.979
forward (MTP):   mtp_logits.shape=[1, 1024] finite=True min=-1.035 max=1.018
SMOKE_TEST_OK
```

What this **proves**:
1. The vendored model can be imported with kernel/fht shimmed.
2. `nn.Linear` rebinding works — every `Linear(...)` call in `Attention`/`MoE`
   constructs an `nn.Linear` instance.
3. The shimmed `sparse_attn` and `hc_split_sinkhorn` don't NaN out on tiny
   inputs (with proper init).
4. **`model.mtp[0](h, 0, input_ids)` runs to completion as a standalone forward** —
   so MTP is a real `nn.Module` and can be invoked separately.

What this **does not prove**:
1. `model(input_ids)` flows through MTP — it does NOT. The upstream
   `Transformer.forward` (line 802 of vendor/dsv4-upstream/model.py) only
   iterates `self.layers`; `self.mtp[0]` is never called. The smoke test invokes
   MTP separately with hand-crafted `h`. See Missing #2.
2. Real BF16 weights load cleanly and forward produces numerically reasonable
   logits — only random-init was tested. See Missing #5.
3. `llmcompressor.oneshot` accepts the shimmed Transformer as `model=`.

### `scripts/quantize_v4_w4a16_mtp.py` (in repo, pre-RTN pivot)

Currently a documented stub. It defines `CalibrationModel(nn.Module)` wrapping
the transformer with a forward that flows main → MTP, defines the recipe with
correct internal-naming regexes, defines the V4 manual chat preprocessing, and
calls `llmcompressor.oneshot(model=CalibrationModel(...), ...)`. The previous
session never confirmed the `oneshot` accepts a non-`PreTrainedModel` wrapper.
The current `main()` raises immediately (intentional stub since the previous
session pivoted to RTN before validating this end-to-end).

### `vendor/dsv4-upstream/`

Verbatim copies of upstream `inference/model.py` (827 LOC), `kernel.py`
(536 LOC), `config.json`, `requirements.txt`, `README.md`. These are not
modified — adaptations live in `scripts/upstream/__init__.py` via namespace
rebinding.

---

## Answering the user's three specific questions

### (a) Does `load_bf16_into_transformer.py` actually load the BF16+MTP weights into a patched `DeepseekV4ForCausalLM`?

**No.** It loads into the **vendored upstream `Transformer`** (from `vendor/dsv4-upstream/model.py`), not into `transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4ForCausalLM`. These are two different module trees with different parameter naming conventions:

| | vendored upstream `Transformer` | `transformers` `DeepseekV4ForCausalLM` |
|---|---|---|
| Attention names | `attn.wq_a`, `attn.wkv`, `attn.wo_a`/`wo_b` | `self_attn.q_a_proj`, `self_attn.kv_proj`, `self_attn.o_a_proj`/`o_b_proj` |
| MoE names | `ffn.experts.X.w1/w2/w3` | `mlp.experts.X.gate_proj/down_proj/up_proj` |
| Norm names | `attn_norm`, `ffn_norm` | `input_layernorm`, `post_attention_layernorm` |
| HC params | top-level `hc_attn_fn` etc. | wrapped in `attn_hc.fn`/`attn_hc.base`/`attn_hc.scale` submodules |
| MTP class | **present** (`MTPBlock` at line 738) | **absent** |

There is a transformers-side conversion shim — `transformers/conversion_mapping.py` has a `deepseek_v4` entry (~80 LOC of `WeightRenaming` rules) that maps internal → HF names on `from_pretrained` load. But it does **not** handle `mtp.*` keys (transformers has no MTP class to map them into).

The scaffolded path loads into the **upstream** Transformer, which uses internal names natively and has an `MTPBlock`. This is the right call — it preserves MTP as a first-class submodule. But it means we are NOT using `transformers.AutoModelForCausalLM.from_pretrained`; we're loading into a vendored class.

### (b) Does `scripts/upstream/__init__.py` expose MTP as a real `nn.Module` inside the forward graph so GPTQ hooks can attach?

**Partial yes.** The upstream `Transformer.__init__` at line 789 does instantiate
`self.mtp = nn.ModuleList([MTPBlock(args.n_layers + layer_id, args)])`, with
`self.mtp[-1].embed = self.embed` and `self.mtp[-1].head = self.head` aliased to
share the main model's embedding and head. So:

- `model.named_modules()` enumerates `mtp.0.attn.wq_a`, `mtp.0.attn.wkv`,
  `mtp.0.ffn.experts.X.w1`, `mtp.0.e_proj`, `mtp.0.h_proj`, etc. ✓
- `isinstance(model.mtp[0].attn.wq_a, nn.Linear)` is True (via the GPTQLinear
  rebind) ✓
- GPTQ's hook-attachment phase CAN hook these Linears ✓

**But the forward graph does not route through MTP.** `Transformer.forward`
(line 802-808) only iterates `self.layers`. During a standard
`model(input_ids)` call, `self.mtp[0]` is never invoked, so:

- The hooks attached to MTP Linears never receive activations
- GPTQ's Hessian accumulation phase collects zero data for MTP
- The final quantization for MTP would have undefined behavior (most likely it
  would skip the layer or quantize with the initialization-time noise, neither
  of which is what we want)

To fix: a wrapper or monkey-patched `forward` that also runs `self.mtp[0](h, 0,
input_ids)` after the main loop. The previous session's
`quantize_v4_w4a16_mtp.py` stub sketched this as `CalibrationModel(nn.Module)`
— it needs to be reinstated and tested. See Missing #2.

### (c) What does `smoke_test_adapter.py` actually prove — does it run a forward pass through the MTP block or just instantiate the class?

**Both, but separately.** Two separate forward invocations:

1. Line 88: `logits = model(input_ids)` — full main forward on the tiny 2-layer
   model. Checks finite logits. This invocation does **NOT** flow through MTP
   (see (b)).
2. Line 99: `mtp_logits = model.mtp[0](h, 0, input_ids)` — invokes
   `MTPBlock.forward` directly with a hand-crafted `h` of shape
   `[1, 16, hc_mult, dim]` (random gaussian). Checks finite mtp_logits.

So the smoke test proves the MTP forward path **works in isolation** given a
plausible-shaped `h` input. It does **not** prove the main → MTP chain works
when MTP is fed real layer-42 outputs. It does not prove real weights load. It
does not prove `llmcompressor.oneshot` accepts the model.

---

## What's missing — the concrete delta

Numbered so the user can approve / push back per item.

### Missing 1 — `oneshot(model=...)` bridge (estimated ~50 LOC)

`llmcompressor.oneshot(model=str | PreTrainedModel, ...)` expects either a path
string (resolves via `transformers.AutoModelForCausalLM.from_pretrained`) or
already-loaded `PreTrainedModel`. Our shimmed upstream `Transformer` is
neither. Inside oneshot the model needs:

- `model.save_pretrained(output_dir)` — to write the compressed checkpoint
- `model.config` — for various pipeline hooks
- Potentially `model.generation_config`

Two options:
- **Option A (preferred):** wrap in a thin `PreTrainedModel` subclass with a
  stub `PretrainedConfig` and a custom `save_pretrained` that writes
  `transformer.state_dict()` via `safetensors.save_file` and a config.json
  carrying the right `quantization_config` block.
- **Option B:** drop `oneshot` and call the lower-level
  `GPTQModifier.apply(model, calibration_data, sequential_targets=[...])`
  directly, then save manually. Skips the save_pretrained dependency entirely
  but more friction with the oneshot lifecycle (less battle-tested).

Will prototype Option A first.

### Missing 2 — Forward wrapper that flows main → MTP (estimated ~30 LOC)

The `CalibrationModel(nn.Module)` from the previous session's stub. Goes back
into `scripts/quantize_v4_w4a16_mtp.py`:

```python
class CalibrationModel(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
    def forward(self, input_ids, **_):
        t = self.transformer
        h = t.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, t.hc_mult, 1)
        for layer in t.layers:
            h = layer(h, 0, input_ids)
        for mtp_layer in t.mtp:                # ← drives MTP for GPTQ hooks
            _ = mtp_layer(h, 0, input_ids)
        logits = t.head(h, t.hc_head_fn, t.hc_head_scale, t.hc_head_base, t.norm)
        return _LogitsOut(logits)              # HF-style output with .logits
```

Sequential calibration with `sequential_targets=["Block"]` will cover both main
`Block` and MTP's inner `Block` (MTPBlock inherits Block). `MTPBlock`'s own
attributes (`e_proj`, `h_proj`, etc.) get caught by the recipe regexes
directly: `re:mtp\.\d+\.(e_proj|h_proj)$` matches them.

### Missing 3 — calibration corpus (no new code, just config)

Predecessor's `quantize_v4_w4a16.py` (verbatim from `/tmp/dsv4-sources/dsv4-flash-w4a16-fp8/scripts/quantize_v4_w4a16.py`, lines 100-140):

```python
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
ds = load_dataset(DATASET_ID, split=get_rank_partition(DATASET_SPLIT, args.samples))
ds = ds.shuffle(seed=42).map(lambda ex: preprocess(ex, tokenizer))
# 768 samples, max_seq_len=512, batch_size=4
```

with the V4 manual chat encoding (no Jinja template):

```python
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
def preprocess(example, tokenizer):
    text = BOS
    for msg in example["messages"]:
        if msg["role"] == "system":    text += msg["content"]
        elif msg["role"] == "user":    text += f"<｜User｜>{msg['content']}"
        elif msg["role"] == "assistant": text += f"<｜Assistant｜></think>{msg['content']}{EOS}"
    return {"text": text}
```

Pin in PLAN.md: **HuggingFaceH4/ultrachat_200k train_sft, 768 samples, seed=42,
max_seq_len=512, batch_size=4, V4 manual chat encoding above.** The user's
fallback (c4/en + openassistant-guanaco at 2048) is not needed — the predecessor's
corpus is recoverable from the cloned source. (The user offered to search past
chats; that's not required, we have it in `/tmp/dsv4-sources/dsv4-flash-w4a16-fp8/`.)

### Missing 4 — dist-aware `world_size` in the shim (estimated ~10 LOC)

`scripts/upstream/__init__.py` lines 108-109 hardcode `world_size = 1, rank = 0`.
Calibration via `torchrun --nproc-per-node 8` (the predecessor and PLAN.md path)
sets `torch.distributed` env vars; the shim should set:

```python
import torch.distributed as dist
_module.world_size = dist.get_world_size() if dist.is_initialized() else 1
_module.rank = dist.get_rank() if dist.is_initialized() else 0
```

This must run **after** distributed init (so probably move into the
calibration script and assign right before instantiating `Transformer`).

With `world_size=8`, the upstream `MoE.__init__` shards 256 experts across
ranks (32 per rank), which is the memory shape the predecessor relied on
(otherwise each rank holds a full 568 GB BF16 model, OOMing immediately).

### Missing 5 — end-to-end real-weights load + forward smoke test (no new code, ~20 min runtime)

`load_bf16_into_transformer.py` has only been validated on a meta-device dry run.
Need a real run that:
1. Instantiates `Transformer(margs)` on CPU (~568 GB BF16).
2. Loads the 46 safetensors shards from `/scratch/weights/bf16-mtp` into CPU
   state_dict. Expects ~10-15 min wall clock at ~1 GB/s SSD read.
3. Moves a single `Block` to GPU 0.
4. Runs one forward batch through `CalibrationModel` (Missing #2) — checks
   finite logits, no shape errors.

This is the load-bearing dress rehearsal. If it fails, the 8-12 h calibration
fails the same way 8-12 hours from now. Cheap to do up front.

### Missing 6 — rewrite `quantize_v4_w4a16_mtp.py` (estimated ~150 LOC)

Currently the file is a documented stub. Final form needs:

- Init dist (`torch.distributed.init_process_group`) so the shim sees the
  right world_size/rank.
- Instantiate + load via `load_safetensors_into(transformer, ...)`.
- Wrap in `CalibrationModel` (Missing #2).
- If Option A from Missing #1: bridge to a `PreTrainedModel`-style class.
- Build dataset per Missing #3.
- Call `oneshot(...)` (or `GPTQModifier.apply(...)`) with:
  - recipe = FP8_BLOCK on attn (`wq_a|wq_b|wkv|wo_a|wo_b` + `mtp.X.e_proj|h_proj`),
            W4A16 on experts (`ffn.experts.X.w1|w2|w3`)
  - sequential_targets = `["Block"]` (covers both regular and MTPBlock)
  - ignore = norms, gates, shared_experts, hc_*, attn_sink, compressor/indexer
  - offload_hessians=True, dampening_frac=0.1
- Save the resulting state_dict via safetensors + write config.json with the
  proper compressed-tensors `quantization_config` block.

### Missing 7 — GPTQ-signature verification (~30 LOC)

New gate `scripts/verify_gptq_signature.py`:

- Read the post-calibration `model.safetensors.index.json` + a sample shard.
- Assert `quantization_config.config_groups[0].quantization_args.actorder` is
  present (`GPTQ` writes it; RTN doesn't).
- For one main-model layer, gather all `experts.X.w1.weight_scale` (256 of
  them) and compute their std dev. RTN tends to produce near-identical scales
  across experts; GPTQ spreads them. Threshold: `std/mean > 0.05` is the loose
  "this is GPTQ" check.
- Same check on `mtp.0` experts to confirm the MTP block was actually
  GPTQ-calibrated, not skipped.
- Exit non-zero if any check fails.

This is the receipt the user asked for — proves we didn't quietly RTN again.

---

## Honest effort estimate

| Piece | LOC | Risk |
|---|---|---|
| Missing 1 (PreTrainedModel bridge) | ~50 | Low — `save_pretrained` + a stub `PretrainedConfig` |
| Missing 2 (forward wrapper) | ~30 | Low — already sketched in prior stub |
| Missing 3 (dataset) | 0 new | None — verbatim port from predecessor source |
| Missing 4 (dist-aware shim) | ~10 | Low |
| Missing 5 (real load + forward smoke) | 0 new code | Medium — the 15-min real load might surface naming gaps the meta-device run didn't (we'll learn this fast) |
| Missing 6 (rewrite quantize script) | ~150 | Medium — integration with oneshot is the main unknown |
| Missing 7 (GPTQ signature verify) | ~30 | Low |
| **Total new code** | **~270 LOC** | |

Plus actual calibration wall clock: **8-12 h on 8× B300** per PLAN.md.

The previous session's "500+ LOC" figure was for reimplementing `MTPBlock`
from scratch (option A in `PHASE2_DESIGN.md`). We are going with option A'
(vendor + adapt), which the previous session already invested ~500 LOC in
(`scripts/upstream/` 354 LOC + `vendor/dsv4-upstream/` 1,400 LOC). The remaining
~270 LOC is the bridge code on top of that.

The scaffold is **roughly 60% there**: the architectural commitment is made,
the model is importable, the MTP block is reachable, real weights map
1-to-1. What's missing is the integration glue between the shimmed model and
`llmcompressor.oneshot`, plus the dress-rehearsal load test.

---

## Open questions for the user before proceeding

1. **Approve the option A' direction?** (vendored upstream + adapter, NOT
   transformers.DeepseekV4ForCausalLM). The user's note in the redirect said
   *"Patch transformers' modeling_deepseek_v4.py in place via the diff in
   patches/ — do not fork the file"* — but transformers does not have an MTP
   class to patch into, and the WeightRenaming pipeline in
   `transformers/conversion_mapping.py` has no rules for `mtp.*` keys. Adding
   an `MTPBlock` to transformers and registering rename rules for it would be
   substantially more work (~400-500 LOC patching the transformers package)
   than wrapping the upstream Transformer. **Want me to pursue that path
   instead?** It would be the cleanest long-term but slower this round.

2. **Calibration on 8× B300 vs single-GPU smoke first?** Predecessor used
   `torchrun --nproc-per-node 8` for 768 samples ≈ 8-12 h. A 4-sample
   single-GPU smoke run takes ~10 min and would prove the pipeline before
   committing the compute. Recommend smoke first.

3. **OK to use predecessor's exact corpus (HuggingFaceH4/ultrachat_200k)?**
   Confirmed it's documented in the cloned predecessor source. Records this
   in PLAN.md.

4. **Acceptance criteria for the GPTQ-vs-RTN check (Missing 7)?** Proposed:
   `actorder` present in `config_groups[0].quantization_args` AND per-expert
   scale std/mean > 0.05 on at least one sampled layer. Adjustable.

**Awaiting confirmation before writing code in Missing 1, 2, 6, 7.** Missing 3-5
do not write new code, so they're safe to do in parallel — happy to wait if
you'd rather they be on the same approval cycle.
