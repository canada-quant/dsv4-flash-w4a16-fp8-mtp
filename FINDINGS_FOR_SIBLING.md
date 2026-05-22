# Findings for the B300 NVFP4 sibling

Cross-pollination document from `canada-quant/dsv4-flash-w4a16-fp8-mtp`
(H200 W4A16+MTP path) to `canada-quant/dsv4-flash-nvfp4-fp8-mtp` (B300
NVFP4+MTP path). Both workstreams use `llmcompressor` on the same
`DeepSeek-V4-Flash` architecture with the same decoupled MoE expert
shard, so most of the diagnostic work transfers — but the two recipes
hit different code paths in `llmcompressor`, so the *which patch where*
question matters.

**Date:** 2026-05-20 (initial) / **2026-05-21 (Update 3 — cross-pollination cycle)**.
**Author:** H200 agent (this repo).
**Status:** mini-GPTQ smoke COMPLETED but with a SECONDARY HANG inside
`compress_module_list` (not in our patches). Phase 2 launch is BLOCKED on
diagnosing this second deadlock. See "Mini-smoke result" section below.

## Cross-pollination cycle (Update 3, 2026-05-21 09:30 PDT)

**Failure of cross-pollination caught.** This H200 agent spent 90 minutes
of 2026-05-21 morning re-discovering bugs the B300 sibling already filed
PRs for. Documenting what the sibling shipped so future cycles don't
repeat this:

The sibling filed 4 PRs against `vllm-project/vllm` mainline on
2026-05-20/21 that map directly to bugs both repos hit:

| PR | Fixes | Applies to W4A16 path? |
|---|---|---|
| [#43248](https://github.com/vllm-project/vllm/pull/43248) | `is_static_input_scheme=bool(...)` wrap | YES — affects FP8 attn schemes |
| [#43288](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` default `.get(..., "ue8m0")` | YES — format-agnostic config field |
| [#43290](https://github.com/vllm-project/vllm/pull/43290) | `attention.py:331` `weight_scale_inv`-or-`weight_scale` fallback | YES — SM90/SM120 shared path |
| [#43319](https://github.com/vllm-project/vllm/pull/43319) | Auto-detect BF16 MTP from safetensors index → skip `quant_config` for MTP tower | YES — format-agnostic, applies to all MTP-preserving artifacts |

The sibling also filed issue [#43304](https://github.com/vllm-project/vllm/issues/43304)
naming the MTP-quant-inheritance bug with three proposed fix shapes;
#43319 implements option (1) — auto-detect from safetensors.

**Sibling's serve build:** mainline `vllm-project/vllm` + PR #42209 (sychen52's
NVFP4 MoE support, near-merge, 3 approvals). They moved off `jasl/dm120`
because the sm120-optimized `fp8_einsum.py` kernel didn't match their
sm100a (B300) artifact layouts.

**H200 implication:** we run on Hopper SM90, our W4A16 main path uses
Marlin (not DeepGemm), so the sibling's sm100a kernel issue doesn't bite
us. But the 4 PRs above ARE applicable to our W4A16/H200 path —
verified by reading the diffs:
- #43290 patches `attention.py` in the SM90/SM120 shared code (per the
  in-line comment `# SM90/SM120: FP32 block scales stay [g, r/128, d/128]`)
- #43319 scans safetensors for ANY quant suffix
  (`.weight_scale|.weight_scale_inv|.weight_packed|.weight_global_scale|
  .input_global_scale|.weight_zero_point`) covering NVFP4 + W4A16 + FP8
- #43288, #43248 are config / scheme-resolution fixes, format-agnostic

**Standing-rule lesson:** "check upstream PRs before patching" was in
memory and not followed. The cost was 90 minutes plus one unilateral
Option Y violation (BF16 → FP8 conversion on MTP attention; rolled back).

**Sibling lessons we should adopt for serve:**
1. `update_config_for_fused_attn.py` shape — recipe `targets=` must use
   FUSED attention names (`fused_wqa_wkv`, `compressor.fused_wkv_wgate`)
   because vLLM mainline DSV4 uses `MergedColumnParallelLinear`, and its
   quant framework only allocates scale params on modules whose prefix
   matches the regex. Our existing targets= already includes these (good).
2. The MTP-preservation pattern with `ignore=[..., r"re:.*mtp\..*"]` is
   what unblocks the inference-tensor crash (`llm-compressor#2745`) and
   produces the BF16 MTP block on disk. Both repos converged on this.
3. The `getattr(..., "weight_scale_inv", None) or self.wo_a.weight_scale`
   fallback is the right shape — DO NOT quantize MTP attention to satisfy
   the hard-coded `weight_scale_inv` access at `attention.py:331`. That
   path was tried on H200 this morning and abandoned as an Option Y
   violation.

**TODO for both sides going forward:** when one repo files an upstream
PR/issue, cross-post the URL + scope into the other repo's
FINDINGS_FOR_SIBLING.md within the same session. This document is the
designated cross-pollination artifact; if it doesn't stay current, both
sides re-discover the same bugs.

---

## TL;DR for the B300 sibling

1. **Observer.synchronize hang is real**, but it only fires when
   activation observers are attached (RTN-style activation quantization).
   NVFP4 + FP8_BLOCK activations does fire it; W4A16 + FP8_BLOCK
   weight-only does not. **Apply the monkey-patch defensively anyway —
   it costs nothing and protects against subtle observer-creation
   paths.**

2. **GPTQ has a separate but parallel multi-rank hang** at
   `_reduce_hessian_to_target_rank` (`gptq/base.py:323`) and
   `_broadcast_quantized_params` (`gptq/base.py:350`). Same root cause
   (disjoint module sets across ranks; ranks call `dist.reduce` /
   `dist.broadcast` on different module subsets), different code path.
   **This is NOT a worry for your RTN path — you don't use
   `GPTQModifier`, so these methods never fire.**

3. **Predecessor recipe used HF auto-offload, not decoupled sharding** —
   confirmed by grep of `canada-quant/dsv4-flash-w4a16-fp8/scripts/quantize_v4_w4a16.py`.
   Zero references to `_expert_world_size`. That's why predecessor didn't
   hit any of these bugs. Our decoupled shard is genuinely new territory
   for both workstreams.

4. **DLAMI version mismatch broke CUDA on the H200 box.** Driver 595.64
   loaded; fabricmanager only available at 595.71.05. CUDA Error 802 out
   of the box. Fix sequence in `RECOVERY.md` section 1. If your B300
   DLAMI is a similar bake, check the driver/fabricmanager versions
   before you waste time on multi-rank work.

5. **`named_modules()` returns names without a leading dot at the
   top level** — `layers.0.ffn.experts.0`, not `.layers.0...`. If you're
   anchoring regexes against module paths, use `(?:^|\.)layers\.` not
   `\.layers\.`. We had a sharding-invariant regex that returned 0
   matches because of this; caught it in the loadtest before mini-smoke.

---

## The three monkey-patches we wrote

`scripts/multirank_patches.py` in this repo. Each carries a signature
guard, inline doc, and PR-candidacy note. Lift wholesale if useful.

### Patch A — `Observer.synchronize` → no-op when `world_size > 1`

**You should apply this.** It fires for RTN/NVFP4 recipes.

```python
import llmcompressor.observers.base as _obs_base
import llmcompressor.observers.moving_base as _obs_moving
_obs_base.Observer.synchronize = lambda self: []
_obs_moving.MovingAverageObserverBase.synchronize = lambda self: []
```

The owning rank computes qparams from its local stats. With 768/8 = 96
samples per rank, min/max observers have plenty to work with — the
cross-rank sync was for accuracy improvement, not correctness.

### Patch B — `GPTQModifier._reduce_hessian_to_target_rank` skip-sharded

**You don't need this** because your `QuantizationModifier` (RTN) path
doesn't compute Hessians. Skip it. (We keep it here for completeness in
case anyone reading this is on the GPTQ path.)

The pattern: pre-filter `module_list` to exclude `.ffn.experts.<id>.`
modules (sharded), then delegate to the original method for the
replicated subset.

### Patch C — `GPTQModifier._broadcast_quantized_params` skip-sharded

Same — you don't need this for RTN.

---

## What you should think about for RTN multi-rank

`QuantizationModifier.compress` (the RTN entrypoint) reads each module's
`weight_scale` and `weight_zero_point` from observers that were attached
during the calibration pass. The observer's `synchronize()` is the
single all-reduce point you need to gate on `world_size`.

Open question we have NOT verified for the NVFP4 path:

> Does `QuantizationModifier` call any *other* cross-rank collective
> beyond `Observer.synchronize` on the activation-observer path?

If yes, you'll need a second patch with the same "filter
module_list to exclude sharded modules" pattern. The loadtest is too
cheap not to use — instrument it the same way we did to assert
disjointness, then run a 1-layer NVFP4 smoke (analog of our
`--dry-run-one-layer`) before the full 8-rank run.

---

## Sharding invariant pattern

`scripts/multirank_patches.py::assert_sharding_invariant` does this:

1. Walk `model.named_modules()`, collect per-rank `(layer_id, expert_id)`
   tuples matching `(?:^|\.)layers\.(\d+)\.ffn\.experts\.(\d+)\b`.
2. `dist.all_gather_object` across ranks.
3. On rank 0: build a `(layer, expert) → [ranks_owning]` map. Assert
   every tuple has exactly 1 owner. Optionally: assert per-layer
   coverage equals `n_routed_experts`.

Result on our loadtest (8 ranks, p5en.48xlarge):

```
[shard-invariant] OK — 44 MoE layers, 11264 total (layer,expert) tuples,
disjoint across 8 ranks (per-rank counts: [1408, 1408, 1408, 1408, 1408,
1408, 1408, 1408])
```

44 main MoE layers × 256 experts ÷ 8 ranks = 1408 per rank. Disjoint.

**Caveat:** the MTP block (`mtp.0.ffn.experts.<id>`) was NOT picked up by
our regex on the first pass — the upstream `Transformer` wraps MTP's
sub-blocks differently and `named_modules()` doesn't surface
`mtp.<i>.ffn.experts.<j>` as a flat path. Open follow-up. Should affect
your NVFP4 recipe the same way if you're using the same vendored
upstream.

---

## DLAMI gotcha (cross-applies if you're on similar AMI)

See `RECOVERY.md` section 1 for the full repro and fix.

Short version: AMI `ami-0bae40837d7422a24` ships
- loaded driver: 595.64
- fabricmanager apt package: 595.71.05 (only version available)

`nv-fabricmanager` refuses to start (interface ABI must match exactly).
HGX H200 NVSwitch can't initialize, CUDA returns Error 802. Fix:
`sudo apt install nvidia-dkms-595-server linux-modules-nvidia-595-server-aws-6.17`
then `sudo reboot`. Instance store survives OS-level reboot. Then
`sudo apt install nvidia-utils-595-server libnvidia-compute-595-server`
to restore `nvidia-smi` userspace if the install dance broke it.

If your B300 box is from a comparable DLAMI bake, the same skew could
exist with the 595-server line — worth checking before debugging NCCL.

---

## Predecessor verification (for both workstreams)

We cloned `canada-quant/dsv4-flash-w4a16-fp8` and grep'd
`scripts/quantize_v4_w4a16.py`:

```
AutoModelForCausalLM.from_pretrained(...)   # silently drops mtp.*
compressed_tensors.offload.load_offloaded_model   # HF auto-offload
linearize_moe_model()                        # MoE → standard nn.Linear

ZERO references to _expert_world_size / _expert_rank / patch_moe_for_expert_sharding
```

Predecessor was a different sharding model entirely. Every rank held the
same module set (modules spilled to disk by HF accelerate). NCCL
collectives matched across ranks because the module set was uniform.
Our decoupled shard is novel for both workstreams; the bugs we surface
are bugs that have been latent in llmcompressor for as long as the
module-set-divergence pattern has existed.

---

## What to forward to the integration layer (the user)

When you (the user) read this, the answer for whether to forward this
doc to the B300 agent depends on which point they're at:

- If B300 hasn't started the multi-rank work yet → forward in full;
  they save the most time.
- If B300 has hit the observer-sync hang already → forward Patch A
  + the sharding invariant pattern.
- If B300 has filed an issue already → cross-link our `CONTRIBUTIONS_QUEUE.md`
  C1 entry so the canada-quant brand work consolidates.

## Option Y — MTP stays BF16 by design (2026-05-21)

Before the recipe questions: the artifact's MTP design choice.

**Decision: keep MTP at BF16 even when the main MoE goes to W4A16.**

MTP is the speculative-decoding draft head. Speculative throughput depends on
**token-acceptance-rate** by the verifier. Quantization noise in the draft
directly degrades acceptance, killing the speedup. DeepSeek's native release
leaves MTP at higher precision than the MXFP4 experts; RedHat dropped MTP
entirely. The right move is preserving MTP at full BF16 while quantizing the
main MoE — sharper positioning than either alternative.

| Component | Recipe |
|---|---|
| Main 43 layers, routed experts (256/layer × 3 projections) | W4A16 INT4 group=128 |
| Main 43 layers, attention (q_a/q_b/kv/o_a/o_b + compressor/indexer) | FP8_BLOCK 128×128 |
| Main 43 layers, shared experts | BF16 (passthrough) |
| Main 43 layers, norms / gates / hyper-connection params | BF16 (passthrough) |
| **MTP block (`mtp.0.*`)** | **BF16 (no quantization)** |

Cost: +10 GB on disk (MTP ~13.2 GB BF16 vs ~3.3 GB W4A16 = 7% size
overhead). Benefit: full MTP acceptance rate, expected ~1.8× decode
speedup at `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
30%+ throughput loss from degraded acceptance avoided.

**Recipe-level implementation** (in `scripts/quantize_v4_w4a16_mtp.py`):

```python
ignore=[
    "lm_head",
    r"re:.*mtp\..*",   # MTP draft head preserved at BF16
]
```

This is deliberate, not accidental. The first observed smoke happened to
produce MTP-in-BF16 because the sequential pipeline's per-subgraph module
collection skipped MTP paths — but relying on that was fragile. The
explicit `ignore` regex makes it permanent and load-bearing in the
recipe.

**For the NVFP4 sibling:** the same logic applies. Even more strongly,
since NVFP4 has higher quantization noise than W4A16 → acceptance hit
would be larger. Recommend `ignore=["lm_head", r"re:.*mtp\..*"]` in the
NVFP4 recipe too. The product position becomes:

> "First DSv4-Flash NVFP4 quant that preserves the MTP draft head at
> BF16 — engineered for maximum speculative-decoding throughput on
> Blackwell hardware."

That's a sharper differentiator than "MTP preserved" alone.

---

## Post-Option-D pivot findings (2026-05-20, post-pivot)

After Option A (decoupled-shard) was diagnosed as structurally dead, we
moved to Option D (predecessor's `linearize_moe_model` + auto_offload +
8-rank DDP recipe + our MTP class shim). During the first end-to-end smoke
on Option D, we hit two compounding silent-correctness bugs that the B300
NVFP4 sibling will likely hit identically if its MTP shim uses the same
patterns. Both bugs are filed upstream (`huggingface/transformers#46127`,
`vllm-project/llm-compressor#2735`, `#2739`) and have to be fixed in any
DSv4 MTP-preserving shim.

### Bug N1 — MTP layer_type, NOT "repeat last main layer's type"

**Cause:** the MTP block in DSv4-Flash uses a *simpler* attention than
the main layers — only the standard projections (`wq_a, wq_b, wkv, wo_a,
wo_b, q_norm, kv_norm, attn_sink`), **no `compressor.*` or `indexer.*`
keys**. Inspection of an upstream-format MTP checkpoint confirms this:

```
mtp.0.attn.attn_sink, mtp.0.attn.kv_norm.weight, mtp.0.attn.q_norm.weight,
mtp.0.attn.wkv.weight, mtp.0.attn.wo_a.weight, mtp.0.attn.wo_b.weight,
mtp.0.attn.wq_a.weight, mtp.0.attn.wq_b.weight, mtp.0.attn_norm.weight,
mtp.0.e_proj.weight, mtp.0.enorm.weight, mtp.0.ffn.experts.*, ...
```

The DSv4 config has three attention `layer_types`:

```
Counter({'compressed_sparse_attention': 21,    # full attn + Compressor + Indexer
         'heavily_compressed_attention': 20,   # full attn + Compressor
         'sliding_attention': 2})              # full attn, compressor = None
```

`DeepseekV4Attention.__init__` instantiates the compressor *conditionally*:

```python
self.compressor = (
    COMPRESSOR_CLASSES[self.layer_type](config)
    if self.layer_type != "sliding_attention" else None
)
```

**Wrong shim choice:** copy the last main-layer's `layer_type`
(`compressed_sparse_attention`). MTP's `DeepseekV4Attention` then
instantiates empty `compressor` + `indexer` submodules → checkpoint has
no `mtp.0.attn.compressor.*` keys to fill them → those modules stay
uninitialized → `_initialize_weights` falls through to `_init_weights`
→ `init.normal_` random-initializes the MTP block. Silent corruption of
the MTP draft head; the artifact ships with random MTP weights.

**Correct shim choice:** `layer_types[mtp_idx] = "sliding_attention"`,
`mlp_layer_types[mtp_idx] = "moe"`.

**For the B300 NVFP4 sibling:** if your MTP shim extends `config.layer_types`,
verify you're using `"sliding_attention"` for the MTP layer. If you used
`"compressed_sparse_attention"` or copied the last main-layer's type,
your shipped artifact has random MTP weights regardless of whether the
calibration "succeeded."

### Bug N2 — `conversion_mapping["deepseek_v4"]` doesn't cover `mtp.*` paths

**Cause:** `transformers.conversion_mapping.get_checkpoint_conversion_mapping("deepseek_v4")`
returns 41 `WeightRenaming` entries that map upstream-internal naming to
HF naming (`attn.` → `self_attn.`, `ffn.` → `mlp.`, `attn_norm.` →
`input_layernorm.`, `attn.wq_a.` → `self_attn.q_a_proj.`, etc.). Entries
6–38 are anchored at `^layers\.(\d+)\.` — they only fire on main-layer
keys. The 6 model-level entries (`embed.`, `head.`, `norm.`, `hc_head_*`)
don't apply to MTP either.

**Result:** MTP keys arrive in upstream form (`mtp.0.attn.wq_a.weight`)
after `from_pretrained` finishes the file-read, then fail to match the
HF-named submodules in `model.mtp[0]` (which expect
`mtp.0.self_attn.q_a_proj.weight`). The keys remain "unexpected", the
submodules remain "uninitialized", and `_init_weights` runs again →
silent random-init.

**Workaround:** at runtime, programmatically clone all 33 main-layer
entries to `mtp.\d+.*` equivalents and re-register the combined mapping:

```python
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
existing = get_checkpoint_conversion_mapping("deepseek_v4")
added = []
for entry in existing:
    sp = getattr(entry, "source_patterns", None)
    tp = getattr(entry, "target_patterns", None)
    if sp is None or tp is None:
        continue
    sp_list = sp if isinstance(sp, (list, tuple)) else [sp]
    tp_list = tp if isinstance(tp, (list, tuple)) else [tp]
    new_sp, new_tp = [], []
    for s, t in zip(sp_list, tp_list):
        if isinstance(s, str) and s.startswith(r"^layers\.(\d+)\."):
            new_sp.append(s.replace(r"^layers\.(\d+)\.", r"^mtp\.(\d+)\.", 1))
            new_tp.append(t.replace("layers.\\1.", "mtp.\\1.", 1))
    if new_sp:
        added.append(type(entry)(
            source_patterns=new_sp if len(new_sp) > 1 else new_sp[0],
            target_patterns=new_tp if len(new_tp) > 1 else new_tp[0],
        ))
register_checkpoint_conversion_mapping(
    "deepseek_v4", list(existing) + added, overwrite=True)
```

**Upstream fix:** transformers itself should add the `mtp.\d+.*`
equivalents in the canonical list. The patch is N+33 entries (mirroring
33 of the original 41); the `embed`, `head`, `norm`, `hc_head_*` entries
do *not* mirror because those are model-level and MTP doesn't have its
own copy of them.

**For the B300 NVFP4 sibling:** if your shim relies on the built-in
conversion mapping, the MTP keys won't rename. Same silent random-init.
Apply the mirror extension before any `from_pretrained` call.

### How to detect both bugs in 50 ms — value-verification assertion

The MISSING-keys count is necessary but not sufficient — even a fully
correct module count can hide silent random-init if the wrong layer_type
or missing conversion_mapping causes some submodules to be ignored
during the load. Permanent assertion to catch this regression class:

```python
# Immediately after AutoModelForCausalLM.from_pretrained(...):
if dist.get_rank() == 0:
    import safetensors.torch as st
    from pathlib import Path
    loaded_w = model.mtp[0].self_attn.q_a_proj.weight
    source_w = None
    for shard in sorted(Path(args.input).glob("model-*.safetensors")):
        with st.safe_open(shard, framework="pt") as f:
            if "mtp.0.attn.wq_a.weight" in f.keys():
                source_w = f.get_tensor("mtp.0.attn.wq_a.weight")
                break
    assert source_w is not None, "could not find mtp.0.attn.wq_a.weight in source"
    diff = (loaded_w.cpu().float() - source_w.cpu().float()).abs().max().item()
    assert diff < 1e-4, f"MTP weight mismatch! diff={diff}"
    print(f"[mtp-verify] max_diff: {diff:.2e}  -> MTP weights loaded correctly")
```

Cost: ~50 ms per launch. Benefit: catches the entire class of
silent-MTP-loading bugs (wrong layer_type, missing conversion entries,
dtype mismatch, sliced load). Required permanent fixture in any
DSv4 MTP-preserving calibration script.

---

## Mini-smoke result (2026-05-20, post-run, Option A path — superseded)

Setup: `--samples 8 --batch-size 1 --max-seq-len 128 --dry-run-one-layer`
(restricts recipe to layer 5 only). 8 ranks, p5en.48xlarge H200.

### What worked

1. **Sharding invariant passed cleanly:**
   ```
   [shard-invariant] OK — 43 main MoE layers + 1 MTP layer(s),
   11264 total (layer,expert) tuples, disjoint across 8 ranks
   (per-rank counts: [1408, 1408, 1408, 1408, 1408, 1408, 1408, 1408])
   ```
   MTP is correctly included; experts are disjoint.

2. **All three patches applied:**
   ```
   [patch A: Observer.synchronize] applied (world_size=8)
   [patch B: _reduce_hessian] applied (world_size=8)
   [patch C: _broadcast_quantized_params] applied (world_size=8)
   ```

3. **Calibration completed on subgraph 6/45** (the layer-5 subgraph):
   `(6/45): Calibrating: 100% ...`

4. **Patch B fired correctly:**
   ```
   [patch B] skipped reduce for 48 sharded modules; reducing 4 replicated
   ```
   The `_reduce_hessian_to_target_rank` filtered out the 48 expert-weight
   instances (sharded across ranks) and delegated only the 4 replicated
   attn modules to the original implementation. The reduce on those 4
   completed without NCCL hang. **The disjoint-set NCCL hang the patches
   were designed to prevent did not occur.**

### What broke

After patch B's reduce, all 8 ranks entered `compress_module_list`
(`gptq/base.py:304`) and **all 8 ranks hung indefinitely** (>17 minutes,
killed manually).

Native py-spy backtrace (identical on multiple ranks):

```
cuStreamSynchronize           (libcuda.so.595.71.05)
cudaStreamSynchronize         (libcudart.so.13)
at::native::_local_scalar_dense_cuda_impl<c10::BFloat16>
at::native::_local_scalar_dense_cuda
at::Tensor::item<double>      (libtorch_cpu.so)
__torch_function__            (torch/utils/_device.py:122)
compress_module_list          (llmcompressor/modifiers/gptq/base.py:304)
_patched                      (gptq_checkpoint.py:212)
compress_modules              (llmcompressor/modifiers/gptq/base.py:293)
```

The exact stuck call is `int(num_samples)` on line 304:

```python
303      logger.info(f"Quantizing {name} using {int(num_samples)} samples")
304      with (
```

`num_samples` is a CUDA scalar tensor that was the destination of a
`dist.reduce` call (for the 4 replicated modules where this rank is
`target_rank`). `int(num_samples)` triggers `cudaStreamSynchronize` on
the default stream — which blocks forever.

### Hypothesis (not yet verified)

The vendored `Transformer.forward` (specifically `MoE.forward` under our
decoupled-expert shard) likely issues cross-rank NCCL kernels during
calibration that don't properly drain into the default stream before
`compress_module_list` reads `num_samples`. The patches close the
*explicit* collective-on-disjoint-modules hang, but a deeper
stream-synchronization issue persists in the MoE forward path itself.

Other plausible angles to investigate:

- `module_to_rank` may map some sharded modules to non-owning ranks,
  causing `_orig`'s `dist.reduce` to enqueue on tensors that don't
  exist on those ranks (would corrupt state but not hang per se).
- The wait_for_comms inside `_orig` may not register a CUDA event
  on the default stream — so subsequent `int(...)` reads block
  waiting for an NCCL kernel that nobody else completes.
- `align_module_device(module)` for sharded experts may trigger
  cross-rank work that this rank's view of the model can't satisfy.

### What this means for the NVFP4 sibling

**You probably DON'T see this exact hang** because your
`QuantizationModifier` (RTN) doesn't compute Hessians or invoke
`compress_module_list` with cross-rank coordination. Your equivalent
risk is whatever your RTN path does in `compress_modules` — gate it
the same way: py-spy the moment it stops printing progress, find the
synchronize point, and apply a targeted fix.

**For both workstreams:** the patches in `multirank_patches.py` are
NECESSARY but NOT SUFFICIENT to ship a multi-rank artifact. There's
real stream-synchronization work left in the MoE forward path.

### Status of upstream issue

`vllm-project/llm-compressor#2734` filed before the hang was diagnosed.
The issue body remains accurate on the patches' purpose and the
disjoint-collective hang they prevent, but adding a comment now to
flag the secondary hang as a separate downstream issue (likely needs a
separate PR).

### Next steps before Phase 2 launch

1. Investigate `MoE.forward` under decoupled shard for unbalanced NCCL
   work (one rank queues a collective the others don't match).
2. Test inserting `torch.cuda.synchronize()` + `dist.barrier()` between
   the calibration loop and `compress_module_list` entry to force
   stream drain.
3. Verify `module_to_rank` consistency across ranks.
4. If 2 doesn't fix it, study the NCCL stream's pending work via
   `nsys`/`nvprof`.

Phase 2 full launch is BLOCKED on this. Mandatory check-in with user
before continuing.

---

## Update 2026-05-21 — pivot to H200, smoke iter 7 + iter 8

After the diagnostics above, we pivoted off B300 to a single
`p5en.48xlarge` (8× H200) box in us-east-2. Hardware family matches
the predecessor's, so the secondary hang in `compress_module_list`
disappeared once we ran the predecessor's HF auto-offload pattern
instead of the decoupled shard. **Net: the `multirank_patches.py`
file is no longer the long pole — the H200 path uses replicated
experts via `from_pretrained(device_map="auto")` and the GPTQ
collectives don't have the disjoint-module problem.**

This whole class of issues evaporates on B300 too if the sibling can
get HF auto-offload to fit in DDR (predecessor needs ~2.5TB DDR, and
B300 has the same). Worth re-checking before the sibling spends more
time on `multirank_patches.py`.

### Bug N4 — recipe targets= must match vLLM's INTERNAL naming, not HF

Found while running Option B serve smoke on the iter 8 artifact. Symptom:
vLLM raises `torch.OutOfMemoryError: Tried to allocate 4.00 GiB. GPU has
1.05 GiB free` during `make_layers` (line 646 of
`vllm/model_executor/models/utils.py`). At the time of the OOM the model
weights aren't even loaded yet — we're still in `__init__` allocating
empty tensors.

Root cause: vLLM's compressed-tensors scheme resolution in
`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:get_moe_method`:

```python
unfused_names = [
    layer_name + proj_name
    for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
]
all_scheme_dicts = [
    quant_config.get_scheme_dict(layer, name) for name in unfused_names
]
scheme_dict = all_scheme_dicts.pop()
...
if scheme_dict is None:  # ignored layer
    return UnquantizedFusedMoEMethod(layer.moe_config)
```

vLLM probes `layer_name + ".0.gate_proj"` against `targets=`. Our
recipe wrote targets with HF-renamed paths
(`model.layers.X.mlp.experts.Y.gate_proj`), but `vllm.models.deepseek_v4`
uses UPSTREAM naming internally:

- `prefix=f"{prefix}.ffn"` (model.py:1012) → MoE layer is
  `model.layers.X.ffn` (NOT `mlp`)
- `prefix=f"{prefix}.experts"` (model.py:751) → experts get
  `model.layers.X.ffn.experts`
- `prefix=f"{prefix}.attn"` (model.py:1008) → attention is
  `model.layers.X.attn` (NOT `self_attn`)

So our `re:.*mlp\.experts\.\d+\.(gate_proj|...)$` doesn't match
`model.layers.0.ffn.experts.0.w1` → unquantized fallback →
~80GB per GPU for what should have been ~20GB W4A16 → OOM.

**Fix:** rewrite targets at postprocess time to predecessor's broad
pattern that matches both naming conventions:

```python
group_0: r"re:.*attn\.(wq_a|wq_b|wkv|wo_a|wo_b|fused_wqa_wkv|q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$"
group_1: r"re:.*experts\.\d+\.(w1|w2|w3|gate_proj|up_proj|down_proj|gate_up_proj)$"
```

**Critical for NVFP4 sibling:** your recipe also writes HF names in
targets= (whatever your equivalent is). Without a postprocess rewrite,
your saved artifact will ALSO fall back to UnquantizedFusedMoEMethod at
serve. You will see the same OOM signature. Apply the same broad pattern
in your postprocess (NVFP4 group_0 also needs both naming).

Detection at the saved artifact: run `vllm serve ... --max-model-len
1024` (small) and watch for the OOM. If it fires with "Tried to allocate
4 GiB" while >130 GiB is "in use" before any weights have loaded, you
have this bug. Fix at the config level, no recalibration needed.

The fix is now landed in `scripts/postprocess_for_vllm.py:patch_config`
in commit `45e580e`.

### Bug N3 — `ignore=` honored at GPTQ calibration but NOT at save

The new one. Filed as
[`vllm-project/compressed-tensors#712`](https://github.com/vllm-project/compressed-tensors/issues/712)
on 2026-05-21.

**Symptom:** recipe has `ignore=[r"re:.*mtp\..*"]`. GPTQ calibration
correctly excludes MTP from Hessian construction (subgraph 43, the
MTP one, is empty — 0 modules processed). But the saved artifact has
`model.mtp.0.mlp.experts.*.weight_packed` (W4A16-quantized) anyway.
The save path's RTN-style compression in
`llmcompressor/transformers/compression/compressed_tensors_utils.py`
(the `Compressing model: 0/N` progress bar) consults only `targets=`,
NOT `ignore=`.

**Workaround for both workstreams:** anchor `targets=` so MTP paths
don't match in the first place. Don't rely on `ignore=` alone:

```python
# WRONG — MTP gets quantized at save despite ignore
targets=[r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]
ignore=[r"re:.*mtp\..*"]

# RIGHT — MTP paths can't match the targets pattern
targets=[r"re:^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]
ignore=[r"re:.*mtp\..*"]  # belt-and-suspenders, kept for visibility
```

The leading `^model\.layers\.\d+\.` anchor is what does the work.
`re:.*` matches `model.mtp.0.mlp.experts.*` too — anchoring forbids it.

**Critical for NVFP4 sibling:** your recipe almost certainly has the
same shape — `targets=re:.*mlp\.experts\.*` with `ignore=mtp.*` — so
your saved artifact will silently ship a quantized MTP block too,
and your speculative-decoding acceptance rate will be ruined. **Audit
your `targets=` regex before save, or you'll need to rerun calibration.**

How to detect post-hoc on a saved artifact:

```python
import json
with open("model.safetensors.index.json") as f:
    wm = json.load(f)["weight_map"]
mtp_quantized = [k for k in wm if "mtp" in k and "weight_packed" in k]
mtp_quantized_scales = [k for k in wm if "mtp" in k and ".weight_scale" in k]
assert not mtp_quantized, f"MTP got quantized at save: {len(mtp_quantized)} packed weights"
assert not mtp_quantized_scales, f"MTP got quantized at save: {len(mtp_quantized_scales)} scales"
```

If both lists are empty, MTP shipped clean. If either has entries,
the recipe's `targets=` matched MTP paths — fix the regex and rerun.

### Status of smoke iter 8

Currently in flight on H200 (PID 143471). If it produces an artifact
where `mtp_quantized == []`, that confirms the targets=-anchor fix
works and we go to Phase 2 full calibration. ETA ~3h from 2026-05-21
06:35 UTC.

---

## Option Y design rationale (for both repos' model cards)

Whether the artifact is W4A16+FP8_BLOCK or NVFP4+FP8_BLOCK, the **MTP
block stays at BF16**. The reasoning is identical:

- MTP is the speculative-decoding draft head.
- Decode throughput speedup from MTP depends on token-acceptance-rate
  by the main verifier — quantization noise in the draft destroys
  acceptance and kills the speedup.
- The MTP block is small (~13 GB at BF16 vs ~3.3 GB at W4A16 ≈ 7%
  artifact overhead) but its quality impact on the speedup factor
  (1.8x at `num_speculative_tokens=2`) is disproportionate.
- DeepSeek's own native release also leaves MTP at higher precision
  than the main routed-expert tensors.

So `ignore=[..., r"re:.*mtp\..*"]` is the design principle for both
this repo and the sibling, but it has to actually fire at save time —
hence Bug N3 matters.

## Update 2026-05-22 — two new bugs surfaced debugging 0% MTP acceptance

### C13 — `transformers.save_pretrained` silently downcasts FP32 to BF16

**Symptom:** the DeepSeek-V4-Flash release spec keeps several tensor
groups at FP32 for numerical precision:

- `hc_attn_{base,fn,scale}` per layer (hyper-connection plumbing)
- `hc_ffn_{base,fn,scale}` per layer
- `hc_head_{base,fn,scale}` (global + MTP)
- `*.attn.attn_sink`
- `*.ffn.gate.bias` (formerly `e_score_correction_bias`)
- `*.attn.compressor.ape` (formerly `position_bias`)
- `*.attn.indexer.compressor.ape` (formerly `position_bias`)

`transformers.save_pretrained` (5.8.1) writes them as BF16 if the
PreTrainedModel's `torch_dtype` is BF16 — even though the source
checkpoint had them FP32. This is silent — no warning, no error.

**Evidence:** loaded source `/scratch/weights/bf16-mtp` via safetensors,
read dtypes — 417 tensors were FP32 in source. After
`save_pretrained(torch_dtype=torch.bfloat16)`, all 417 are BF16 in
output.

**Impact:** numerical precision loss on the HC paths and on the LM head
gating math. The sibling's published artifact also has all these
restored to FP32 in postprocess, so they ran into this too — the
restore step is necessary.

**Workaround:** postprocess script reads source's FP32 tensors back and
writes them over the BF16 versions in the artifact shards. The
predicate must be applied AFTER all renames (the original
`e_score_correction_bias → bias` and `position_bias → ape` renames mean
the predicate has to match POST-rename names). One predicate-after-
rename pass catches all 417; predicate-before-rename misses 103.

**Status:** carried as postprocess (see `/tmp/fixup_artifact.py` and the
predecessor's postprocess). Not yet filed upstream against transformers.

### C14 — vLLM MTP loader silently skips top-level head/embed

**Symptom:** with our artifact containing top-level `head.weight` +
`embed.weight` and NO `mtp.0.head.weight` / `mtp.0.emb.tok_emb.weight`,
vLLM's MTP load proceeds cleanly (no error), produces draft tokens,
and gets 0% acceptance.

**Root cause:**
`vllm/model_executor/models/deepseek_v4_mtp.py::DeepSeekV4MTP.load_weights`
contains:

```python
for name, loaded_weight in weights:
    name = name.replace("mtp.0.", ...)   # no-op on top-level keys
    spec_layer = get_spec_layer_idx(name)
    if spec_layer is None:
        continue          # <- top-level head.weight + embed.weight die here
    ...
```

Result: the MTP layer's `shared_head.head` (ParallelLMHead) and
`embed_tokens` (VocabParallelEmbedding) stay uninitialized → garbage
logits → 100% rejection by the verifier model.

**Evidence:**
- Sibling artifact `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` (works
  at >0.5 acceptance per public README) has 799 `mtp.*` keys; ours had
  797. The 2 extras are full duplicates of the top-level head and
  embed.
- Sibling artifact also UPCASTS `head.weight` from BF16 (DeepSeek
  release) to FP32 in both the top-level and the MTP-alias copy. This
  is presumably to preserve logit precision on the draft head.
- vLLM has the same load path for both this repo (W4A16) and the
  sibling (NVFP4) — both rely on the duplicates being on disk.

**Impact:** any DSv4-Flash artifact built without explicit MTP
head/embed duplicates will produce 0% acceptance silently. There's no
load-time error to catch it; the only signal is acceptance metrics.

**Workaround:** postprocess script that injects:
- `mtp.0.head.weight` = FP32 copy of `head.weight` (also upcast top-level
  to FP32 for precision)
- `mtp.0.emb.tok_emb.weight` = BF16 copy of `embed.weight`

into the shard holding `mtp.0.*` keys and updates the safetensors index.
Cost ~+4 GB on a 165 GB artifact.

**Status:** carried as postprocess. Worth filing against vLLM — either
the loader should explicitly handle top-level head/embed for the MTP
slot, or it should raise on an MTP layer with uninitialized
shared_head.head. Silent 0%-acceptance is the worst possible failure
mode for this. File reference: `vllm/model_executor/models/deepseek_v4_mtp.py`.

### Sibling-specific note

The sibling NVFP4-FP8-MTP repo presumably either:
(a) hit C14 and added the duplicate-injection step, or
(b) inherited it from the predecessor's postprocess that did the same.

Either way, the duplicate-injection pattern is part of the sibling's
working postprocess. The W4A16 sibling (this repo) needs the same. Any
*future* DSv4-Flash quant — by anyone — needs the same. This is a
recipe-level invariant, not a per-quantization choice.

### C15 — DeepGemm `paged_mqa_logits` kernel asserts on `next_n > 2`, capping `num_speculative_tokens` at 1

**Symptom:** vLLM serve with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
crashes during `profile_cudagraph_memory` with:

```
RuntimeError: Assertion error
  (.../deepgemm-src/csrc/apis/../jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp:233):
  next_n == 1 or next_n == 2
```

The assertion fires before /health ever returns. `num_speculative_tokens=1`
works fine. Confirmed reproducible on:
- vLLM build `~/src/vllm` HEAD `50d9dd902` (cherry-pick PRs #43248+#43288+#43290+#43319)
- H200 (Hopper, sm_90a)
- Phase 2 W4A16+FP8+MTP artifact

**Root cause (diagnosis):** vLLM passes `next_n = num_speculative_tokens + 1`
into the DeepGemm `smxx_fp8_fp4_paged_mqa_logits` kernel (k draft tokens + 1
main verifier token in the lookahead window). The assertion `next_n == 1 or
next_n == 2` therefore enforces `num_speculative_tokens <= 1`. With `k=1`,
next_n=2 (passes). With `k=2`, next_n=3 (fails). The error message is
misleading — it sounds like the kernel allows 1 or 2 spec tokens, but it
actually allows 1 or 2 total lookahead positions.

Other DeepGemm assertions confirm the hard-coded ceiling:
- `attention.hpp:210`: `arch_major == 10 and next_n == 1 and (block_kv == 64 or block_kv == 32)`
- `attention.hpp:338`: `arch_major == 10 and next_n == 1`

These are Blackwell-only paths (`arch_major == 10`). The H200 (Hopper,
`arch_major == 9`) falls into the `paged_mqa_logits` path which accepts
`next_n <= 2`.

**Workaround attempts:**
- `--attention-backend FLASHINFER_MLA_SPARSE`: same kernel still fires
  (the paged_mqa_logits kernel is logits-side, not attention-backend-specific).
- `--enforce-eager`: not yet tried (disables cudagraphs entirely, which is
  the launch-throughput-killing trade we don't want for production).

**Impact:** on H200, the practical MTP `num_speculative_tokens` ceiling is 1.
Theoretical k=1 speedup at 89% acceptance is ~1.49× (matches what we
measured at bs=1 → 6.02ms TPOT vs 8.93ms without MTP). With k=2 unlocked,
theoretical speedup would be ~1.9-2.0× (more lookahead per verifier pass).

**Status:** filed as C15 here (2026-05-22). Not yet pushed upstream to
DeepGemm / vLLM. **Fix proposal:** widen the assertion in
`smxx_fp8_fp4_paged_mqa_logits.hpp` to allow next_n up to 4 (or whatever
the kernel actually supports), OR document the
`num_speculative_tokens <= 1` constraint clearly in vLLM's spec-decode
config validation so users get a clean error before the cudagraph
capture step blows up.

**Why this matters for the launch story:** our published throughput number
(1.49× bs=1 with k=1) is real and reproducible, but it leaves ~25% of
the MTP speedup on the table compared to what k=2 would deliver. Worth
calling out as "current vLLM build limits us to k=1; with k=2 unlocked
expected speedup is ~1.9×" if the model card discusses throughput.
