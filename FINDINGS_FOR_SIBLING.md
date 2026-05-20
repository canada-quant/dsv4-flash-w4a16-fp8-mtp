# Findings for the B300 NVFP4 sibling

Cross-pollination document from `canada-quant/dsv4-flash-w4a16-fp8-mtp`
(H200 W4A16+MTP path) to `canada-quant/dsv4-flash-nvfp4-fp8-mtp` (B300
NVFP4+MTP path). Both workstreams use `llmcompressor` on the same
`DeepSeek-V4-Flash` architecture with the same decoupled MoE expert
shard, so most of the diagnostic work transfers — but the two recipes
hit different code paths in `llmcompressor`, so the *which patch where*
question matters.

**Date:** 2026-05-20.
**Author:** H200 agent (this repo).
**Status:** mini-GPTQ smoke COMPLETED but with a SECONDARY HANG inside
`compress_module_list` (not in our patches). Phase 2 launch is BLOCKED on
diagnosing this second deadlock. See "Mini-smoke result" section below.

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
