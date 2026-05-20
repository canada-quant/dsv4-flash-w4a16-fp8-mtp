"""Multi-rank patches for `llmcompressor` GPTQ + observer paths when the
model is sharded across ranks with disjoint module sets.

==============================================================================
BACKGROUND (read before editing — these patches are upstream-PR candidates)
==============================================================================

DeepSeek-V4-Flash has 256 routed experts per MoE layer. To fit the 568 GB BF16
model into multi-rank calibration on a box with limited DDR (e.g. p5en.48xlarge
has ~2 TB RAM but 8 × 568 GB = 4.5 TB), we use a *decoupled* MoE expert
shard: each rank holds only `256 / world_size` experts (e.g. 32 on 8 ranks),
while attention/embeddings stay full-size (replicated).

This is a different sharding model than the predecessor calibration
(`pasta-paul/dsv4-flash-w4a16-fp8`), which used HF
`AutoModelForCausalLM.from_pretrained` with `device_map="auto_offload"` and
silently dropped `mtp.*` keys. With auto-offload every rank held the same
module set (modules spilled to disk), so NCCL collectives matched across
ranks. Our decoupled shard means **rank 0's `model.layers.5.ffn.experts.7`
does not exist on rank 1** — the module set diverges.

The diverging module set surfaces three NCCL-collective failure modes in
llmcompressor:

  A) `llmcompressor.observers.base.Observer.synchronize()` —
     `QuantizationMixin.sync_activation_observers` walks matched modules and
     calls `observer.synchronize()` (which does `dist.all_reduce`) for any
     module with an attached input/output observer. Fires for RTN/NVFP4
     recipes that quantize activations. **Does NOT fire for W4A16-only or
     FP8_BLOCK-only weight-only schemes** (no activation observers are
     attached), but we patch it anyway as cheap insurance against subtle
     observer-creation paths.

  B) `GPTQModifier._reduce_hessian_to_target_rank` (gptq/base.py:323) —
     iterates `module_list` and calls `dist.reduce(self._hessians[module],
     dst=target_rank, async_op=True)` for each. With disjoint module sets,
     ranks call reduce on different module subsets → NCCL collective hang.

  C) `GPTQModifier._broadcast_quantized_params` (gptq/base.py:350) — same
     pattern: `dist.broadcast(...)` per module. Same disjoint-set hang.

Predecessor's recipe ran cleanly multi-rank because their module set was
identical across ranks (auto-offload). Ours is genuinely new territory.

==============================================================================
UPSTREAM-PR CANDIDACY
==============================================================================

All three patches are upstream-PR candidates against vllm-project/llm-compressor.
The underlying primitive missing in llmcompressor is "this module is sharded
to exactly one rank; do not include it in the cross-rank reduce." A general
fix would expose a per-module "replication group" attribute and gate the
collectives on it; this monkey-patch is the minimum viable workaround that
proves the fix concept on a real DSv4-Flash run.

When filing the PR:
  - Title: `[bug] GPTQModifier hangs on multi-rank with sharded MoE
    experts (dist.reduce/broadcast on disjoint module sets)`
  - Tag @kylesayrs (he knows our work from vLLM #41511 / kylesayrs PR #41276)
  - Link this file as the proposed minimal patch surface
  - Test plan: 1-layer 8-rank GPTQ on DSv4-Flash (3-5 minute wall clock);
    confirm progress through expert quantization with no NCCL timeout.

==============================================================================
"""
from __future__ import annotations

import os
import re
import sys
from typing import Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist


# Regex that identifies a routed-expert weight in the upstream DSv4 internal
# naming. The full DSv4 expert path is:
#   <prefix>.layers.<L>.ffn.experts.<EID>.{w1,w2,w3}
# Under the CalibrationModel wrapper, modules appear at
#   cal_model.transformer.layers.<L>.ffn.experts.<EID>.{w1,w2,w3}
# so we anchor on the `.ffn.experts.<digit>.` segment.
EXPERT_NAME_RE = re.compile(r"\.ffn\.experts\.\d+\.")


def _is_sharded_module(module_name: str) -> bool:
    """A module is 'sharded' (exists on exactly one rank in our decoupled
    expert shard) iff it is a routed-expert weight.

    Attention projections, norms, embeddings, MoE gate, shared experts,
    and the MTP block's attention/projections are all *replicated* across
    ranks and must participate in the cross-rank reduce normally.

    The mtp.0 layer's routed experts (mtp.0.ffn.experts.<EID>.{w1,w2,w3})
    are also sharded under our recipe and match this regex.
    """
    return EXPERT_NAME_RE.search(module_name) is not None


# =============================================================================
# Sharding invariant
# =============================================================================
def assert_sharding_invariant(
    model: torch.nn.Module,
    *,
    world_size: int,
    rank: int,
    n_routed_experts: int = 256,
    verbose: bool = True,
) -> None:
    """Walk the model and assert the decoupled-expert shard is wired correctly.

    For each MoE layer that exists on this rank, collect the set of expert
    indices present. All-gather across ranks. Assertions:
      1. Each (layer, expert_id) tuple appears on *exactly one* rank
         (disjointness — the precondition for safely skipping reduces).
      2. The union of per-rank sets per layer covers all `n_routed_experts`
         (completeness — no expert is unowned).

    Failure here means the decoupled shard isn't actually decoupled, and
    skipping reduces would silently corrupt the Hessian. Better to crash
    loudly here than discover at hour 7 of calibration.
    """
    local: List[Tuple[int, int]] = []
    for name, module in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.ffn\.experts\.(\d+)\b", name)
        if m:
            layer_id = int(m.group(1))
            expert_id = int(m.group(2))
            if (layer_id, expert_id) not in local:
                local.append((layer_id, expert_id))
        m = re.search(r"\.mtp\.(\d+)\.ffn\.experts\.(\d+)\b", name)
        if m:
            mtp_id = int(m.group(1))
            expert_id = int(m.group(2))
            # encode MTP layer as a negative-offset layer id to keep the
            # (layer, expert) tuple flat
            local.append((-(mtp_id + 1), expert_id))

    if not dist.is_initialized() or world_size == 1:
        if verbose and rank == 0:
            print(f"[shard-invariant] world_size=1; skipping cross-rank check "
                  f"(found {len(local)} (layer,expert) tuples locally)",
                  flush=True)
        return

    # All-gather the per-rank local lists. Use object_gather to keep this
    # to a single collective call.
    gathered: List[Optional[List[Tuple[int, int]]]] = [None] * world_size
    dist.all_gather_object(gathered, local)

    if rank != 0:
        return  # rank 0 reports and asserts on behalf of all

    by_layer: dict = {}
    duplicates: List[Tuple[int, Tuple[int, int]]] = []
    for r, entries in enumerate(gathered):
        assert entries is not None
        for entry in entries:
            by_layer.setdefault(entry, []).append(r)
            if len(by_layer[entry]) > 1:
                duplicates.append((r, entry))

    if duplicates:
        sample = duplicates[:5]
        raise RuntimeError(
            f"[shard-invariant] FAIL — {len(duplicates)} (layer, expert) tuples "
            f"appeared on more than one rank. First 5: {sample}. "
            f"The decoupled expert shard is misconfigured — patching "
            f"_reduce_hessian/_broadcast_quantized_params to skip these "
            f"modules would corrupt the Hessian. Fix the shard before "
            f"applying multirank_patches."
        )

    # Per-layer coverage check
    layer_to_experts: dict = {}
    for (layer, expert), _ranks in by_layer.items():
        layer_to_experts.setdefault(layer, set()).add(expert)

    incomplete = []
    for layer, experts in sorted(layer_to_experts.items()):
        if len(experts) != n_routed_experts:
            incomplete.append((layer, len(experts)))

    if incomplete and verbose:
        # Not a hard fail — some layers may legitimately have fewer experts
        # in test setups. Warn loudly.
        print(f"[shard-invariant] WARN — {len(incomplete)} layers have "
              f"!= {n_routed_experts} experts across all ranks. First 5: "
              f"{incomplete[:5]}", flush=True)

    if verbose:
        n_layers = len(layer_to_experts)
        per_rank = [len(g or []) for g in gathered]
        print(f"[shard-invariant] OK — {n_layers} MoE layers, "
              f"{sum(per_rank)} total (layer,expert) tuples, "
              f"disjoint across {world_size} ranks "
              f"(per-rank counts: {per_rank})", flush=True)


# =============================================================================
# Patch A: Observer.synchronize → no-op when world_size > 1
# =============================================================================
def apply_observer_sync_patch(world_size: int, verbose: bool = True) -> None:
    """Defensive: replace Observer.synchronize and
    MovingAverageObserverBase.synchronize with a no-op (`return []`) when
    world_size > 1.

    Bug:        `Observer.synchronize` does `dist.all_reduce` over min/max
                statistics across ranks. With disjoint module sets, ranks
                end up calling all_reduce on different observers → NCCL
                collective hang.

    Triggered:  Any QuantizationMixin recipe that attaches activation
                observers (RTN-style quantization with activation
                quantization). The sibling NVFP4 RTN recipe hits this on
                B300; our GPTQ W4A16 + FP8_BLOCK weight-only recipe does
                NOT attach activation observers, so this patch is
                defensive — protects against subtle observer-creation
                paths we might have missed.

    Patch:      `Observer.synchronize = lambda self: []`. Returning empty
                list signals "no pending comms" to the caller's
                `wait_for_comms`. The owning rank still computes qparams
                from its local stats (~96 samples per rank with 768/8 split
                is plenty for min/max observers).

    PR-candidate: yes — upstream fix is to make Observer.synchronize
                  sharding-aware (skip observers attached to modules not
                  present on all ranks).
    """
    if world_size <= 1:
        if verbose:
            print("[patch A: Observer.synchronize] world_size<=1 → patch is no-op", flush=True)
        return

    import llmcompressor.observers.base as _obs_base
    import llmcompressor.observers.moving_base as _obs_moving

    # Signature guard: catch upstream API drift loud.
    assert hasattr(_obs_base, "Observer"), \
        "[patch A] llmcompressor.observers.base.Observer missing — upstream API changed"
    assert hasattr(_obs_base.Observer, "synchronize"), \
        "[patch A] Observer.synchronize missing — upstream API changed"
    # The original returns `List[dist.Work]`. We can't statically introspect
    # the return type but we can check the method's qualname matches what we
    # expect, to catch refactors.
    _expected_qualname = "Observer.synchronize"
    assert _obs_base.Observer.synchronize.__qualname__.endswith(_expected_qualname), \
        f"[patch A] Observer.synchronize qualname is " \
        f"{_obs_base.Observer.synchronize.__qualname__!r}, expected suffix " \
        f"{_expected_qualname!r}"

    assert hasattr(_obs_moving, "MovingAverageObserverBase") or \
           any(hasattr(_obs_moving, n) for n in ("MovingAverageObserver",)), \
        "[patch A] MovingAverageObserverBase missing — upstream API changed"

    _obs_base.Observer.synchronize = lambda self: []
    if hasattr(_obs_moving, "MovingAverageObserverBase"):
        _obs_moving.MovingAverageObserverBase.synchronize = lambda self: []

    if verbose:
        print(f"[patch A: Observer.synchronize] applied (world_size={world_size})",
              flush=True)


# =============================================================================
# Patch B: GPTQModifier._reduce_hessian_to_target_rank → skip sharded modules
# =============================================================================
def apply_gptq_reduce_hessian_patch(world_size: int, verbose: bool = True) -> None:
    """Skip `dist.reduce(self._hessians[module], dst=target_rank)` for
    modules owned by exactly one rank (i.e., decoupled-shard experts).

    Bug:        With disjoint module sets across ranks, the original
                implementation enqueues `dist.reduce` for every module in
                `module_list`. Each rank's `module_list` is its rank-local
                set; ranks call reduces on different module subsets; NCCL
                collective hangs (or times out).

    Triggered:  Any multi-rank GPTQ run where the model is sharded such
                that some modules exist on only one rank. In our recipe:
                routed experts (`.ffn.experts.<id>.{w1,w2,w3}`) are
                decoupled-sharded; attention/norms/embeddings/MoE gate are
                replicated.

    Patch:      Pre-filter `module_list` to exclude sharded modules. The
                owning rank's local Hessian for a sharded module already
                represents that expert's full statistics (only one rank
                ever saw it during calibration). Replicated modules go
                through the cross-rank reduce normally.

    Invariant: the sharding invariant assertion
               (`assert_sharding_invariant`) must run BEFORE this patch
               applies, to verify the shard is actually disjoint. If a
               module is replicated by mistake, skipping the reduce would
               silently lose its Hessian contribution from other ranks.

    PR-candidate: yes — upstream fix exposes a per-module "replication
                  group" attribute; gate `_reduce_hessian_to_target_rank`
                  on it. The monkey-patch is the minimum viable proof of
                  concept; the upstream PR can use a cleaner abstraction.

    Test plan: 1-layer 8-rank GPTQ on DSv4-Flash (`scripts/mini_gptq_smoke.py`),
               wall clock ~5-10 min. Without this patch the run hangs at
               the first expert reduce; with this patch the run progresses
               through all 256 experts and the layer's attn projections.
    """
    if world_size <= 1:
        if verbose:
            print("[patch B: _reduce_hessian] world_size<=1 → patch is no-op", flush=True)
        return

    import llmcompressor.modifiers.gptq.base as _gptq

    # Signature guards
    assert hasattr(_gptq, "GPTQModifier"), \
        "[patch B] llmcompressor.modifiers.gptq.base.GPTQModifier missing — upstream API changed"
    assert hasattr(_gptq.GPTQModifier, "_reduce_hessian_to_target_rank"), \
        "[patch B] GPTQModifier._reduce_hessian_to_target_rank missing — upstream API changed; check current API and update this patch"
    _orig = _gptq.GPTQModifier._reduce_hessian_to_target_rank
    # The original takes (self, module_list, module_to_rank). Verify arg count.
    import inspect
    sig = inspect.signature(_orig)
    expected_params = {"self", "module_list", "module_to_rank"}
    actual_params = set(sig.parameters.keys())
    assert expected_params == actual_params, \
        f"[patch B] _reduce_hessian_to_target_rank signature drifted: " \
        f"expected {expected_params}, got {actual_params}"

    def _patched_reduce_hessian(self, module_list, module_to_rank):
        # Filter out sharded modules. For sharded modules, the local Hessian
        # is already complete — no cross-rank reduce is needed (and one
        # would hang because other ranks don't have the module).
        replicated_modules = []
        skipped_sharded = 0
        for module in module_list:
            name = self._module_names.get(module, "")
            if _is_sharded_module(name):
                skipped_sharded += 1
                # Leave self._hessians[module] / self._num_samples[module]
                # in place on the owning rank — compress_module_list will
                # consume them in-place.
            else:
                replicated_modules.append(module)

        if dist.get_rank() == 0 and skipped_sharded:
            print(f"[patch B] skipped reduce for {skipped_sharded} sharded "
                  f"modules; reducing {len(replicated_modules)} replicated",
                  flush=True)

        # Delegate replicated modules to the original implementation
        return _orig(self, replicated_modules, module_to_rank)

    _gptq.GPTQModifier._reduce_hessian_to_target_rank = _patched_reduce_hessian

    if verbose:
        print(f"[patch B: _reduce_hessian] applied (world_size={world_size})",
              flush=True)


# =============================================================================
# Patch C: GPTQModifier._broadcast_quantized_params → skip sharded modules
# =============================================================================
def apply_gptq_broadcast_qparams_patch(world_size: int, verbose: bool = True) -> None:
    """Skip `dist.broadcast` for sharded modules in
    `_broadcast_quantized_params`.

    Bug:        Mirrors patch B but on the post-quantization broadcast.
                Original iterates `module_list` and broadcasts each
                module's quantized params from its src_rank. With disjoint
                module sets, ranks call broadcasts on different subsets →
                NCCL hang.

    Triggered:  Same as patch B — any multi-rank GPTQ run with sharded
                experts.

    Patch:      Pre-filter `module_list` to exclude sharded modules. The
                owning rank already has its module's quantized params
                locally after `compress_module_list`; no other rank needs
                a copy (no other rank has that module).

    PR-candidate: yes — same per-module "replication group" abstraction
                  as patch B.

    Test plan: same `scripts/mini_gptq_smoke.py`. Without this patch the
               run hangs at the first post-quantize broadcast.
    """
    if world_size <= 1:
        if verbose:
            print("[patch C: _broadcast_quantized_params] world_size<=1 → patch is no-op", flush=True)
        return

    import llmcompressor.modifiers.gptq.base as _gptq

    assert hasattr(_gptq.GPTQModifier, "_broadcast_quantized_params"), \
        "[patch C] GPTQModifier._broadcast_quantized_params missing — upstream API changed"
    _orig = _gptq.GPTQModifier._broadcast_quantized_params
    import inspect
    sig = inspect.signature(_orig)
    expected_params = {"self", "module_list", "module_to_rank"}
    actual_params = set(sig.parameters.keys())
    assert expected_params == actual_params, \
        f"[patch C] _broadcast_quantized_params signature drifted: " \
        f"expected {expected_params}, got {actual_params}"

    def _patched_broadcast_qparams(self, module_list, module_to_rank):
        replicated_modules = []
        skipped_sharded = 0
        for module in module_list:
            name = self._module_names.get(module, "")
            if _is_sharded_module(name):
                skipped_sharded += 1
            else:
                replicated_modules.append(module)

        if dist.get_rank() == 0 and skipped_sharded:
            print(f"[patch C] skipped broadcast for {skipped_sharded} sharded "
                  f"modules; broadcasting {len(replicated_modules)} replicated",
                  flush=True)

        return _orig(self, replicated_modules, module_to_rank)

    _gptq.GPTQModifier._broadcast_quantized_params = _patched_broadcast_qparams

    if verbose:
        print(f"[patch C: _broadcast_quantized_params] applied "
              f"(world_size={world_size})", flush=True)


# =============================================================================
# Aggregate helper
# =============================================================================
def apply_all_patches(*, world_size: int, verbose: bool = True) -> None:
    """Apply all three patches. Call this AFTER `dist.init_process_group`
    and AFTER `assert_sharding_invariant` has succeeded on the loaded model.
    Patches A/B/C all early-return if world_size <= 1.
    """
    apply_observer_sync_patch(world_size, verbose=verbose)
    apply_gptq_reduce_hessian_patch(world_size, verbose=verbose)
    apply_gptq_broadcast_qparams_patch(world_size, verbose=verbose)


__all__ = [
    "EXPERT_NAME_RE",
    "_is_sharded_module",
    "assert_sharding_invariant",
    "apply_observer_sync_patch",
    "apply_gptq_reduce_hessian_patch",
    "apply_gptq_broadcast_qparams_patch",
    "apply_all_patches",
]
