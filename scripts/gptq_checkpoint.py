"""Per-subgraph GPTQ checkpointing for long-running calibration.

A 10-14h DSv4-Flash GPTQ run on 8 ranks has too much exposure to transient
faults (NCCL flake, OOM on a single rank, accidental kill) to launch without
some form of incremental progress preservation. This module patches
`GPTQModifier.compress_module_list` to dump the just-quantized modules'
state_dicts to disk after each subgraph (layer) completes.

Resume granularity: per-subgraph. If a crash happens partway through
subgraph N, on the next run subgraphs 0..N-1 are restored from checkpoint
and subgraph N restarts from scratch (its Hessian is rebuilt during the
forward passes on subgraphs 0..N-1, which run normally — no GPTQ work
because they were already quantized).

Atomic writes: `torch.save` to `<dst>.tmp`, `fsync`, `os.rename(<dst>.tmp,
<dst>)`. POSIX rename is atomic, so even a spot reclaim or `kill -9`
mid-write leaves either the old file (or nothing) plus a `.tmp` that can
be cleaned up on next start.

Resume path: walk the checkpoint dir, try `torch.load` on each file, mark
any that fail (truncated, corrupt) as missing. Pass the set of validated
completed-subgraph indices to the GPTQModifier; we DO NOT skip data flow
through them (Hessians for later layers still need their forward activations)
— we only skip the GPTQ compress step.

Limitations (deliberate):
  - We don't checkpoint mid-subgraph (the Hessian state for an in-progress
    subgraph). Subgraph granularity is sufficient because the longest
    subgraph in V4-Flash takes ~30-45 min, well under acceptable replay.
  - The checkpoint only includes the modules' state_dicts that were just
    quantized — not the rest of the model. Use `restore_checkpoints_into`
    at load time to splice them back in before calibration starts.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist


_CKPT_DIR_ENV = "GPTQ_CKPT_DIR"
_DEFAULT_CKPT_DIR = "/scratch/weights/checkpoints"


def _resolve_ckpt_dir(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get(_CKPT_DIR_ENV, _DEFAULT_CKPT_DIR))


def _atomic_torch_save(obj, dst: Path) -> None:
    """Write `obj` to `dst` atomically: torch.save to .tmp, fsync, rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    # torch.save accepts a file-like; use one we can fsync.
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, dst)


def list_completed_subgraphs(
    ckpt_dir: Optional[Path] = None,
    *,
    verbose: bool = True,
) -> Set[int]:
    """Scan the checkpoint dir, return the set of subgraph indices that
    have a validated (loadable) checkpoint file.

    A checkpoint file is named `subgraph_<N>.pt` where N is the 0-indexed
    subgraph order GPTQ processes them in.

    Files that fail `torch.load` (corrupt, truncated by a kill mid-write)
    are reported but excluded — the run will redo those subgraphs.
    """
    d = _resolve_ckpt_dir(ckpt_dir)
    if not d.exists():
        if verbose:
            print(f"[ckpt] no checkpoint dir at {d}; starting fresh", flush=True)
        return set()
    completed: Set[int] = set()
    corrupt: List[Path] = []
    for f in sorted(d.glob("subgraph_*.pt")):
        try:
            n = int(f.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        try:
            torch.load(f, map_location="cpu", weights_only=False)
            completed.add(n)
        except Exception as e:
            corrupt.append(f)
            if verbose:
                print(f"[ckpt] WARN: {f} unloadable ({type(e).__name__}: "
                      f"{str(e)[:120]}); will re-quantize subgraph {n}", flush=True)
    # Clean up any leftover .tmp files (from a kill during _atomic_torch_save)
    for tmp in d.glob("subgraph_*.pt.tmp"):
        try:
            tmp.unlink()
        except OSError:
            pass
    if verbose:
        print(f"[ckpt] {len(completed)} validated checkpoints in {d}: "
              f"{sorted(completed)[:10]}{'...' if len(completed)>10 else ''}",
              flush=True)
        if corrupt:
            print(f"[ckpt] {len(corrupt)} corrupt checkpoints ignored", flush=True)
    return completed


def restore_checkpoints_into(
    model: torch.nn.Module,
    completed: Set[int],
    *,
    subgraph_index_to_modules: Dict[int, List[str]],
    ckpt_dir: Optional[Path] = None,
    verbose: bool = True,
) -> int:
    """For each completed subgraph, load its checkpoint and splice the
    saved state_dicts back into `model`. Returns the number of subgraphs
    successfully restored.

    `subgraph_index_to_modules` maps subgraph index N → list of module
    names that should be in checkpoint N. Used to verify the checkpoint
    isn't from a different recipe / world-size config.
    """
    d = _resolve_ckpt_dir(ckpt_dir)
    n_restored = 0
    for n in sorted(completed):
        ckpt_path = d / f"subgraph_{n}.pt"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        expected = set(subgraph_index_to_modules.get(n, []))
        saved = set(payload.get("module_names", []))
        if expected and saved != expected:
            if verbose:
                print(f"[ckpt] subgraph {n}: name set mismatch "
                      f"(expected {len(expected)} modules, got {len(saved)}); "
                      f"skipping restore — will re-quantize this subgraph",
                      flush=True)
            continue
        sd = payload["state_dict"]
        # Map names back to modules in the live model
        named = dict(model.named_parameters())
        named_buf = dict(model.named_buffers())
        missing = []
        for key, tensor in sd.items():
            if key in named:
                with torch.no_grad():
                    named[key].copy_(tensor.to(named[key].dtype))
            elif key in named_buf:
                with torch.no_grad():
                    named_buf[key].copy_(tensor.to(named_buf[key].dtype))
            else:
                missing.append(key)
        if missing and verbose:
            print(f"[ckpt] subgraph {n}: {len(missing)} keys not found in "
                  f"current model (first 3: {missing[:3]})", flush=True)
        n_restored += 1
        if verbose:
            print(f"[ckpt] restored subgraph {n} ({len(sd)} tensors)", flush=True)
    return n_restored


def install_subgraph_checkpoint_hook(
    *,
    rank: int,
    world_size: int,
    completed: Set[int],
    ckpt_dir: Optional[Path] = None,
    verbose: bool = True,
) -> None:
    """Patch `GPTQModifier.compress_module_list` to:
      1. Skip subgraphs already in `completed` (no-op the compress step,
         leaving the restored weights in place).
      2. After a compress completes, atomically dump the just-quantized
         modules' state_dicts to `<ckpt_dir>/subgraph_<N>.pt` (only on
         rank 0 — sharded modules' owning rank's view is canonical and
         rank 0 already gets the broadcast result from patch C).

    The subgraph index is incremented each call. GPTQModifier calls
    `compress_module_list` once per pipeline subgraph (one per
    DeepseekV4DecoderLayer in our recipe).
    """
    import llmcompressor.modifiers.gptq.base as _gptq

    assert hasattr(_gptq.GPTQModifier, "compress_module_list"), \
        "[ckpt] GPTQModifier.compress_module_list missing — upstream API changed"
    _orig = _gptq.GPTQModifier.compress_module_list
    d = _resolve_ckpt_dir(ckpt_dir)

    # State held across calls (one GPTQModifier instance per run, so this
    # state lives in module-global scope rather than per-instance).
    _state = {"idx": 0}

    def _patched(self, module_list):
        n = _state["idx"]
        _state["idx"] += 1

        if n in completed:
            if rank == 0 and verbose:
                print(f"[ckpt] subgraph {n}: SKIP (already in checkpoint)",
                      flush=True)
            return

        t0 = time.time()
        try:
            _orig(self, module_list)
        except Exception:
            if rank == 0 and verbose:
                print(f"[ckpt] subgraph {n}: FAILED at "
                      f"{time.time()-t0:.0f}s; no checkpoint written",
                      flush=True)
            raise

        # Dump checkpoint on rank 0 only. After patch C ran the broadcast
        # for replicated modules, rank 0 has the correct quantized state
        # for everything. For sharded modules (only one rank holds the
        # state), each owning rank dumps its slice separately.
        d.mkdir(parents=True, exist_ok=True)
        sd: Dict[str, torch.Tensor] = {}
        names: List[str] = []
        for module in module_list:
            name = self._module_names.get(module, "")
            names.append(name)
            # state_dict includes weight, weight_scale, weight_zero_point
            local_sd = module.state_dict(prefix=name + ".", keep_vars=False)
            for k, v in local_sd.items():
                # Only persist tensors that are populated. (e.g. zero_point
                # may be missing on symmetric quant).
                if isinstance(v, torch.Tensor):
                    sd[k] = v.detach().to("cpu")
        payload = {
            "subgraph_index": n,
            "module_names": names,
            "state_dict": sd,
            "rank": rank,
            "world_size": world_size,
            "timestamp": time.time(),
        }
        # On multi-rank, each rank dumps its slice to a rank-suffixed file.
        # On single-rank, drop the suffix.
        if world_size > 1:
            dst = d / f"subgraph_{n}.rank{rank}.pt"
        else:
            dst = d / f"subgraph_{n}.pt"
        _atomic_torch_save(payload, dst)
        if rank == 0 and verbose:
            print(f"[ckpt] subgraph {n}: wrote {dst.name} "
                  f"({len(sd)} tensors, {time.time()-t0:.0f}s)", flush=True)

    _gptq.GPTQModifier.compress_module_list = _patched

    if verbose:
        print(f"[ckpt] hook installed; ckpt_dir={d}, "
              f"completed={len(completed)} subgraphs", flush=True)


__all__ = [
    "list_completed_subgraphs",
    "restore_checkpoints_into",
    "install_subgraph_checkpoint_hook",
]
