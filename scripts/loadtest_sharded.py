"""Sharded-load test — instantiate Transformer with expert sharding enabled
and load_safetensors_into filtering by expert_id % world_size, then print
RSS. No GPTQ, no GPU work, no oneshot.

Run via:
  torchrun --nproc-per-node 8 --master-port 29510 scripts/loadtest_sharded.py \
    --weights /scratch/weights/bf16-mtp --config /scratch/weights/bf16-mtp/config.json

The expected outcome: per-rank RSS ~120GB (vs 568GB unsharded), total
system RSS ~960GB (well under 3.9TB).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import resource
import torch
import torch.distributed as dist
import torch.nn as nn

# IMPORTANT: import the shim first so it patches kernels etc.
from scripts.upstream import (
    Transformer,
    Expert,
    Gate,
    build_model_args,
)
import dsv4_upstream_model as _module
from safetensors import safe_open


def patch_moe_for_expert_sharding(world_size: int, rank: int):
    """Monkey-patch MoE.__init__ + MoE.forward to use OUR ews/erk while
    keeping upstream's global ``world_size`` at 1 (so ParallelEmbedding /
    Head / Attention stay full-size).
    """
    _module._expert_world_size = world_size
    _module._expert_rank = rank

    def _moe_init_ep(self, layer_id, args):
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.dim = args.dim
        ews = _module._expert_world_size
        er = _module._expert_rank
        assert args.n_routed_experts % ews == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // ews
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = er * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = (
            torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        )
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype,
                   swiglu_limit=args.swiglu_limit)
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(
            args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit
        )

    def _moe_fwd_ep(self, x, input_ids):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            y[idx] += self.experts[i](x[idx], weights[idx, top, None])
        if _module._expert_world_size > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    _module.MoE.__init__ = _moe_init_ep
    _module.MoE.forward = _moe_fwd_ep


_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def _is_owned(name: str, ews: int, erk: int) -> bool:
    m = _EXPERT_RE.search(name)
    if m is None:
        return True  # non-expert tensor — every rank needs it
    eid = int(m.group(1))
    return eid % ews == erk


def sharded_load(transformer, weights_dir: Path, ews: int, erk: int,
                 verbose: bool = False):
    state = transformer.state_dict()
    state_keys = set(state.keys())
    loaded = 0
    skipped_not_owned = 0
    skipped_not_in_model = 0
    for shard in sorted(weights_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for raw_name in f.keys():
                if not _is_owned(raw_name, ews, erk):
                    skipped_not_owned += 1
                    continue
                if raw_name not in state_keys:
                    skipped_not_in_model += 1
                    continue
                tensor = f.get_tensor(raw_name)
                target = state[raw_name]
                if tensor.dtype != target.dtype:
                    tensor = tensor.to(target.dtype)
                target.copy_(tensor)
                loaded += 1
                if verbose and loaded % 5000 == 0:
                    print(f"  [rank {erk}] loaded {loaded}", flush=True)
    return loaded, skipped_not_owned, skipped_not_in_model


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # Init dist (gloo, no GPU)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=f"env://",
            rank=rank,
            world_size=world_size,
        )
    erk = rank
    ews = world_size

    is_main = (rank == 0)
    if is_main:
        print(f"[dist] world_size={ews} rank={erk}", flush=True)

    patch_moe_for_expert_sharding(ews, erk)

    if is_main:
        print(f"[args] building ModelArgs", flush=True)
    margs = build_model_args(args.config, max_batch_size=1, max_seq_len=128)
    if is_main:
        print(f"[args] n_routed_experts={margs.n_routed_experts} "
              f"n_layers={margs.n_layers} n_mtp={margs.n_mtp_layers}",
              flush=True)
        print(f"[args] per-rank local experts: "
              f"{margs.n_routed_experts // ews}", flush=True)

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")

    # Transformer.__init__ has ``global world_size; world_size = dist.get_world_size()``
    # which would re-shard ParallelEmbedding/ParallelHead/Attention to WS=8.
    # Mask dist during construction so the upstream globals stay at 1; MoE
    # uses ``_module._expert_world_size`` independently.
    _orig_is_init = dist.is_initialized
    _orig_get_ws = dist.get_world_size
    _orig_get_rk = dist.get_rank
    dist.is_initialized = lambda: False
    dist.get_world_size = lambda *a, **kw: 1
    dist.get_rank = lambda *a, **kw: 0
    t0 = time.time()
    try:
        transformer = Transformer(margs)
    finally:
        dist.is_initialized = _orig_is_init
        dist.get_world_size = _orig_get_ws
        dist.get_rank = _orig_get_rk
        _module.world_size = 1
        _module.rank = 0
    rss_after_init = rss_gb()
    n_params = sum(p.numel() for p in transformer.parameters())
    if is_main:
        print(f"[init] rank0 instantiated in {time.time()-t0:.1f}s; "
              f"params={n_params/1e9:.2f}B  RSS={rss_after_init:.1f} GB",
              flush=True)
    # Each rank reports
    print(f"[init] rank{erk} params={n_params/1e9:.2f}B  RSS={rss_after_init:.1f} GB",
          flush=True)

    t1 = time.time()
    loaded, skip_owned, skip_model = sharded_load(
        transformer, Path(args.weights), ews, erk, verbose=is_main
    )
    rss_after_load = rss_gb()
    print(f"[load] rank{erk} loaded={loaded} "
          f"skip_not_owned={skip_owned} skip_not_in_model={skip_model} "
          f"in {time.time()-t1:.1f}s  RSS={rss_after_load:.1f} GB",
          flush=True)

    # Compute total system RSS (each rank's RSS summed via all_reduce on a
    # gloo group — fine since these are small scalars)
    rss_tensor = torch.tensor([rss_after_load], dtype=torch.float64)
    if ews > 1:
        dist.all_reduce(rss_tensor)
    if is_main:
        print(f"[summary] total system RSS across {ews} ranks: "
              f"{rss_tensor.item():.1f} GB", flush=True)
        print(f"[summary] mean per-rank: {rss_tensor.item()/ews:.1f} GB",
              flush=True)

    # Sharding invariant: walk the model and assert experts are disjoint
    # across ranks. This is a precondition for the multirank_patches
    # (skip-reduce-for-sharded-modules) to be safe. If a layer's experts
    # are accidentally replicated, "skip reduce" would silently lose the
    # other ranks' Hessian contributions.
    from scripts.multirank_patches import assert_sharding_invariant
    assert_sharding_invariant(
        transformer,
        world_size=ews,
        rank=erk,
        n_routed_experts=margs.n_routed_experts,
        verbose=is_main,
    )


if __name__ == "__main__":
    main()
