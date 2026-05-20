"""scripts.upstream — calibration-friendly adapter for vendor/dsv4-upstream/model.py.

Side-effect imports patch sys.modules to make the vendored upstream model.py
importable without tilelang or fast_hadamard_transform, with upstream's
custom Linear class swapped for an ``nn.Linear`` subclass so
``llmcompressor``'s GPTQ matcher recognises calibration targets.

Usage::

    from scripts.upstream import Transformer, ModelArgs, MTPBlock, build_model_args

The strategy avoids forking the 827-line vendor model.py — every patch is a
namespace rebinding *after* import, exploiting the fact that Python looks up
class names at attribute-access time (so reassigning ``model.Linear`` after
import affects every subsequent ``Linear(...)`` call inside class
``__init__`` methods, even ones defined before the swap).

The one place this trick wouldn't work — ``class ColumnParallelLinear(Linear):``
which captures Linear at class-def time — is handled by *also* rebinding
``ColumnParallelLinear`` and ``RowParallelLinear`` to the plain Linear class.
With ``world_size == 1`` (single-GPU calibration), the TP variants are
behaviourally identical to plain Linear.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist

# ---- 1) shim out tilelang-dependent kernel.py ---------------------------------
from . import kernel_shim
sys.modules["kernel"] = kernel_shim

# ---- 2) stub fast_hadamard_transform (used only by FP8 QAT path) --------------
_fht = types.ModuleType("fast_hadamard_transform")
_fht.hadamard_transform = lambda x, scale=1.0: x  # identity stub
sys.modules["fast_hadamard_transform"] = _fht

# ---- 3) import the vendored model module by file path -------------------------
_VENDOR = Path(__file__).resolve().parent.parent.parent / "vendor" / "dsv4-upstream"
_MODEL_PY = _VENDOR / "model.py"
if not _MODEL_PY.exists():
    raise RuntimeError(f"vendor model.py missing at {_MODEL_PY}")

# Important: cwd-relative imports inside the vendor file ("from kernel import ...")
# need 'kernel' on sys.modules BEFORE this exec — already handled above.
_spec = importlib.util.spec_from_file_location("dsv4_upstream_model", str(_MODEL_PY))
_module = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_upstream_model"] = _module
_spec.loader.exec_module(_module)


# ---- 4) replace upstream Linear with nn.Linear subclass ----------------------
class GPTQLinear(nn.Linear):
    """nn.Linear with upstream-Linear-compatible signature.

    Upstream Linear: ``Linear(in_features, out_features, bias=False, dtype=None)``
    nn.Linear:      ``nn.Linear(in_features, out_features, bias=True)``

    This wrapper bridges them: default ``bias=False`` (matches upstream),
    default ``dtype=bfloat16`` (the calibration dtype), and exposes the
    ``.scale`` attribute as ``None`` for compat with any upstream code that
    touches it (the dispatch in upstream's free function ``linear()`` checks
    ``weight.scale``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            dtype=dtype or torch.bfloat16,
        )
        # Compat: upstream's linear() dispatch reads weight.scale to decide
        # FP8/FP4 vs F.linear. None -> F.linear path (what we want).
        self.weight.scale = None
        self.register_parameter("scale", None)

    def reset_parameters(self) -> None:
        """No-op. ``nn.Linear.__init__`` calls ``reset_parameters`` (kaiming
        init) on every weight; for 568 GB of expert weights that's tens of
        minutes of CPU random-number generation we throw away when we
        immediately overwrite from safetensors. Skipping the init leaves the
        weight as uninitialized memory, which is fine because ``copy_`` from
        the safetensors fills every byte before any forward pass reads it.
        """
        pass


# Swap upstream's Linear class in the vendored module's namespace.
# Every Attention / MoE / Expert __init__ inside the vendored module
# resolves "Linear" via the module dict at call time, so this rebinding
# affects all subsequent instantiations.
_module.Linear = GPTQLinear

# ColumnParallelLinear / RowParallelLinear inherit from upstream Linear at
# class-def time, so they kept the old base. With world_size = 1 they are
# behaviourally identical to plain Linear; rebind their names directly.
_module.ColumnParallelLinear = GPTQLinear
_module.RowParallelLinear = GPTQLinear


# ---- 5) world_size / rank — decoupled expert sharding ----
# Upstream world_size stays at 1 (so ParallelEmbedding / Head / Attention stay
# full-size, since our GPTQLinear-rebound Column/RowParallelLinear does NOT
# implement TP-aware reduce). Independently, _expert_world_size / _expert_rank
# drive MoE.__init__ to shard experts N-way across ranks, giving us the
# memory savings without the TP correctness rewrite.
_module.world_size = 1
_module.rank = 0
_module._expert_world_size = 1
_module._expert_rank = 0


# Patch MoE.__init__ + MoE.forward to use _expert_world_size / _expert_rank
# instead of upstream's globals.
_orig_moe_init = _module.MoE.__init__


def _moe_init_ep(self, layer_id, args):
    nn.Module.__init__(self)
    self.layer_id = layer_id
    self.dim = args.dim
    ews = _module._expert_world_size
    er = _module._expert_rank
    assert args.n_routed_experts % ews == 0, (
        f"n_routed_experts={args.n_routed_experts} not divisible by "
        f"expert_world_size={ews}"
    )
    self.n_routed_experts = args.n_routed_experts
    self.n_local_experts = args.n_routed_experts // ews
    self.n_activated_experts = args.n_activated_experts
    self.experts_start_idx = er * self.n_local_experts
    self.experts_end_idx = self.experts_start_idx + self.n_local_experts
    self.gate = _module.Gate(layer_id, args)
    expert_dtype = (
        torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
    )
    self.experts = nn.ModuleList([
        _module.Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype,
                       swiglu_limit=args.swiglu_limit)
        if self.experts_start_idx <= i < self.experts_end_idx else None
        for i in range(self.n_routed_experts)
    ])
    assert args.n_shared_experts == 1
    self.shared_experts = _module.Expert(
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
    # Cross-rank reduce only when expert sharding is active.
    if _module._expert_world_size > 1:
        dist.all_reduce(y)
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


_module.MoE.__init__ = _moe_init_ep
_module.MoE.forward = _moe_fwd_ep


def apply_dist_state() -> tuple[int, int]:
    """Set decoupled expert-sharding state from torch.distributed.

    Upstream's ``world_size`` is forced to 1 (preserves full-size
    ParallelEmbedding / Head / Attention since our GPTQLinear-rebound
    Column/RowParallelLinear lacks TP-aware reduce). Separately,
    ``_module._expert_world_size`` / ``_expert_rank`` are set from
    ``torch.distributed`` so MoE.__init__ shards experts N-way.

    Memory math (8 ranks): each rank holds ~26 GB instead of ~568 GB,
    because 32 of 256 experts are populated per rank. Cross-rank Hessian
    reduce in llm-compressor + MoE.forward all_reduce handle correctness.

    Returns (expert_world_size, expert_rank).
    """
    if dist.is_available() and dist.is_initialized():
        _module._expert_world_size = dist.get_world_size()
        _module._expert_rank = dist.get_rank()
    else:
        _module._expert_world_size = 1
        _module._expert_rank = 0
    _module.world_size = 1
    _module.rank = 0
    return _module._expert_world_size, _module._expert_rank


# ---- 6) re-export the public API --------------------------------------------
ModelArgs = _module.ModelArgs
Transformer = _module.Transformer
Block = _module.Block
MTPBlock = _module.MTPBlock
Attention = _module.Attention
MoE = _module.MoE
Expert = _module.Expert
Gate = _module.Gate
RMSNorm = _module.RMSNorm
ParallelEmbedding = _module.ParallelEmbedding
ParallelHead = _module.ParallelHead


def build_model_args(
    upstream_config_path: str | os.PathLike[str],
    *,
    max_batch_size: int = 1,
    max_seq_len: int = 512,
    dtype: str = "bf16",
) -> "ModelArgs":
    """Construct a ``ModelArgs`` for calibration from upstream's config.json.

    Forces ``dtype="bf16"`` and small ``max_batch_size`` / ``max_seq_len`` so
    the kv_cache buffer in each Attention module is sized for calibration,
    not deployment. Drops ``expert_dtype="fp4"`` so all routed experts are
    materialised as BF16 (matching the dequant output at
    ``/scratch/weights/bf16-mtp``).
    """
    with open(upstream_config_path) as f:
        cfg: dict[str, Any] = json.load(f)

    # The bf16-mtp config.json was produced by HF transformers and uses HF
    # field names; upstream ModelArgs uses its own. Map HF -> upstream where
    # the keys differ. Same-named fields fall through via the filter below.
    hf_to_upstream = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "num_hash_layers": "n_hash_layers",
        "num_nextn_predict_layers": "n_mtp_layers",
        "num_attention_heads": "n_heads",
        "num_experts_per_tok": "n_activated_experts",
        "moe_intermediate_size": "moe_inter_dim",
        "scoring_func": "score_func",
        "routed_scaling_factor": "route_scale",
        "qk_rope_head_dim": "rope_head_dim",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "original_seq_len",
    }
    mapped: dict[str, Any] = {}
    for k, v in cfg.items():
        mapped[hf_to_upstream.get(k, k)] = v
    # rope_scaling subtree -> rope_factor / beta_fast / beta_slow / original_seq_len
    if isinstance(mapped.get("rope_scaling"), dict):
        rs = mapped.pop("rope_scaling")
        if "factor" in rs:
            mapped["rope_factor"] = rs["factor"]
        if "beta_fast" in rs:
            mapped["beta_fast"] = rs["beta_fast"]
        if "beta_slow" in rs:
            mapped["beta_slow"] = rs["beta_slow"]
        if "original_max_position_embeddings" in rs:
            mapped["original_seq_len"] = rs["original_max_position_embeddings"]

    import dataclasses
    arg_fields = {f.name for f in dataclasses.fields(ModelArgs)}
    kwargs = {k: v for k, v in mapped.items() if k in arg_fields}
    kwargs["dtype"] = dtype  # force bf16
    kwargs.pop("expert_dtype", None)  # force routed experts bf16 (no FP4 path)
    kwargs.pop("scale_fmt", None)  # only relevant for FP8/FP4 paths
    kwargs["max_batch_size"] = max_batch_size
    kwargs["max_seq_len"] = max_seq_len
    # ModelArgs.n_mtp_layers defaults to 1 (matches upstream `num_nextn_predict_layers`).
    # Upstream config.json does not carry this key so leave it at the default.

    # compress_ratios in the config is a list; ModelArgs declares it as a Tuple
    if "compress_ratios" in kwargs and isinstance(kwargs["compress_ratios"], list):
        kwargs["compress_ratios"] = tuple(kwargs["compress_ratios"])

    return ModelArgs(**kwargs)


__all__ = [
    "ModelArgs",
    "Transformer",
    "Block",
    "MTPBlock",
    "Attention",
    "MoE",
    "Expert",
    "Gate",
    "RMSNorm",
    "ParallelEmbedding",
    "ParallelHead",
    "GPTQLinear",
    "build_model_args",
    "apply_dist_state",
]
