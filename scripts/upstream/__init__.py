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


# ---- 5) ensure world_size / rank globals do not require torch.distributed ----
_module.world_size = 1
_module.rank = 0


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

    # Upstream config has fields that aren't ModelArgs members (e.g.,
    # `expert_dtype` in the config is honoured at construction time; we
    # force bf16 here). Filter to known ModelArgs fields.
    import dataclasses

    arg_fields = {f.name for f in dataclasses.fields(ModelArgs)}
    kwargs = {k: v for k, v in cfg.items() if k in arg_fields}
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
]
