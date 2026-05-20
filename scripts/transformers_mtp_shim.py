"""Runtime monkey-patch that adds the `DeepseekV4NextNPredictor` MTP class
to `transformers.models.deepseek_v4.modeling_deepseek_v4` and re-instantiates
`DeepseekV4Model.__init__` so `self.mtp` exists.

Use this at the top of any script that loads DSv4 via
`AutoModelForCausalLM.from_pretrained(...)` and needs the `mtp.*` weights
preserved. The shim mirrors `patches/transformers_dsv4_mtp.py.diff` and
is the runtime equivalent of that patch — useful for development and
running on a pip-installed transformers without sudo-editing site-packages.

  from scripts.transformers_mtp_shim import install_mtp_shim
  install_mtp_shim()
  # ...then from_pretrained as normal

When the upstream PR lands and transformers ships an MTP class natively,
delete this module and the corresponding patch.

PR-candidacy status: see patches/UPSTREAM_PR_DRAFTS.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def install_mtp_shim(verbose: bool = True) -> None:
    """Patch `transformers.models.deepseek_v4.modeling_deepseek_v4` in-place
    to add `DeepseekV4NextNPredictor` and re-wrap `DeepseekV4Model.__init__`
    so it instantiates `self.mtp` based on `config.num_nextn_predict_layers`.

    Idempotent: calling twice is a no-op (re-detects the existing shim and
    returns).
    """
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as _m

    if hasattr(_m, "DeepseekV4NextNPredictor"):
        if verbose:
            print("[mtp-shim] already installed; skipping", flush=True)
        return

    DecoderLayer = _m.DeepseekV4DecoderLayer
    RMSNorm = _m.DeepseekV4RMSNorm

    def _extend_layer_type_lists(config):
        """Extend `config.layer_types` and `config.mlp_layer_types` to cover
        the MTP layer's index. The MTP block in DSv4-Flash uses a SIMPLER
        attention than the main `compressed_sparse_attention` /
        `heavily_compressed_attention` types — it has only the
        wq_a/wq_b/wkv/wo_a/wo_b projections + attn_sink/q_norm/kv_norm, with
        NO compressor and NO indexer submodules. The matching layer_type is
        `sliding_attention` (which sets `self.compressor = None` in
        `DeepseekV4Attention.__init__`).

        For mlp_layer_types, MTP uses the standard `moe` type — same MoE
        structure as the main layers (256 routed experts + shared experts).
        """
        n_mtp = getattr(config, "num_nextn_predict_layers", 0)
        if n_mtp <= 0:
            return
        need = config.num_hidden_layers + n_mtp
        if getattr(config, "layer_types", None) is not None and len(config.layer_types) < need:
            config.layer_types = list(config.layer_types) + ["sliding_attention"] * (need - len(config.layer_types))
        if getattr(config, "mlp_layer_types", None) is not None and len(config.mlp_layer_types) < need:
            config.mlp_layer_types = list(config.mlp_layer_types) + ["moe"] * (need - len(config.mlp_layer_types))

    class DeepseekV4NextNPredictor(DecoderLayer):
        """DSv4 MTP draft-head block — see
        patches/transformers_dsv4_mtp.py.diff for the canonical docstring
        and patches/UPSTREAM_PR_DRAFTS.md for the upstream-PR plan."""

        def __init__(self, config, layer_idx: int):
            _extend_layer_type_lists(config)
            super().__init__(config, layer_idx)
            hidden = config.hidden_size
            eps = config.rms_norm_eps
            self.e_proj = nn.Linear(hidden, hidden, bias=False)
            self.h_proj = nn.Linear(hidden, hidden, bias=False)
            self.enorm = RMSNorm(hidden, eps=eps)
            self.hnorm = RMSNorm(hidden, eps=eps)
            self.norm = RMSNorm(hidden, eps=eps)
            hc_mult = config.hc_mult
            hc_dim = hc_mult * hidden
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32))
            self.hc_head_base = nn.Parameter(
                torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32))

        # NOTE: forward is intentionally not overridden in the shim — for
        # calibration we only need the weights to land in nn.Modules so
        # named_modules()/named_parameters() enumerate them. The full MTP
        # inference forward composition (enorm, hnorm, e_proj/h_proj,
        # super().forward, shared_head) is captured in the upstream-PR
        # diff at patches/transformers_dsv4_mtp.py.diff.

    _m.DeepseekV4NextNPredictor = DeepseekV4NextNPredictor

    # Re-wrap DeepseekV4Model.__init__ to instantiate self.mtp.
    # `_extend_layer_type_lists` is defined above (used by both the class
    # __init__ and the Model wrapper) — see its docstring for rationale.
    Model = _m.DeepseekV4Model
    _orig_init = Model.__init__

    def _patched_init(self, config):
        _extend_layer_type_lists(config)
        _orig_init(self, config)
        n_mtp = getattr(config, "num_nextn_predict_layers", 0)
        if n_mtp > 0:
            self.mtp = nn.ModuleList(
                [
                    DeepseekV4NextNPredictor(config, config.num_hidden_layers + i)
                    for i in range(n_mtp)
                ]
            )
        else:
            self.mtp = nn.ModuleList()

    Model.__init__ = _patched_init

    if verbose:
        print("[mtp-shim] installed: DeepseekV4NextNPredictor + Model.mtp",
              flush=True)


def install_mtp_conversion_mapping_extension(verbose: bool = True) -> None:
    """Extend `transformers.conversion_mapping`'s registered DSv4 entries
    with `mtp.<M>.*` equivalents for every `^layers\\.` entry.

    Background: transformers ships 41 `WeightRenaming` entries for DSv4 that
    rename the upstream-internal checkpoint naming to HF naming (`embed.`
    → `embed_tokens.`, `attn.` → `self_attn.`, `ffn.` → `mlp.`,
    `attn_norm.` → `input_layernorm.`, `attn.attn_sink` → `self_attn.sinks`,
    `hc_attn_fn` → `attn_hc.fn`, etc.). All are anchored at `^layers\\.`.

    The MTP block (`mtp.0.*`) has structurally the same submodules as a
    `DeepseekV4DecoderLayer` (per the `DeepseekV4NextNPredictor` shim that
    inherits from it), so it needs the same rename rules — but with the
    anchor swapped from `^layers\\.` to `^mtp\\.`. Without this, `mtp.0.*`
    keys stay in upstream naming after load (e.g. `mtp.0.attn.wq_a.weight`),
    don't match any of the shim's submodules (which use HF names like
    `mtp.0.self_attn.q_a_proj.weight`), and the modules are flagged as
    uninitialized → `_init_weights` random-initializes them.

    This helper builds the parallel `mtp.\\d+.*` entries by string-substitution
    on the existing `^layers\\.(\\d+)\\.` patterns, and re-registers the
    combined list.
    """
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
        # Normalize: both may be list-of-strings or string. We treat lists.
        sp_list = sp if isinstance(sp, (list, tuple)) else [sp]
        tp_list = tp if isinstance(tp, (list, tuple)) else [tp]
        # Only convert entries anchored at `^layers\.(\d+)\.` — these map main
        # decoder layer keys. The 6 entries that match `^embed\.`, `^head\.`,
        # `^norm\.`, `^hc_head_*` are model-level and don't apply to MTP.
        new_sp = []
        new_tp = []
        for s, t in zip(sp_list, tp_list):
            if isinstance(s, str) and s.startswith(r"^layers\.(\d+)\."):
                new_sp.append(s.replace(r"^layers\.(\d+)\.", r"^mtp\.(\d+)\.", 1))
                new_tp.append(t.replace("layers.\\1.", "mtp.\\1.", 1))
        if new_sp:
            # Reconstruct a WeightRenaming with the same constructor kwargs
            cls = type(entry)
            mtp_entry = cls(
                source_patterns=new_sp if len(new_sp) > 1 else new_sp[0],
                target_patterns=new_tp if len(new_tp) > 1 else new_tp[0],
            )
            added.append(mtp_entry)
    combined = list(existing) + added
    register_checkpoint_conversion_mapping("deepseek_v4", combined, overwrite=True)
    if verbose:
        print(f"[mtp-conv-map] added {len(added)} mtp.\\d+.* entries to "
              f"deepseek_v4 conversion_mapping (was {len(existing)}, "
              f"now {len(combined)})", flush=True)


__all__ = ["install_mtp_shim", "install_mtp_conversion_mapping_extension"]
