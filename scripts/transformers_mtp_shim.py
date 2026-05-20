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
        """See docstring on the patched __init__ below."""
        n_mtp = getattr(config, "num_nextn_predict_layers", 0)
        if n_mtp <= 0:
            return
        for attr in ("layer_types", "mlp_layer_types"):
            lst = getattr(config, attr, None)
            if lst is None:
                continue
            need = config.num_hidden_layers + n_mtp
            if len(lst) < need:
                extension = [lst[-1]] * (need - len(lst))
                setattr(config, attr, list(lst) + extension)

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


__all__ = ["install_mtp_shim"]
