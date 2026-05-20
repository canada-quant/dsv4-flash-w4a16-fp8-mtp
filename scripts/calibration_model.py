"""CalibrationModel — wrapper that drives the vendored upstream Transformer
through main layers AND the MTP block on a single forward call.

The upstream ``Transformer.forward`` (vendor/dsv4-upstream/model.py line 802)
only iterates ``self.layers``; ``self.mtp[i]`` is never invoked. That means
during calibration via ``llmcompressor.oneshot`` the MTP block's Linears
receive zero activation hooks and GPTQ collects zero Hessian data for them.

Per the upstream ``MTPBlock.forward`` signature (vendor/dsv4-upstream/model.py
line 738+), MTP takes:

    forward(x, start_pos, input_ids)
    # x: [b, s, hc_mult, d]   the hidden state from the previous layer (pre-norm)
    # input_ids: [b, s]       embedded internally by mtp.embed (aliased to t.embed)

The MTP block itself applies enorm / hnorm on the embedding and hidden state
respectively, so we hand it the PRE-NORM main-layers output (i.e., the same
``h`` shape that the main path's ``self.head`` receives — but ``head``'s
internal norm is NOT applied to MTP's input).

This wrapper:
  1. Replicates the main forward path explicitly (embed -> hc-expand ->
     iterate ``t.layers``).
  2. Feeds the final ``h`` (pre-norm, post-HC-mixing) to each ``t.mtp[i]`` so
     GPTQ hooks attached to MTP Linears get real, layer-42-output-distribution
     activations.
  3. Returns the main path's logits in an HF-style object with ``.logits`` so
     llmcompressor's calibration loop sees the expected interface.

The MTP logits are discarded — we only need the activations to flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# _LogitsOut removed — torch.fx's create_arg (via llm-compressor's
# SequentialTracer) raises NotImplementedError on any custom return type that
# isn't a Tensor / list / dict / NamedTuple-the-tracer-recognises. The class
# was wrapping logits for HF-style ``.logits`` access; calibration doesn't
# need that. forward returns Tensor.


class CalibrationModel(nn.Module):
    """Forward = main 0..N-1 layers, then drive every mtp[i] with the final
    main-layer hidden state.

    Args:
        transformer: a shimmed ``vendor.dsv4-upstream.model.Transformer``
            instance (already loaded with BF16 weights).
    """

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids: torch.Tensor, **_unused) -> torch.Tensor:
        """Return the raw logits tensor — NOT a wrapped object.

        llm-compressor's sequential calibrator runs fx symbolic tracing on
        forward; arbitrary classes raise ``NotImplementedError`` from fx's
        ``create_arg``. The Tensor return type is first-class fx-traceable.
        Calibration uses the return value only as a trace-graph sink; it
        doesn't read ``.logits``.
        """
        t = self.transformer

        h = t.embed(input_ids)
        # Expand to hc_mult copies for the Hyper-Connection residual stream.
        h = h.unsqueeze(2).repeat(1, 1, t.hc_mult, 1)

        for layer in t.layers:
            h = layer(h, 0, input_ids)

        # h here is exactly what the upstream Transformer.forward hands to
        # self.head (line 808). It's also exactly what MTPBlock.forward
        # expects as its `x` input (which then calls hnorm(x) internally).
        # Drive MTP — discard logits, but hooks fire on e_proj / h_proj /
        # attn.* / ffn.experts.* inside the MTP block.
        for mtp_layer in t.mtp:
            _ = mtp_layer(h, 0, input_ids)

        return t.head(h, t.hc_head_fn, t.hc_head_scale, t.hc_head_base, t.norm)
