#!/usr/bin/env python3
"""Phase 2: GPTQ W4A16-FP8 calibration of DeepSeek-V4-Flash, including MTP.

**Status (2026-05-19): structural scaffold, not yet runnable end-to-end.**

Why this script is a stub
-------------------------
The predecessor's W4A16 recipe (`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`) ran
GPTQ over `DeepseekV4DecoderLayer` modules with `llmcompressor.oneshot`. That
worked because every routed-expert MLP and every attention projection it
needed was reachable as an `nn.Module` attribute of the loaded transformers
model.

For this repo we need to also calibrate the MTP layer. Verified by direct
inspection of `transformers==5.8.1` on 2026-05-19:

  $ ls .../transformers/models/deepseek_v4/
  __init__.py  configuration_deepseek_v4.py  modeling_deepseek_v4.py  modular_deepseek_v4.py

  $ grep -ni "class.*mtp\|class.*MultiToken\|class.*NextN" .../modeling_deepseek_v4.py
  (no matches)

Transformers 5.8.1 has the DSv4 *architecture* but **no MTP module class** —
only `num_nextn_predict_layers: int = 1` in `configuration_deepseek_v4.py`.
With the load-time regex neutralized (see patches/modeling_deepseek_v4.py.diff
hunk 1), `from_pretrained` will *deserialize* the 1,575 mtp.* tensors but
they have no `nn.Module` to attach to — they sit in the state-dict dropbox
and are absent from `model.parameters()`. GPTQ traverses
`model.modules()`, so it cannot see them.

Three resolutions, in increasing scope:

  (1) **Shim module in this script** — define a minimal `DeepSeekV4MTPLayer`
      that owns the e_proj, h_proj, shared_head, norms, the inner decoder
      layer, and the hc_* heads. Attach it as `model.mtp = MTPLayer(...)` in
      the calibration entry point. Run the calibration with a custom
      forward that pipes hidden_states from the main model's layer 42 output
      through the MTP layer so GPTQ hooks see real activations.

  (2) **Patch transformers** to add a `DeepSeekV4MTP` class. Heavier; will
      conflict the moment upstream lands MTP support.

  (3) **Two-pass calibration** — first oneshot over the main model with
      `mtp.*` keys present-but-unattached, then a second oneshot focused on
      the MTP block alone, fed with hidden_states captured during pass one.
      Most decoupled but doubles the calibration wall-clock.

Approach (1) is the planned path. Filling in `MTPLayer` requires reading the
upstream `inference/model.py` to learn the exact wiring of e_proj, h_proj,
hc_0..3 (hypercompressed vocab projection), enorm/hnorm, attn_sink, and the
ffn.gate / ffn.shared_experts / ffn.experts layout. That work is the next
piece of this script; it is intentionally deferred from this commit so the
scaffold can land and Phase 1 results can be inspected first.

Recipe topology (target)
------------------------
- routed experts, layers 0..42 AND mtp.0: W4A16 INT4 group=128 sym, GPTQ
- attention projections, layers 0..42 AND mtp.0: FP8_BLOCK 128x128 data-free
- ignore (BF16 passthrough): lm_head, embeddings, all *norm*, *gate*,
  *shared_experts*, *hc_*, *attn_sink*, MTP-specific: e_proj, h_proj,
  shared_head.*, enorm, hnorm, attn_norm, attn_sink, hc_0..3, ffn.gate,
  ffn.shared_experts, input_layernorm, post_attention_layernorm

Names below use DeepSeek's *internal* convention (`attn.wq_a`, `ffn.experts`,
no `model.` prefix), verified against the upstream HF checkpoint and the
post-dequant BF16 output of `scripts/dequant_mtp.py`.

Calibration data: HuggingFaceH4/ultrachat_200k, V4 manual chat encoding
(no Jinja template). Same as predecessor.

Launch
------
    torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16_mtp.py \\
        --input  /scratch/weights/bf16-mtp \\
        --output /scratch/weights/w4a16-fp8-mtp \\
        --samples 768 --batch-size 4

Memory budget on 8x B300 (275 GB HBM each): 284B BF16 ~568 GB, with
oneshot offload-hessians and sequential targets the residency is one
decoder layer at a time + offloaded ghosts.
"""
import argparse
import os
import sys

# ---- TODO: implement DeepSeekV4MTPLayer + main() ----


def main():
    raise SystemExit(
        "quantize_v4_w4a16_mtp.py is a scaffold; the MTP shim module is the "
        "next implementation step. See the docstring for the chosen approach "
        "(option 1: in-script shim) and the open questions about hc_*, e_proj, "
        "h_proj wiring that need answering from inference/model.py first."
    )


if __name__ == "__main__":
    main()
