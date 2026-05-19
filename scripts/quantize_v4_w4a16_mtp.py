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

Approach (1) is the planned path.

MTPBlock structure (extracted from upstream inference/model.py, 2026-05-19):

    class MTPBlock(Block):
        e_proj      Linear(dim, dim)      # quantize FP8_BLOCK
        h_proj      Linear(dim, dim)      # quantize FP8_BLOCK
        enorm       RMSNorm               # ignore (BF16)
        hnorm       RMSNorm               # ignore
        norm        RMSNorm               # ignore
        hc_head_fn   FP32 param            # ignore
        hc_head_base FP32 param            # ignore
        hc_head_scale FP32 param           # ignore
        embed, head  aliases to main model # shared, do not duplicate

    class Block (parent of MTPBlock):
        attn        Attention              # FP8_BLOCK on wq_a/wq_b/wkv/wo_a/wo_b
        ffn         MoE                    # W4A16 on experts.*.w1/w2/w3
        attn_norm   RMSNorm                # ignore
        ffn_norm    RMSNorm                # ignore
        hc_attn_fn/base/scale FP32 params  # ignore (hyper-connection)
        hc_ffn_fn/base/scale  FP32 params  # ignore

    class Attention (used by both):
        wq_a, wq_b, wkv, wo_a, wo_b  Linear in FP8_BLOCK
        q_norm, kv_norm              RMSNorm
        attn_sink                    FP32 param
        compressor                   nested submodule (wkv, wgate, norm)
        indexer                      nested submodule (wq_b, weights_proj, compressor)

The non-standard `hc_pre` / `hc_post` hyper-connection ops in Block.forward
do NOT contain any Linear modules — they are pure tensor algebra over
hc_attn_fn, hc_attn_base, hc_attn_scale parameters. GPTQ will not need to
hook them. The MTP shim therefore only needs to make the Linear modules
reachable by `model.named_modules()` and produce reasonable activations
during the calibration forward.

Inventory of mtp.0.* tensors in the upstream checkpoint (1,575 total,
verified 2026-05-19):
  ffn         1,544  — 256 experts x w1/w2/w3 weight+scale = 1,536,
                       plus shared_experts.w1/w2/w3 weight+scale and gate
  attn           13  — wq_a/wq_b/wkv/wo_a/wo_b weight+scale = 10, plus
                       attn_sink, q_norm.weight, kv_norm.weight = 3
  e_proj          2  — weight + scale (quantized — Linear(dim, dim))
  h_proj          2  — weight + scale (quantized — Linear(dim, dim))
  hc_*           9  — hc_attn_{fn,base,scale}, hc_ffn_{fn,base,scale},
                       hc_head_{fn,base,scale} — all FP32 params, BF16 pass
  attn_norm/ffn_norm/enorm/hnorm/norm  5  — RMSNorm weights, BF16
                       (norm = shared_head.norm)

The earlier PLAN.md note that ``e_proj`` and ``h_proj`` should be in the
*ignore* list is incorrect — they ARE quantized in the upstream (each has a
.scale). They belong in the FP8_BLOCK regex along with attn.wq_a etc.

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
