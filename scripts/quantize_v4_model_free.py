#!/usr/bin/env python3
"""Phase 2 — Model-free RTN quantization of DSv4-Flash including MTP.

Uses ``llmcompressor.entrypoints.model_free.model_free_ptq`` which operates
directly on safetensors files — no model class, no HF PreTrainedModel
integration, no calibration data. The trade-off vs GPTQ is round-to-nearest
weight quantization (RTN) without Hessian-based refinement, applied
uniformly to every weight matching the scheme targets.

Why RTN is acceptable here:
  - FP8_BLOCK quantization of BF16 attention weights is essentially
    lossless via RTN (FP8 has ~3 mantissa bits, more than enough for
    BF16-trained activations); GPTQ refinement gives minor quality gains.
  - W4A16 is noticeably worse via RTN vs GPTQ on the main 43 layers,
    but the **predecessor's** `pastapaul/DeepSeek-V4-Flash-W4A16-FP8`
    already exists as a GPTQ-calibrated W4A16 main model. The unique
    contribution of *this* repo is the MTP layer being included; that
    contribution is preserved end-to-end here even with RTN.
  - This produces a fully functional quantized artifact that vLLM can
    load and serve, validating the entire Phase 0->Phase 4 pipeline
    in this session.

The script applies two passes to chain the schemes (model_free_ptq accepts
one scheme per call):

  Pass 1: W4A16 on routed experts only (everything else passes through BF16)
  Pass 2: FP8_BLOCK on attention/projections only (experts now W4A16
          pass through unchanged because they are int32-packed, not BF16)

CLI::

    python scripts/quantize_v4_model_free.py \\
        --input  /scratch/weights/bf16-mtp \\
        --output /scratch/weights/w4a16-fp8-mtp \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---- monkey-patch: handle dotless tensor names (top-level params) ----
# DSv4-Flash has 3 top-level params with no module prefix:
# hc_head_fn / hc_head_base / hc_head_scale. compressed_tensors's
# match_quantizable_tensors does name.rsplit('.', 1) which raises ValueError
# on dotless names. These tensors are never quantization targets (they're in
# our IGNORE list) but the function crashes before reaching ignore-check.
# Pre-filter them out of `tensors` so the rsplit succeeds.
import compressed_tensors.utils.match as _ctmatch  # noqa: E402

_orig_match = _ctmatch.match_quantizable_tensors


def _safe_match(tensors, ignore, targets, allow_nonquantizable=False):
    filtered = {k: v for k, v in tensors.items() if "." in k}
    return _orig_match(filtered, ignore, targets, allow_nonquantizable=allow_nonquantizable)


_ctmatch.match_quantizable_tensors = _safe_match
# Also patch the import already done by process.py
import llmcompressor.entrypoints.model_free.process as _mfproc  # noqa: E402
_mfproc.match_quantizable_tensors = _safe_match

from compressed_tensors.quantization import QuantizationScheme  # noqa: E402
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, W4A16  # noqa: E402
from llmcompressor.entrypoints.model_free import model_free_ptq  # noqa: E402


# ----------------------------- schemes -----------------------------

# Routed experts: W4A16 INT4 group=128 sym.  Targets in DeepSeek internal
# naming. Matches both main-model and mtp.0 expert paths because the regex
# is anchored on `.ffn.experts.N.{w1,w2,w3}$`.
EXPERTS_SCHEME = QuantizationScheme(
    targets=[r"re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$"],
    **W4A16,
)

# Attention projections + MTP entry/hidden projections: FP8_BLOCK 128x128.
# Same regexes match main-model attn.* and mtp.0.attn.*.
ATTN_SCHEME = QuantizationScheme(
    targets=[
        r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
        r"re:mtp\.\d+\.(e_proj|h_proj)$",
    ],
    **FP8_BLOCK,
)


# Modules deliberately left BF16. Names in DeepSeek internal convention.
# model_free_ptq automatically ignores anything ending in "norm" — we add the
# rest explicitly. `re:` prefix marks regex; bare strings are exact-match.
IGNORE = [
    "head",
    "embed",
    r"re:.*\.ffn\.gate$",
    r"re:.*\.ffn\.gate\.bias$",
    r"re:.*\.ffn\.gate\.tid2eid$",
    r"re:.*\.ffn\.shared_experts\..*",
    r"re:.*\.hc_.*",
    r"re:hc_.*",
    r"re:.*\.attn\.attn_sink$",
    # compressor / indexer auxiliary submodules under attn.* — keep BF16
    r"re:.*\.attn\.compressor\..*",
    r"re:.*\.attn\.indexer\..*",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Phase-1 BF16 dir")
    ap.add_argument("--output", required=True, help="output W4A16-FP8 dir")
    ap.add_argument("--device", default="cuda:0",
                    help="device for the on-the-fly quantization compute")
    ap.add_argument("--max-workers", type=int, default=1,
                    help="parallel shard workers; >1 risks GPU OOM on a single device")
    ap.add_argument("--intermediate", default=None,
                    help="dir for the W4A16-only intermediate (default: <output>.w4a16-tmp)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.exists():
        sys.exit(f"FATAL: input dir not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    intermediate = Path(args.intermediate) if args.intermediate else output_dir.parent / (output_dir.name + ".w4a16-tmp")
    intermediate.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ---- Pass 1: W4A16 on routed experts ----
    print(f"[pass 1] W4A16 routed experts: {input_dir} -> {intermediate}")
    print(f"         scheme targets: {EXPERTS_SCHEME.targets}")
    model_free_ptq(
        model_stub=str(input_dir),
        save_directory=str(intermediate),
        scheme=EXPERTS_SCHEME,
        ignore=IGNORE + [
            # During pass 1, attention is in ignore — pass 2 handles it
            r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
            r"re:mtp\.\d+\.(e_proj|h_proj)$",
        ],
        max_workers=args.max_workers,
        device=device,
    )

    # ---- Pass 2: FP8_BLOCK on attention/projections ----
    print(f"[pass 2] FP8_BLOCK attention + MTP projections: {intermediate} -> {output_dir}")
    print(f"         scheme targets: {ATTN_SCHEME.targets}")
    model_free_ptq(
        model_stub=str(intermediate),
        save_directory=str(output_dir),
        scheme=ATTN_SCHEME,
        ignore=IGNORE + [
            # Experts now W4A16-packed; pass them through unchanged
            r"re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$",
        ],
        max_workers=args.max_workers,
        device=device,
    )

    print()
    print(f"QUANTIZATION_DONE  output -> {output_dir}")
    print("Next: scripts/verify_mtp_quantized.py <output_dir>")


if __name__ == "__main__":
    main()
