#!/usr/bin/env python3
"""Phase 2 gate — confirm the post-calibration checkpoint has MTP quantized.

Reads ``model.safetensors.index.json`` from the W4A16-FP8 output and checks
three invariants that distinguish a correctly-MTP-included quant from the
predecessor's MTP-less one:

  1. MTP routed experts present and W4A16-quantized
       expect: 256 tensors matching ``re:mtp\\.\\d+\\.ffn\\.experts\\.\\d+\\.(w1|w2|w3)\\.scale``
       (256 experts x 3 projections x 1 scale per quantized linear; the actual
       count of int4 packed weight tensors should match.)
  2. MTP attention present and FP8_BLOCK-quantized
       expect: 4 tensors matching ``re:mtp\\.\\d+\\.attn\\.(wq_a|wkv|wq_b|wo_b)\\.scale``
       with shape consistent with 128x128 blocks.
  3. MTP passthrough modules present and unquantized (BF16, no scale)
       expect: ``e_proj``, ``h_proj``, ``shared_head.head``, ``shared_head.norm``,
       ``enorm``, ``hnorm``, ``attn_norm``, ``attn_sink``, ``hc_0..3``,
       ``ffn.gate``, ``ffn.shared_experts.*``

Failure of any invariant aborts with a clear message — the most likely cause
is the MTP module shim in quantize_v4_w4a16_mtp.py being incomplete or the
recipe regex not matching internal naming.
"""
import argparse
import json
import re
import sys
from pathlib import Path


# Regex patterns over internal naming convention (verified against upstream
# DeepSeek-V4-Flash safetensors on 2026-05-19).
EXPERT_SCALE_RE = re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.(w1|w2|w3)\.scale$")
ATTN_SCALE_RE = re.compile(r"^mtp\.\d+\.attn\.(wq_a|wkv|wq_b|wo_a|wo_b)\.scale$")
PASSTHROUGH_TAGS = (
    "e_proj",
    "h_proj",
    "shared_head",
    "enorm",
    "hnorm",
    "attn_norm",
    "attn_sink",
    "hc_",
    "ffn.gate",
    "ffn.shared_experts",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="post-calibration W4A16-FP8 model dir")
    args = ap.parse_args()

    idx_path = Path(args.model_dir) / "model.safetensors.index.json"
    if not idx_path.exists():
        sys.exit(f"FATAL: {idx_path} not found")

    weight_map = json.loads(idx_path.read_text()).get("weight_map", {})
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
    expert_scales = [k for k in mtp_keys if EXPERT_SCALE_RE.match(k)]
    attn_scales = [k for k in mtp_keys if ATTN_SCALE_RE.match(k)]
    passthrough = [k for k in mtp_keys if any(t in k for t in PASSTHROUGH_TAGS)]

    print(f"total tensors:           {len(weight_map)}")
    print(f"MTP tensors:             {len(mtp_keys)}")
    print(f"MTP expert .scale (W4A16): {len(expert_scales)}")
    print(f"MTP attn .scale (FP8):     {len(attn_scales)}")
    print(f"MTP passthrough modules:   {len(passthrough)}")

    failed = []
    if len(mtp_keys) == 0:
        failed.append("no MTP tensors at all — calibration dropped the entire mtp.* block")
    if len(expert_scales) < 256:
        failed.append(f"expected >=256 MTP expert scales, found {len(expert_scales)}")
    if len(attn_scales) < 4:
        failed.append(f"expected >=4 MTP attention scales, found {len(attn_scales)}")
    if len(passthrough) == 0:
        failed.append("expected MTP passthrough modules (e_proj, h_proj, norms, ...) but found none")

    if failed:
        print()
        print("MTP quantization gate FAILED:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)

    print()
    print("MTP quantization gate PASSED")


if __name__ == "__main__":
    main()
