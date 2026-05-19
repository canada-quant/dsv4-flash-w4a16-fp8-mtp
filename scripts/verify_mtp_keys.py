#!/usr/bin/env python3
"""Phase 1 verification gate — confirm MTP tensors survived dequant.

Reads ``model.safetensors.index.json`` from the output dir, counts every key
containing ``mtp`` (case-insensitive), prints a sample, exits non-zero if zero
MTP tensors are present. Run this before Phase 2 (GPTQ calibration); a zero
count means the dequant silently dropped the MTP block and any downstream
calibration would silently emit an MTP-less checkpoint, regenerating the bug
that motivated this whole repo.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <bf16-mtp-dir>", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(sys.argv[1])
    index_path = out_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"FATAL: {index_path} not found", file=sys.stderr)
        sys.exit(1)

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    mtp_keys = sorted(k for k in weight_map if "mtp" in k.lower())

    print(f"total tensors: {len(weight_map)}")
    print(f"MTP tensors:   {len(mtp_keys)}")
    if mtp_keys:
        print("\nfirst 12 MTP keys:")
        for k in mtp_keys[:12]:
            print(f"  {k}")
        print(f"\n(showing 12 of {len(mtp_keys)})")

    if not mtp_keys:
        print("\nFATAL: zero MTP tensors. Phase 2 will not calibrate layer 43.",
              file=sys.stderr)
        sys.exit(1)

    print("\nMTP gate PASSED")


if __name__ == "__main__":
    main()
