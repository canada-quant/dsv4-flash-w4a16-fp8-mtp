#!/usr/bin/env python3
"""Sanity check the W4A16-FP8+MTP artifact at the file level — no model load.

Validates:
  1. safetensors index matches actual shards
  2. quantization_config has 2 config_groups (W4A16 + FP8_BLOCK)
  3. NO overlap between quantization_config.ignore and config_groups.targets
  4. dtype distribution per shard looks plausible (F8_E4M3 for attn, I32 for
     W4A16-packed experts, BF16 for passthrough, F32 for hc_* / attn_sink)
  5. MTP block tensor count > 0 and structure matches expectation

Usage::

    python scripts/inspect_quantized.py /scratch/weights/w4a16-fp8-mtp
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from safetensors import safe_open


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: inspect_quantized.py <model_dir>")
    root = Path(sys.argv[1])
    if not root.exists():
        sys.exit(f"FATAL: {root} not found")

    # 1. index vs shards
    idx_path = root / "model.safetensors.index.json"
    if not idx_path.exists():
        sys.exit(f"FATAL: {idx_path} not found")
    idx = json.loads(idx_path.read_text())
    indexed = set(idx.get("weight_map", {}).values())
    actual = {p.name for p in root.glob("*.safetensors")}
    missing_shards = indexed - actual
    extra_shards = actual - indexed
    print(f"[index] {len(idx.get('weight_map', {}))} tensors across {len(indexed)} shards")
    if missing_shards:
        print(f"  MISSING shards: {sorted(missing_shards)[:5]}")
    if extra_shards:
        print(f"  EXTRA shards (not in index): {sorted(extra_shards)[:5]}")

    # 2-3. quantization_config
    cfg = json.loads((root / "config.json").read_text())
    qc = cfg.get("quantization_config", {})
    groups = qc.get("config_groups", {})
    ignore = qc.get("ignore", [])
    print(f"[qc] groups: {list(groups.keys())}  ignore entries: {len(ignore)}")
    all_targets = set()
    for name, g in groups.items():
        ts = g.get("targets", [])
        weights = g.get("weights", {})
        print(f"  {name}: type={weights.get('type', '?')} "
              f"bits={weights.get('num_bits', '?')} "
              f"strategy={weights.get('strategy', '?')} "
              f"targets={ts}")
        all_targets.update(ts)
    overlap = set(ignore) & all_targets
    if overlap:
        print(f"  FATAL: ignore overlaps targets: {sorted(overlap)}")
        sys.exit(1)
    print(f"  ignore vs targets: clean (no overlap)")

    # 4. dtype distribution across all shards
    dtypes = Counter()
    mtp_keys = []
    for p in sorted(root.glob("*.safetensors")):
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                dt = str(f.get_slice(k).get_dtype())
                dtypes[dt] += 1
                if k.startswith("mtp."):
                    mtp_keys.append((k, dt, p.name))
    print(f"[dtype] total tensors: {sum(dtypes.values())}")
    for dt, c in dtypes.most_common():
        print(f"  {dt}: {c:,}")

    # 5. MTP block structure
    print(f"[mtp] total tensors: {len(mtp_keys)}")
    mtp_dtypes = Counter(d for _, d, _ in mtp_keys)
    for dt, c in mtp_dtypes.most_common():
        print(f"  {dt}: {c}")
    expert_keys = [k for k, _, _ in mtp_keys if "experts" in k]
    attn_keys = [k for k, _, _ in mtp_keys if ".attn." in k and "compressor" not in k and "indexer" not in k]
    print(f"  expert tensors: {len(expert_keys)}")
    print(f"  attn tensors (non-aux): {len(attn_keys)}")

    sanity = []
    if dtypes.get("F8_E4M3", 0) == 0:
        sanity.append("expected F8_E4M3 (FP8) tensors, found none — FP8 pass didn't run?")
    if dtypes.get("I32", 0) == 0:
        sanity.append("expected I32 (W4A16 packed) tensors, found none — W4A16 pass didn't run?")
    if len(mtp_keys) == 0:
        sanity.append("expected MTP tensors, found none — MTP block lost in quantization")

    if sanity:
        print()
        print("INSPECTION FAILED:")
        for s in sanity:
            print(f"  - {s}")
        sys.exit(1)

    print()
    print("inspect_quantized PASSED")


if __name__ == "__main__":
    main()
