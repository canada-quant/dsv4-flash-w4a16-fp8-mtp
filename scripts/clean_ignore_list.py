#!/usr/bin/env python3
"""Phase 3 — clean up duplicate entries between quantization_config.ignore
and config_groups[*].targets.

Running model_free_ptq twice (W4A16 experts pass, then FP8_BLOCK attn pass)
causes pass 2 to write a config.json where pass-1's ``ignore`` list (which
deliberately included the attention/MTP-projection targets so they were
skipped during pass 1) leaked into the final ``ignore`` even though those
same names appear in ``config_groups[config_group_1].targets``.

vLLM's compressed-tensors loader semantics on this overlap are undefined —
some versions take config_group precedence (correct quantization), others
take ignore precedence (load as BF16, then fail at runtime when fused
shards expect quantized weights). Cleanest fix is to strip any ignore
entry that also appears as a target.

Idempotent. Modifies config.json in place; backs up to config.json.bak on
first run.

Usage::

    python scripts/clean_ignore_list.py --config /scratch/weights/w4a16-fp8-mtp/config.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to model config.json")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"FATAL: {path} not found")

    config = json.loads(path.read_text())
    qc = config.get("quantization_config", {})
    ignore = qc.get("ignore", [])
    groups = qc.get("config_groups", {})

    # Gather all targets across all groups
    all_targets: set[str] = set()
    for grp_name, grp in groups.items():
        for t in grp.get("targets", []):
            all_targets.add(t)

    # Remove ignore entries that match a target string exactly
    cleaned = [i for i in ignore if i not in all_targets]
    removed = [i for i in ignore if i in all_targets]

    if not removed:
        print(f"no overlap between ignore and config_group targets ({len(ignore)} ignore, "
              f"{len(all_targets)} targets) — nothing to clean")
        return

    # Backup once
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copyfile(path, bak)
        print(f"backed up original to {bak}")

    qc["ignore"] = cleaned
    config["quantization_config"] = qc
    path.write_text(json.dumps(config, indent=2))

    print(f"removed {len(removed)} duplicate entries from ignore:")
    for r in removed:
        print(f"  - {r}")
    print(f"ignore length: {len(ignore)} -> {len(cleaned)}")
    print(f"config_group targets: {sorted(all_targets)}")


if __name__ == "__main__":
    main()
