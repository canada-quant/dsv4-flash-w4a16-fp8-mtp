#!/usr/bin/env python3
"""Rewrite ``quantization_config.ignore`` in config.json after the rename pass.

After llm-compressor finishes, ``config.json`` lists the modules excluded from
quantization under HF naming (model.layers.X.self_attn.q_a_proj, etc). vLLM
loads with internal naming (layers.X.attn.wq_a, etc). Without this rewrite,
the W4A16 loader emits "Unable to find matching target" errors.

Renames both:
  * the main-model paths (model.layers.X.*) — same logic as the predecessor's
    dsv4-flash-w4a16-fp8 script
  * the MTP paths (mtp.layers.43.*, model.mtp.layers.43.*) — new for this
    repo, because the MTP block is now included in calibration and shows up
    in the ignore list

BUGFIX vs reasoning-agent's port: the reasoning-agent repo's version of this
script uses ``json.dumps(config, f, indent=2)`` on line 53, which returns the
serialized string instead of writing it (json.dumps takes no file arg). The
result: config.json is never written, vLLM silently loads with stale ignore
entries, and MTP submodules fail. We use ``json.dump(config, f, indent=2)``.

Idempotent: re-running on already-renamed entries is a no-op (the renames are
guarded by anchor patterns that only match HF naming).

Usage:
    python scripts/patch_ignore_list.py --config ./weights/w4a16-fp8-mtp/config.json
"""
import argparse
import json
import re
from pathlib import Path


def rename(p: str) -> str:
    """HF parameter path -> vLLM internal path."""
    if p == "lm_head":
        return "lm_head"

    # Strip "model." prefix for main-model paths but preserve the "mtp." root.
    # Upstream HF stores MTP under model.mtp.X; after stripping the leading
    # "model." we get mtp.X, which is what the vLLM loader expects.
    if p.startswith("model."):
        p = p[len("model."):]

    p = p.replace(".self_attn.", ".attn.")
    p = p.replace(".mlp.", ".ffn.")
    p = p.replace(".shared_experts.gate_proj", ".shared_experts.w1")
    p = p.replace(".shared_experts.up_proj", ".shared_experts.w3")
    p = p.replace(".shared_experts.down_proj", ".shared_experts.w2")
    p = re.sub(r"\.attn\.kv_proj$", ".attn.wkv", p)
    p = re.sub(r"\.attn\.kv_a_proj_with_mqa$", ".attn.wkv_a", p)
    p = re.sub(r"\.attn\.kv_b_proj$", ".attn.wkv_b", p)
    p = re.sub(r"\.attn\.q_proj$", ".attn.wq", p)
    p = re.sub(r"\.attn\.q_a_proj$", ".attn.wq_a", p)
    p = re.sub(r"\.attn\.q_b_proj$", ".attn.wq_b", p)
    p = re.sub(r"\.attn\.o_proj$", ".attn.wo", p)
    p = re.sub(r"\.attn\.o_a_proj$", ".attn.wo_a", p)
    p = re.sub(r"\.attn\.o_b_proj$", ".attn.wo_b", p)
    p = re.sub(r"\.ffn\.gate_proj$", ".ffn.w1", p)
    p = re.sub(r"\.ffn\.up_proj$", ".ffn.w3", p)
    p = re.sub(r"\.ffn\.down_proj$", ".ffn.w2", p)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to model config.json")
    args = ap.parse_args()

    path = Path(args.config)
    config = json.loads(path.read_text())
    qc = config.get("quantization_config", {})
    old = qc.get("ignore", [])
    new = [rename(p) for p in old]
    qc["ignore"] = new
    config["quantization_config"] = qc

    with path.open("w") as f:
        # NOTE: json.dump, not json.dumps. Calling json.dumps with a file arg
        # silently produces the string and never writes (see docstring).
        json.dump(config, f, indent=2)

    changed = sum(1 for a, b in zip(old, new) if a != b)
    print(f"updated {len(new)} ignore entries ({changed} renamed)")
    for a, b in zip(old[:6], new[:6]):
        marker = "  " if a == b else "->"
        print(f"  {a}\n{marker}{b}")


if __name__ == "__main__":
    main()
