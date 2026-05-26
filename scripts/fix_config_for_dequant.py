#!/usr/bin/env python3
"""Fix Card D config.json: remove compressor + indexer regex from group_0 targets
(those layers were dequantized to nothing during the shipping-bug fix), and add
defensive explicit ignore patterns for any lingering quant_method references.

Also fix the broader naming issue: the artifact uses `.weight_scale` everywhere
(no `_inv` suffix), but vLLM's FP8 block path expects `_inv`. We can address that
either via vLLM patch upstream or by also marking the entire attention FP8 group
as ignored (forcing BF16 attention).

CONSERVATIVE PLAN: only remove compressor/indexer targets in this pass. If the
patched build still crashes on main-attn weight_scale_inv, we'll then file the
defensive vLLM PR. That way the config.json change is surgical."""
import json
import shutil
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/.cache/huggingface/hub/models--canada-quant--DeepSeek-V4-Flash-W4A16-FP8-MTP/snapshots/c0b7e8700282b618bda250c7007e098dffa9883c/config.json"
out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/config_fixed.json"

with open(src) as f:
    d = json.load(f)

qc = d["quantization_config"]
g0 = qc["config_groups"]["group_0"]

old_targets = list(g0["targets"])
# Keep only the main-attention regex; drop compressor + indexer regexes since
# those sub-modules have no weights in the dequant'd artifact.
new_targets = [
    t for t in old_targets
    if "compressor" not in t and "indexer" not in t
]

print("=== group_0 targets ===")
print("BEFORE:", old_targets)
print("AFTER:", new_targets)
g0["targets"] = new_targets

# Add explicit ignores too — belt-and-suspenders. Use the exact module paths
# that would have been matched. Layer count is 61 for DSv4-Flash.
existing_ignore = list(qc.get("ignore", []))
new_ignore_patterns = []
# Note: we use explicit per-layer paths rather than regex because the loader
# is more deterministic about matching exact names.
for layer_idx in range(61):
    # compressor sub-modules (dequant'd, no weights present)
    new_ignore_patterns.extend([
        f"layers.{layer_idx}.attn.compressor.wgate",
        f"layers.{layer_idx}.attn.compressor.wkv",
        f"layers.{layer_idx}.attn.compressor.fused_wkv_wgate",
        f"layers.{layer_idx}.attn.compressor.gate_proj",
        f"layers.{layer_idx}.attn.compressor.kv_proj",
        # indexer sub-modules
        f"layers.{layer_idx}.attn.indexer.weights_proj",
        f"layers.{layer_idx}.attn.indexer.wq_b",
        f"layers.{layer_idx}.attn.indexer.q_b_proj",
        f"layers.{layer_idx}.attn.indexer.compressor.wgate",
        f"layers.{layer_idx}.attn.indexer.compressor.wkv",
        f"layers.{layer_idx}.attn.indexer.compressor.gate_proj",
        f"layers.{layer_idx}.attn.indexer.compressor.kv_proj",
    ])

qc["ignore"] = existing_ignore + new_ignore_patterns
print(f"\nIgnore list: was {len(existing_ignore)}, now {len(qc['ignore'])} entries")
print("Sample new ignores:")
for x in new_ignore_patterns[:6]:
    print(f"  {x}")

# Write
with open(out, "w") as f:
    json.dump(d, f, indent=2)

print(f"\n[done] wrote {out}")
