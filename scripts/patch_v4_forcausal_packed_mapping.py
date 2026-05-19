#!/usr/bin/env python3
"""Add packed_modules_mapping to DeepseekV4ForCausalLM (post-refactor path).

vLLM PR #43004-#43077 (2026-05-19) moved deepseek_v4 from
``vllm/model_executor/models/deepseek_v4.py`` to
``vllm/models/deepseek_v4/nvidia/model.py``. The class definition and the
hf_to_vllm_mapper anchor are otherwise unchanged from the pre-refactor layout.

Why this patch is still needed: kylesayrs's PR #41276 references
``self.packed_modules_mapping`` (used by is_layer_skipped() and the
compressed-tensors loader to resolve fused module names) but never defines it
on the class. Without this attribute, FP8_BLOCK on attention fails to load
with:
    ValueError: Unable to find matching target for
    model.layers.0.attn.fused_wqa_wkv in the compressed-tensors config.

Idempotent — re-running on an already-patched tree exits cleanly.

Usage:
    python scripts/patch_v4_forcausal_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
"""
import sys
from pathlib import Path

ANCHOR = '''class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")'''

REPLACEMENT = '''class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    # PATCH (paul/dsv4): mapping from fused module names to their constituent
    # shard names. Used by is_layer_skipped() and the compressed-tensors loader
    # to determine the quantization scheme for fused layers (which are constructed
    # at vLLM init from the underlying ColumnParallelLinear shards). Without this,
    # FP8_BLOCK on attn fails with "Unable to find matching target for
    # model.layers.0.attn.fused_wqa_wkv".
    packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["w1", "w3"],
    }'''


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_pkg_root>", file=sys.stderr)
        sys.exit(2)

    vllm_root = Path(sys.argv[1])
    target = vllm_root / "models" / "deepseek_v4" / "nvidia" / "model.py"

    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        sys.exit(1)

    src = target.read_text()

    if "packed_modules_mapping" in src and "fused_wqa_wkv" in src:
        print(f"already patched: {target}")
        return

    if ANCHOR not in src:
        print(f"ERROR: anchor not found in {target}", file=sys.stderr)
        sys.exit(1)

    target.write_text(src.replace(ANCHOR, REPLACEMENT))
    print(f"patched: {target}")


if __name__ == "__main__":
    main()
