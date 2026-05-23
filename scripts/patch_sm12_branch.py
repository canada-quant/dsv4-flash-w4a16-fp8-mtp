#!/usr/bin/env python3
"""Apply packed_modules_mapping patches to the jasl/vllm@ds4-sm120-experimental
branch's pre-refactor file layout (vllm/model_executor/models/deepseek_v4.py
and deepseek_v4_mtp.py).

Equivalent to running both patch_v4_forcausal_packed_mapping.py and
patch_mtp_packed_mapping.py from this repo, but targets the OLDER file paths
that exist in the SM12 branch (the upstream refactor #43004-#43077 moved
these to vllm/models/deepseek_v4/nvidia/{model,mtp}.py post-2026-05-19).

Usage:
    python scripts/patch_sm12_branch.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
"""
import sys
from pathlib import Path


V4_ANCHOR = '''class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")'''

V4_REPLACEMENT = V4_ANCHOR + '''

    # PATCH (paul/dsv4): mapping from fused module names to constituent shards.
    # Required by is_layer_skipped() and compressed-tensors loader to resolve
    # fused module quant schemes. Without this, FP8_BLOCK attn fails with
    # "Unable to find matching target for model.layers.0.attn.fused_wqa_wkv".
    packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["w1", "w3"],
    }'''


MTP_ANCHOR = '''class DeepSeekV4MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):'''

MTP_REPLACEMENT = '''class DeepSeekV4MTP(nn.Module):
    # PATCH (paul/dsv4): packed_modules_mapping mirrors DeepseekV4ForCausalLM
    # so the compressed-tensors scheme resolution finds fused MTP attn modules.
    packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["w1", "w3"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):'''


def patch_file(path: Path, anchor: str, replacement: str, marker: str) -> str:
    if not path.exists():
        return f"{path}: NOT FOUND"
    src = path.read_text()
    if marker in src:
        return f"{path.name}: already patched"
    if anchor not in src:
        return f"{path.name}: ANCHOR MISSING"
    path.write_text(src.replace(anchor, replacement))
    return f"{path.name}: PATCHED"


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_pkg_root>", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1])
    mods = root / "model_executor" / "models"
    print(patch_file(mods / "deepseek_v4.py", V4_ANCHOR, V4_REPLACEMENT, "PATCH (paul/dsv4):"))
    print(patch_file(mods / "deepseek_v4_mtp.py", MTP_ANCHOR, MTP_REPLACEMENT, "PATCH (paul/dsv4):"))


if __name__ == "__main__":
    main()
