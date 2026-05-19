#!/usr/bin/env python3
"""Add packed_modules_mapping to DeepSeekV4MTP class.

The MTP class (used by speculative decoding) needs the same
packed_modules_mapping that the main ForCausalLM class gets, so the
compressed-tensors loader can resolve fused attention names like
fused_wqa_wkv and fused_wkv_wgate.

Pre-refactor path:  vllm/model_executor/models/deepseek_v4_mtp.py
Post-refactor path: vllm/models/deepseek_v4/nvidia/mtp.py
                    (PR #43004-#43077 split nvidia/ vs amd/)

DIFFERENCE FROM REASONING-AGENT'S patch_mtp_mapping.py:
That older script also globally replaced ``.weight_scale_inv`` with
``.weight_scale``. The post-refactor mtp.py (mtp.py:357-389) already chooses
``.weight_scale`` vs ``.weight_scale_inv`` per-tensor based on FP4 vs FP8
expert dtype. Running a global replace now would break FP8 expert loading.
This script only adds the mapping.

Idempotent.

Usage:
    python scripts/patch_mtp_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
"""
import re
import sys
from pathlib import Path

PATCH_BLOCK = '''
    # PATCH (paul/dsv4): mapping from fused module names to their constituent
    # shard names. Mirrors the same attribute on DeepseekV4ForCausalLM. Without
    # this, compressed-tensors W4A16+FP8_BLOCK loading of MTP attention fails
    # at vLLM init with "Unable to find matching target for ...fused_wqa_wkv".
    packed_modules_mapping = {
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
        "gate_up_proj": ["w1", "w3"],
    }
'''


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_pkg_root>", file=sys.stderr)
        sys.exit(2)

    vllm_root = Path(sys.argv[1])
    target = vllm_root / "models" / "deepseek_v4" / "nvidia" / "mtp.py"

    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        sys.exit(1)

    src = target.read_text()

    if "packed_modules_mapping" in src and "fused_wqa_wkv" in src:
        print(f"already patched: {target}")
        return

    # Match `class DeepSeekV4MTP(...):` line — insert PATCH_BLOCK right after.
    m = re.search(r"^class DeepSeekV4MTP\b[^:]*:\s*\n", src, re.MULTILINE)
    if not m:
        print(f"ERROR: class DeepSeekV4MTP not found in {target}", file=sys.stderr)
        sys.exit(1)

    insert_at = m.end()
    new_src = src[:insert_at] + PATCH_BLOCK + src[insert_at:]
    target.write_text(new_src)
    print(f"patched: {target}")


if __name__ == "__main__":
    main()
