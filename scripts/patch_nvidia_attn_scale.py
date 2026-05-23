"""Patch nvidia/ops/attention.py:370 to fall back to weight_scale when
weight_scale_inv is absent.

The artifact has `wo_a.weight_scale` (no `_inv`) — compressed-tensors
W8A8Fp8 path uses this name. PR #43290 added a fallback to the shared
attention.py but preview-dev has a separate nvidia/ops/attention.py
that needs the same patch.
"""
import sys
from pathlib import Path

ANCHOR = "        wo_a_scale = self.wo_a.weight_scale_inv"
REPLACEMENT = (
    "        # PATCH (paul/dsv4): fallback to weight_scale if no _inv suffix\n"
    "        wo_a_scale = getattr(self.wo_a, \"weight_scale_inv\", None)\n"
    "        if wo_a_scale is None:\n"
    "            wo_a_scale = self.wo_a.weight_scale"
)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: <script> <vllm_pkg_root>")
    target = (Path(sys.argv[1]) / "models" / "deepseek_v4" / "nvidia" /
              "ops" / "attention.py")
    if not target.exists():
        sys.exit(f"missing: {target}")
    src = target.read_text()
    if "PATCH (paul/dsv4): fallback to weight_scale" in src:
        print(f"{target.name}: already patched")
        return
    if ANCHOR not in src:
        sys.exit(f"{target.name}: ANCHOR MISSING")
    target.write_text(src.replace(ANCHOR, REPLACEMENT))
    print(f"{target.name}: PATCHED")


if __name__ == "__main__":
    main()
