#!/usr/bin/env bash
# Patch vllm/models/deepseek_v4/nvidia/ops/attention.py to route BF16 wo_a
# (e.g. the MTP block in Option-Y W4A16+FP8+MTP artifacts) through the
# rocm_inv_rope_einsum path instead of the FP8 einsum that needs scale.
#
# CRITICAL — dynamo-safe form: branches on `self.wo_a.weight.dtype ==
# torch.bfloat16`. An earlier version used `getattr(self.wo_a,
# "weight_scale", None)` which trips torch.compile / dynamo's
# `_getattr_static` (the lookup only inspects the TYPE, not the
# dynamically-registered weight_scale parameter on the instance) and
# forced `--enforce-eager`. The dtype check is constant-foldable at
# trace time, so cudagraphs work.
#
# Run AFTER bootstrap_rtx6000pro.sh and AFTER
# patch_v4_forcausal_packed_mapping.py.

set -uo pipefail
TARGET="${1:-$(python -c 'import vllm; print(vllm.__path__[0])')}/models/deepseek_v4/nvidia/ops/attention.py"

if [[ ! -f "$TARGET" ]]; then
    echo "ERROR: $TARGET not found" >&2
    exit 1
fi

python3 - "$TARGET" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1])
src = p.read_text()

old_block = '''        # Keep ROCm on the BF16 reference wo_a path util kernel ready.
        if current_platform.is_rocm():
            z = rocm_inv_rope_einsum(
                self.rotary_emb,
                o,
                positions,
                self.rope_head_dim,
                self.n_local_groups,
                self.o_lora_rank,
                self.wo_a,
            )
            return self.wo_b(z.flatten(1))'''

new_block = '''        # Keep ROCm on the BF16 reference wo_a path util kernel ready.
        # PATCH (paul/dsv4): also take BF16 path when wo_a is BF16
        # (e.g. MTP block in Option-Y W4A16+FP8+MTP artifact, where layer 43
        # is excluded from quantization). The .weight.dtype check is
        # dynamo-friendly (constant-fold at trace time), so torch.compile
        # + cudagraphs work — do NOT switch to getattr(self.wo_a,
        # "weight_scale", None) which trips dynamo's _getattr_static.
        if current_platform.is_rocm() or self.wo_a.weight.dtype == torch.bfloat16:
            z = rocm_inv_rope_einsum(
                self.rotary_emb,
                o,
                positions,
                self.rope_head_dim,
                self.n_local_groups,
                self.o_lora_rank,
                self.wo_a,
            )
            return self.wo_b(z.flatten(1))'''

if "self.wo_a.weight.dtype == torch.bfloat16" in src:
    print(f"{p.name}: already patched")
elif old_block in src:
    p.write_text(src.replace(old_block, new_block))
    print(f"{p.name}: PATCHED (dynamo-safe dtype check)")
else:
    print(f"{p.name}: ANCHOR MISSING", file=sys.stderr)
    sys.exit(1)
PYEOF
