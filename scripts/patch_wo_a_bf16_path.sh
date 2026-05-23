#!/bin/bash
# Replace the FP8 wo_a fast path with a check: if wo_a has no scale (BF16
# from MTP block), fall through to the BF16 reference path used on ROCm.
F=/home/ubuntu/venv-serve/lib/python3.10/site-packages/vllm/models/deepseek_v4/nvidia/ops/attention.py
python3 <<'PYEOF'
from pathlib import Path
p = Path("/home/ubuntu/venv-serve/lib/python3.10/site-packages/vllm/models/deepseek_v4/nvidia/ops/attention.py")
src = p.read_text()
# Replace our current debug patch with the proper fallback that uses
# rocm_inv_rope_einsum when wo_a has no scale (i.e. BF16 unquantized MTP).
old = '''        # Keep ROCm on the BF16 reference wo_a path util kernel ready.
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
new = '''        # Keep ROCm on the BF16 reference wo_a path util kernel ready.
        # PATCH (paul/dsv4): also take BF16 path when wo_a has no scale
        # (e.g. MTP block in Option-Y W4A16+FP8+MTP artifact, where layer 43
        # is excluded from quantization and wo_a is plain BF16).
        wo_a_has_scale = (
            getattr(self.wo_a, "weight_scale_inv", None) is not None
            or getattr(self.wo_a, "weight_scale", None) is not None
        )
        if current_platform.is_rocm() or not wo_a_has_scale:
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
if old in src:
    src = src.replace(old, new)

# Strip the previous debug raise block, replace with simple wo_a_scale assignment
old2 = '''        wo_a_fp8 = self.wo_a.weight
        # PATCH (paul/dsv4): fallback to weight_scale if no _inv suffix
        wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
        if wo_a_scale is None:
            wo_a_scale = getattr(self.wo_a, "weight_scale", None)
        if wo_a_scale is None:
            import sys
            attrs = [a for a in dir(self.wo_a) if not a.startswith("_") and ("weight" in a.lower() or "scale" in a.lower())]
            print(f"[debug-rt] wo_a has no scale; attrs={attrs}  type={type(self.wo_a).__name__}", flush=True, file=sys.stderr)
            print(f"[debug-rt] wo_a.weight.dtype={self.wo_a.weight.dtype}  shape={tuple(self.wo_a.weight.shape)}", flush=True, file=sys.stderr)
            print(f"[debug-rt] params dict: {list(self.wo_a._parameters.keys())}", flush=True, file=sys.stderr)
            print(f"[debug-rt] buffers dict: {list(self.wo_a._buffers.keys())}", flush=True, file=sys.stderr)
            raise AttributeError("wo_a has no weight_scale or weight_scale_inv")'''
new2 = '''        wo_a_fp8 = self.wo_a.weight
        # PATCH (paul/dsv4): fallback to weight_scale if no _inv suffix
        wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
        if wo_a_scale is None:
            wo_a_scale = self.wo_a.weight_scale'''
if old2 in src:
    src = src.replace(old2, new2)

p.write_text(src)
print("patched")
PYEOF

pkill -9 -f vllm 2>/dev/null
sleep 3
cd /opt/dlami/nvme/dsv4-flash-w4a16-fp8-mtp
CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/w4a16-fp8-mtp-gptq 8000 2 \
    > /tmp/serve_tp2.log 2>&1 &
disown
sleep 8
tail -5 /tmp/serve_tp2.log
