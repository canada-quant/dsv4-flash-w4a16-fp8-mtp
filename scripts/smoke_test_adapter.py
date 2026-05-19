#!/usr/bin/env python3
"""Tiny-model smoke test for the upstream adapter.

Confirms that the side-effect-imported scripts.upstream package:
  * imports vendor/dsv4-upstream/model.py without tilelang
  * rebinds Linear to an nn.Linear subclass (so GPTQ sees calibration targets)
  * runs a forward pass through a small Transformer + MTP block without
    crashing in the kernel_shim's sparse_attn / hc_split_sinkhorn / etc.

If this passes, the same adapter applied to the full 43-layer / 256-expert
model is expected to work (modulo memory).
"""
import sys
from pathlib import Path

# Make `scripts.upstream` importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from scripts.upstream import Transformer, ModelArgs


def main():
    margs = ModelArgs(
        max_batch_size=2,
        max_seq_len=128,
        dtype="bf16",
        vocab_size=1024,
        dim=256,
        moe_inter_dim=256,
        n_layers=2,
        n_hash_layers=0,
        n_mtp_layers=1,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=128,
        head_dim=64,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=128,
        window_size=32,
        compress_ratios=(0, 0, 0),  # 2 main + 1 mtp; no compression for smoke test
        index_n_heads=4,
        index_head_dim=16,
        index_topk=16,
    )
    print(f"args: dim={margs.dim} layers={margs.n_layers} experts={margs.n_routed_experts}")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = Transformer(margs)

    # Initialize all parameters with small Gaussian (Transformer.__init__ uses
    # torch.empty, which leaves uninitialized memory that NaNs out in a few
    # softmax layers).
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
        else:
            torch.nn.init.zeros_(p)
    # RMSNorm weights should start near 1
    for m in model.modules():
        if type(m).__name__ == "RMSNorm":
            torch.nn.init.ones_(m.weight)
    # attn_sink: standard init in attention papers is 0 (we already zeroed)
    print("initialized parameters")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    print(f"nn.Linear instances: {len(linear_modules)}")
    print(f"first 5 Linear shapes:")
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            print(f"  {n:60s}  in={m.in_features} out={m.out_features}")
            if "experts.0.w1" in n:
                break

    print("forward (main):")
    input_ids = torch.randint(0, margs.vocab_size, (1, 16))
    with torch.inference_mode():
        logits = model(input_ids)
    print(f"  logits.shape={list(logits.shape)} dtype={logits.dtype}")
    print(f"  finite={torch.isfinite(logits).all().item()} "
          f"min={logits.min().item():.3f} max={logits.max().item():.3f}")
    if not torch.isfinite(logits).all():
        print("FAIL: non-finite logits")
        sys.exit(1)

    print("forward (MTP):")
    h = torch.randn(1, 16, margs.hc_mult, margs.dim)
    with torch.inference_mode():
        mtp_logits = model.mtp[0](h, 0, input_ids)
    print(f"  mtp_logits.shape={list(mtp_logits.shape)} dtype={mtp_logits.dtype}")
    print(f"  finite={torch.isfinite(mtp_logits).all().item()} "
          f"min={mtp_logits.min().item():.3f} max={mtp_logits.max().item():.3f}")
    if not torch.isfinite(mtp_logits).all():
        print("FAIL: non-finite mtp_logits")
        sys.exit(1)

    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
