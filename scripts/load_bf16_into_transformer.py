#!/usr/bin/env python3
"""Load the Phase-1 BF16 checkpoint into the adapted upstream Transformer.

Phase 1 writes 46 safetensors shards at ``/scratch/weights/bf16-mtp`` using
DeepSeek's *internal* parameter naming convention (verified 2026-05-19 —
see ``memory:dsv4_naming_convention``):

  layers.X.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight
  layers.X.ffn.experts.Y.{w1,w2,w3}.weight
  layers.X.{attn_norm,ffn_norm,...}.weight
  layers.X.hc_{attn,ffn}_{fn,base,scale}
  mtp.0.X (same shape as main layers + e_proj/h_proj/enorm/hnorm/norm)
  embed.weight  (or embed.tok_emb.weight — see notes)
  head.weight   (or shared_head.head.weight)
  norm.weight   (or shared_head.norm.weight)
  hc_head_{fn,base,scale}

The adapted upstream Transformer's parameter paths follow the same
convention almost verbatim — apart from possible (embed | embed_tokens),
(head | shared_head.head), (norm | shared_head.norm) discrepancies which
``upstream/convert.py`` documents.

This loader iterates every safetensors tensor, attempts direct assignment
into the Transformer's state_dict, applies a small rename map for the
known divergences, and reports unmatched keys.

CLI::

    python scripts/load_bf16_into_transformer.py \\
        --weights /scratch/weights/bf16-mtp \\
        --config  vendor/dsv4-upstream/config.json \\
        [--dump-state-dict /tmp/sd.json]      # debug
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

# Side-effect-importing scripts.upstream installs the kernel/fht stubs.
from scripts.upstream import Transformer, build_model_args


# Map any safetensors key that does NOT directly match a Transformer
# state_dict key. Suffix-based, applied last-match-wins on each tensor name.
# Anchor patterns are intentionally narrow so we don't accidentally rename
# layer-internal tensors.
RENAME_RULES = [
    # The main model embedding / head / norm are shared with MTP. Upstream's
    # convert.py describes the canonical remapping; we apply the inverse here
    # for whatever survives the Phase-1 dequant.
    ("emb.tok_emb.weight",  "embed.weight"),
    ("embed_tokens.weight", "embed.weight"),
    ("shared_head.head.weight", "head.weight"),
    ("shared_head.norm.weight", "norm.weight"),
]

# Parameter aliases: nn.Module attribute assignment makes mtp[0].embed and
# self.embed reference the same Parameter; state_dict() lists both names but
# loading one implicitly loads the other. Treat the second name as already
# satisfied once the first is in the safetensors.
PARAM_ALIASES = {
    "mtp.0.embed.weight": "embed.weight",
    "mtp.0.head.weight": "head.weight",
}


def maybe_rename(name: str) -> str:
    for src, dst in RENAME_RULES:
        if name.endswith(src):
            return name[: -len(src)] + dst
    return name


def load_safetensors_into(
    transformer: Transformer,
    weights_dir: Path,
    *,
    verbose: bool = False,
) -> tuple[int, list[str], list[str]]:
    """Returns (n_loaded, unmatched_safetensors, missing_params)."""
    state = transformer.state_dict()
    state_keys = set(state.keys())

    loaded = 0
    unmatched: list[str] = []
    for shard in sorted(weights_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for raw_name in f.keys():
                name = maybe_rename(raw_name)
                if name not in state_keys:
                    unmatched.append(raw_name)
                    continue
                tensor = f.get_tensor(raw_name)
                target = state[name]
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"shape mismatch loading {raw_name} -> {name}: "
                        f"checkpoint {list(tensor.shape)} vs model {list(target.shape)}"
                    )
                # Cast if dtypes differ (e.g. RMSNorm weight in upstream is
                # fp32; the safetensors carry bf16 — F.linear casts at fwd
                # time but the param itself should match).
                if tensor.dtype != target.dtype:
                    tensor = tensor.to(target.dtype)
                target.copy_(tensor)
                loaded += 1
                if verbose and loaded % 5000 == 0:
                    print(f"  loaded {loaded} tensors...", flush=True)

    # 'missing' = a model param whose name isn't covered by any safetensors
    # key (after rename) AND isn't a known alias.
    seen_state_keys: set[str] = set()
    for s in weights_dir.glob("*.safetensors"):
        with safe_open(s, framework="pt", device="cpu") as f:
            for raw in f.keys():
                seen_state_keys.add(maybe_rename(raw))
    seen_state_keys.update(
        alias for alias, src in PARAM_ALIASES.items() if src in seen_state_keys
    )
    missing = sorted(state_keys - seen_state_keys)
    return loaded, unmatched, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Phase-1 BF16 dir")
    ap.add_argument("--config", required=True,
                    help="upstream config.json (vendor/dsv4-upstream/config.json)")
    ap.add_argument("--max-batch-size", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--dump-state-dict", default=None,
                    help="write the Transformer's state_dict keys to this JSON for debugging")
    ap.add_argument("--probe-forward", action="store_true",
                    help="after load, run a small forward and print logits stats")
    args = ap.parse_args()

    print("[args] building ModelArgs from upstream config.json")
    margs = build_model_args(
        args.config, max_batch_size=args.max_batch_size, max_seq_len=args.max_seq_len
    )
    print(f"  vocab_size={margs.vocab_size}  dim={margs.dim}  n_layers={margs.n_layers}  "
          f"n_routed_experts={margs.n_routed_experts}  n_mtp_layers={margs.n_mtp_layers}")

    print("[model] instantiating Transformer (BF16, world_size=1)")
    torch.set_default_dtype(torch.bfloat16)
    transformer = Transformer(margs)
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"  parameters: {n_params:,} ({n_params * 2 / 1e9:.2f} GB BF16)")

    if args.dump_state_dict:
        sd_keys = sorted(transformer.state_dict().keys())
        Path(args.dump_state_dict).write_text(json.dumps(sd_keys, indent=2))
        print(f"  dumped state_dict keys to {args.dump_state_dict}")

    print(f"[load] reading safetensors from {args.weights}")
    loaded, unmatched, missing = load_safetensors_into(
        transformer, Path(args.weights), verbose=True
    )
    print(f"[load] loaded {loaded} tensors")
    print(f"[load] unmatched (in safetensors, not in model): {len(unmatched)}")
    if unmatched and len(unmatched) <= 30:
        for k in unmatched[:30]:
            print(f"  - {k}")
    elif unmatched:
        for k in unmatched[:15]:
            print(f"  - {k}")
        print(f"  ... ({len(unmatched) - 15} more)")
    print(f"[load] missing (in model, not in safetensors): {len(missing)}")
    if missing and len(missing) <= 30:
        for k in missing[:30]:
            print(f"  - {k}")
    elif missing:
        for k in missing[:15]:
            print(f"  - {k}")
        print(f"  ... ({len(missing) - 15} more)")

    if args.probe_forward:
        print("[probe] running forward on a small dummy input")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transformer = transformer.to(device)
        x = torch.randint(0, margs.vocab_size, (1, 8), device=device)
        with torch.inference_mode():
            logits = transformer(x)
        print(f"  logits dtype={logits.dtype}  shape={list(logits.shape)}")
        print(f"  finite: {torch.isfinite(logits).all().item()}")
        print(f"  min={logits.min().item():.3f}  max={logits.max().item():.3f}  "
              f"mean-abs={logits.abs().mean().item():.3f}")

    if unmatched:
        sys.exit(2)


if __name__ == "__main__":
    main()
