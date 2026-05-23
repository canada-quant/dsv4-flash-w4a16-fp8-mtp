"""Dequantize the compressor/indexer.weights_proj/indexer.wq_b weights from
FP8 (with BF16 block scales) to BF16 in place in the artifact shards.

Why: jasl/vllm's preview-dev (and upstream main) construct compressor's
fused_wkv_wgate, indexer.weights_proj, and indexer.compressor's
fused_wkv_wgate with quant_config=None — i.e. as unquantized BF16
modules. Our W4A16+FP8+MTP artifact calibrated these as FP8_BLOCK, so
the safetensors shards have `.weight_scale` keys + FP8 `.weight` keys
that don't fit into an unquantized module.

Fix: precompute the dequantized BF16 weight = fp8_to_bf16(weight) * scale
once, write it back as the .weight key (BF16), and drop the .weight_scale
key. The unquantized module then loads normally.

Modules dequantized:
  - layers.{i}.attn.compressor.{wkv,wgate}.weight + .weight_scale → BF16 .weight
  - layers.{i}.attn.indexer.weights_proj.weight + .weight_scale → BF16 .weight
  - layers.{i}.attn.indexer.wq_b.weight + .weight_scale → BF16 .weight
  - layers.{i}.attn.indexer.compressor.{wkv,wgate}.weight + .weight_scale → BF16 .weight

Block size: 128 (from quantization_config.config_groups).

Idempotent: detects already-dequantized files via dtype check.

Usage:
    python dequant_compressor.py /scratch/weights/w4a16-fp8-mtp-gptq
"""
import json
import sys
from pathlib import Path

import torch
import safetensors.torch as st


def dequant_block_fp8(weight_fp8: torch.Tensor, weight_scale: torch.Tensor,
                       block_size: int = 128) -> torch.Tensor:
    """Dequantize FP8_BLOCK weights using block-scale.

    weight_fp8: (M, K) fp8_e4m3fn
    weight_scale: (M // block_size, K // block_size) bf16 (or fp32)
    Returns: (M, K) bf16
    """
    w_fp32 = weight_fp8.float()
    M, K = w_fp32.shape
    bM = (M + block_size - 1) // block_size
    bK = (K + block_size - 1) // block_size
    assert weight_scale.shape == (bM, bK), (
        f"scale shape {weight_scale.shape} != expected ({bM}, {bK}) for "
        f"weight {weight_fp8.shape} with block_size={block_size}"
    )
    # Expand scale to (M, K)
    scale_expanded = weight_scale.float().repeat_interleave(
        block_size, dim=0).repeat_interleave(block_size, dim=1)
    # Trim if M or K isn't a multiple of block_size
    scale_expanded = scale_expanded[:M, :K]
    out_fp32 = w_fp32 * scale_expanded
    return out_fp32.to(torch.bfloat16)


def is_compressor_or_indexer_target(name: str) -> bool:
    """Module names whose .weight needs dequantizing to BF16."""
    targets = [
        ".attn.compressor.wkv.weight",
        ".attn.compressor.wgate.weight",
        ".attn.indexer.weights_proj.weight",
        ".attn.indexer.wq_b.weight",
        ".attn.indexer.compressor.wkv.weight",
        ".attn.indexer.compressor.wgate.weight",
    ]
    return any(name.endswith(t) for t in targets)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: dequant_compressor.py <artifact_dir>")
    art = Path(sys.argv[1])
    idx_path = art / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]
    total_size = idx.get("metadata", {}).get("total_size", 0)

    # Group keys by shard
    shard_to_keys: dict[str, list[str]] = {}
    for k, shard in wm.items():
        shard_to_keys.setdefault(shard, []).append(k)

    n_dequant = 0
    new_wm = dict(wm)
    new_total_size = total_size

    for shard_name in sorted(shard_to_keys.keys()):
        shard_path = art / shard_name
        print(f"[shard] {shard_name}: loading...", flush=True)
        with st.safe_open(shard_path, framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in shard_to_keys[shard_name]}

        modified = False
        new_tensors: dict[str, torch.Tensor] = {}
        scale_keys_to_drop = []

        for k in sorted(tensors.keys()):
            t = tensors[k]
            if is_compressor_or_indexer_target(k) and t.dtype == torch.float8_e4m3fn:
                # Use rsplit so "indexer.weights_proj.weight" → ".weight_scale"
                # at the suffix only (not at the "weights" substring).
                scale_key = k.rsplit(".weight", 1)[0] + ".weight_scale"
                if scale_key not in tensors:
                    # Could be in another shard; resolve from index
                    scale_shard = wm.get(scale_key)
                    if scale_shard is None:
                        print(f"  WARN: {k} is FP8 but no {scale_key} found", flush=True)
                        new_tensors[k] = t
                        continue
                    if scale_shard == shard_name:
                        # already in this shard
                        scale = tensors[scale_key]
                    else:
                        with st.safe_open(art / scale_shard, framework="pt") as sf:
                            scale = sf.get_tensor(scale_key)
                else:
                    scale = tensors[scale_key]
                bf16 = dequant_block_fp8(t, scale, block_size=128)
                new_tensors[k] = bf16
                scale_keys_to_drop.append(scale_key)
                # Update total_size (delta: t bytes + scale bytes → bf16 bytes)
                old_bytes = t.element_size() * t.nelement() + scale.element_size() * scale.nelement()
                new_bytes = bf16.element_size() * bf16.nelement()
                new_total_size += (new_bytes - old_bytes)
                n_dequant += 1
                modified = True
                print(f"  dequant: {k}  ({t.dtype} {tuple(t.shape)} → bf16)", flush=True)
            elif k.endswith(".weight_scale") and is_compressor_or_indexer_target(
                    k.rsplit(".weight_scale", 1)[0] + ".weight"):
                # This scale was consumed by dequant of a sibling .weight
                # in this or another shard; drop it
                if k not in scale_keys_to_drop:
                    scale_keys_to_drop.append(k)
                modified = True
            else:
                new_tensors[k] = t

        for sk in scale_keys_to_drop:
            new_tensors.pop(sk, None)
            new_wm.pop(sk, None)

        if modified:
            print(f"  saving {len(new_tensors)} tensors ({len(scale_keys_to_drop)} dropped)", flush=True)
            st.save_file(new_tensors, shard_path)
        else:
            print(f"  no change", flush=True)

    # Update index
    new_idx = dict(idx)
    new_idx["weight_map"] = new_wm
    if "metadata" not in new_idx:
        new_idx["metadata"] = {}
    new_idx["metadata"]["total_size"] = new_total_size
    idx_path.write_text(json.dumps(new_idx, indent=2))
    print(f"\n[done] {n_dequant} weights dequantized; index updated", flush=True)


if __name__ == "__main__":
    main()
