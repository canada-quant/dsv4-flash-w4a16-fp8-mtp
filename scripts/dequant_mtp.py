#!/usr/bin/env python3
"""Dequantize DeepSeek-V4-Flash native checkpoint to BF16, preserving MTP.

The upstream HF checkpoint at deepseek-ai/DeepSeek-V4-Flash uses:
  * FP4 (e2m1, packed 2-per-int8) routed experts with a per-32-elem ue8m0 scale
  * FP8 (e4m3) attention/dense linear weights with a 128x128-block ue8m0 scale
  * BF16/F32/I64 for everything else (norms, embeddings, gates, attn_sink, ...)
  * mtp.* keys for the 1 MTP layer (num_nextn_predict_layers=1)

This script outputs an HF-format BF16 checkpoint:
  * unpacks FP4 (using the same FP4_TABLE as upstream inference/convert.py)
  * dequantizes FP8 with per-block scale
  * keeps mtp.* tensors verbatim (dequantizing them the same way)
  * drops every consumed *.weight_scale / *.weight_scale_inv tensor
  * writes BF16-only safetensors shards + a regenerated index + config.json
    with quantization_config stripped (the dequant output is unquantized).

Why this exists separately from upstream's convert.py:
  Upstream's convert.py targets DeepSeek's *internal* model-parallel format and
  intentionally `continue`s on mtp.* keys matching ("emb" in name or name.endswith(
  "head.weight")). For W4A16-FP8 + MTP re-quant we must preserve the entire MTP
  block intact so the GPTQ pass in Phase 2 can calibrate layer 43.

Idempotent: re-running on an existing output dir overwrites shards.

Usage:
    python scripts/dequant_mtp.py \\
        --input  ./weights/upstream \\
        --output ./weights/bf16-mtp \\
        --device cuda

Memory: streaming, processes one input shard at a time. Largest intermediate
tensor is one expert MLP slab dequanted to BF16 (~50-300 MB on B300). For 8x
B300 with 275 GB HBM each, GPU 0 alone is sufficient.
"""
import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


FP4_BLOCK = 32
FP8_BLOCK = 128


def unpack_fp4_to_bf16(weight_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 (e2m1 packed 2-per-int8) + per-32-elem scale -> BF16.

    weight_int8: shape (out_dim, in_dim_packed) where in_dim_packed = in_dim // 2
    scale:       shape (out_dim, in_dim // FP4_BLOCK), any float-castable dtype
    returns:     shape (out_dim, in_dim) bfloat16
    """
    assert weight_int8.dtype == torch.int8, f"expected int8 FP4, got {weight_int8.dtype}"
    assert weight_int8.ndim == 2
    out_dim, in_dim_packed = weight_int8.shape
    in_dim = in_dim_packed * 2
    assert scale.shape == (out_dim, in_dim // FP4_BLOCK), \
        f"scale shape {scale.shape} != ({out_dim}, {in_dim // FP4_BLOCK})"

    table = FP4_TABLE.to(weight_int8.device)
    x = weight_int8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    unpacked = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(1)
    # unpacked: (out_dim, in_dim) fp32

    scale_fp32 = scale.float().repeat_interleave(FP4_BLOCK, dim=-1)
    return (unpacked * scale_fp32).to(torch.bfloat16)


def dequant_fp8_block_to_bf16(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 (e4m3) + 128x128-block scale -> BF16.

    weight_fp8: shape (M, N) float8_e4m3fn
    scale:      shape (M // 128, N // 128), ue8m0 or fp32-castable
    returns:    shape (M, N) bfloat16
    """
    assert weight_fp8.dtype == torch.float8_e4m3fn, \
        f"expected float8_e4m3fn, got {weight_fp8.dtype}"
    M, N = weight_fp8.shape
    assert M % FP8_BLOCK == 0 and N % FP8_BLOCK == 0
    assert scale.shape == (M // FP8_BLOCK, N // FP8_BLOCK)

    scale_fp32 = (
        scale.float()
        .repeat_interleave(FP8_BLOCK, dim=0)
        .repeat_interleave(FP8_BLOCK, dim=1)
    )
    return (weight_fp8.float() * scale_fp32).to(torch.bfloat16)


def dequant_fp8_per_token_to_bf16(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 with per-row (1D) scale, used by ``wo_a.weight`` style layers.

    Mirrors the special case in upstream inference/convert.py.
    """
    assert weight_fp8.dtype == torch.float8_e4m3fn
    out_dim, in_dim = weight_fp8.shape
    # Two possible 1D layouts: (out_dim,) per-row, or (out_dim, in_dim // 128) per-row-blocks
    if scale.ndim == 1:
        assert scale.shape == (out_dim,)
        return (weight_fp8.float() * scale.float().unsqueeze(-1)).to(torch.bfloat16)
    if scale.ndim == 2 and scale.shape == (out_dim, in_dim // FP8_BLOCK):
        scale_fp32 = scale.float().repeat_interleave(FP8_BLOCK, dim=1)
        return (weight_fp8.float() * scale_fp32).to(torch.bfloat16)
    raise ValueError(f"unexpected FP8 scale shape {scale.shape} for weight {weight_fp8.shape}")


def collect_tensor_index(input_dir: Path) -> dict[str, str]:
    """Return {tensor_name -> shard_filename} by scanning every input shard."""
    index = {}
    for shard in sorted(input_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                index[k] = shard.name
    return index


def dequant_shard(
    shard_path: Path,
    name_to_shard: dict[str, str],
    input_dir: Path,
    device: str,
) -> dict[str, torch.Tensor]:
    """Load ``shard_path``, dequant any FP4/FP8 weights, return BF16 dict.

    Scale tensors from other shards are loaded on demand so that a weight in
    one shard can be paired with its scale in a different shard (HF default
    shard splits do not guarantee co-location).
    """
    out: dict[str, torch.Tensor] = {}

    def fetch_scale(scale_name: str) -> torch.Tensor:
        path = input_dir / name_to_shard[scale_name]
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(scale_name)

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    for name in keys:
        if name.endswith(".weight_scale") or name.endswith(".weight_scale_inv"):
            # Scales are consumed when we dequant their paired weight; skip here.
            continue

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)

        # FP4 (int8 packed) — paired with .weight_scale
        if name.endswith(".weight") and tensor.dtype == torch.int8:
            scale_name = name[: -len(".weight")] + ".weight_scale"
            if scale_name not in name_to_shard:
                # Some int8 tensors are genuine int8 (bookkeeping); pass through.
                out[name] = tensor
                continue
            scale = fetch_scale(scale_name)
            w = tensor.to(device, non_blocking=True)
            s = scale.to(device, non_blocking=True)
            out[name] = unpack_fp4_to_bf16(w, s).cpu()
            del w, s
            continue

        # FP8 (e4m3) — paired with .weight_scale_inv (block) or .weight_scale (per-token)
        if name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn:
            inv_name = name[: -len(".weight")] + ".weight_scale_inv"
            ts_name = name[: -len(".weight")] + ".weight_scale"
            if inv_name in name_to_shard:
                scale = fetch_scale(inv_name)
                w = tensor.to(device, non_blocking=True)
                s = scale.to(device, non_blocking=True)
                out[name] = dequant_fp8_block_to_bf16(w, s).cpu()
                del w, s
            elif ts_name in name_to_shard:
                scale = fetch_scale(ts_name)
                w = tensor.to(device, non_blocking=True)
                s = scale.to(device, non_blocking=True)
                out[name] = dequant_fp8_per_token_to_bf16(w, s).cpu()
                del w, s
            else:
                raise RuntimeError(f"FP8 weight {name} has no scale in checkpoint")
            continue

        # Everything else (BF16, F32, I64) — pass through
        out[name] = tensor

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return out


def write_shard(state: dict[str, torch.Tensor], path: Path) -> int:
    save_file(state, str(path), metadata={"format": "pt"})
    return sum(t.numel() * t.element_size() for t in state.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="upstream HF checkpoint dir")
    p.add_argument("--output", required=True, help="output BF16 dir")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (input_dir / "config.json").exists():
        raise SystemExit(f"missing config.json in {input_dir}")

    print(f"[scan] indexing tensors in {input_dir}")
    name_to_shard = collect_tensor_index(input_dir)
    print(f"[scan] {len(name_to_shard)} tensors across {len(set(name_to_shard.values()))} shards")

    shards = sorted(input_dir.glob("*.safetensors"))
    weight_index: dict[str, str] = {}
    total_bytes = 0

    for i, shard in enumerate(tqdm(shards, desc="dequant")):
        out_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        state = dequant_shard(shard, name_to_shard, input_dir, args.device)
        if not state:
            continue
        size = write_shard(state, output_dir / out_name)
        for k in state:
            weight_index[k] = out_name
        total_bytes += size
        del state
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- assertion gate ----
    mtp_keys = [k for k in weight_index if "mtp" in k.lower()]
    if not mtp_keys:
        raise SystemExit(
            "FATAL: zero MTP tensors written. Phase 2 cannot calibrate layer 43. "
            "Check the upstream checkpoint and the dequant scale-pairing logic."
        )
    print(f"[gate] wrote {len(mtp_keys)} mtp.* tensors (first 5: {mtp_keys[:5]})")

    # ---- write safetensors index ----
    index_path = output_dir / "model.safetensors.index.json"
    index_payload = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_index,
    }
    index_path.write_text(json.dumps(index_payload, indent=2))
    print(f"[index] {index_path} -> {len(weight_index)} keys, {total_bytes / 1e9:.2f} GB")

    # ---- copy + scrub config.json ----
    config = json.loads((input_dir / "config.json").read_text())
    config.pop("quantization_config", None)
    config["torch_dtype"] = "bfloat16"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[config] stripped quantization_config, torch_dtype=bfloat16")

    # ---- copy tokenizer + generation files verbatim ----
    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ):
        src = input_dir / fname
        if src.exists():
            shutil.copyfile(src, output_dir / fname)
            print(f"[copy] {fname}")

    print("\nDEQUANT_DONE")


if __name__ == "__main__":
    main()
