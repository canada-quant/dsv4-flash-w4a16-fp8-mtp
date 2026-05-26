#!/usr/bin/env python3
"""Rename `.weight_scale` → `.weight_scale_inv` for FP8 keys.
Optionally rename `ffn.gate.e_score_correction_bias` → `ffn.gate.bias` (Card A architecture-drift fix).

Writes new shards to an output directory (since HF cache is on disk-tight /).
Outputs a complete snapshot dir ready to be served via --model <output_dir>."""
import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

def classify_fp8_modules(header):
    fp8 = set()
    for k, v in header.items():
        if k == "__metadata__":
            continue
        if v.get("dtype", "") == "F8_E4M3" and k.endswith(".weight"):
            fp8.add(k[:-len(".weight")])
    return fp8

def build_renames(header, also_e_score_bias=False):
    fp8_modules = classify_fp8_modules(header)
    renames = {}
    for k in header:
        if k == "__metadata__":
            continue
        if k.endswith(".weight_scale"):
            module = k[:-len(".weight_scale")]
            if module in fp8_modules:
                renames[k] = k + "_inv"
        if also_e_score_bias and k.endswith("ffn.gate.e_score_correction_bias"):
            renames[k] = k[:-len("e_score_correction_bias")] + "bias"
    return renames

def rewrite_shard(src_path, dst_path, also_e_score_bias=False):
    with open(src_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_len)
        data_offset = 8 + header_len

    header = json.loads(header_bytes.decode("utf-8"))
    renames = build_renames(header, also_e_score_bias=also_e_score_bias)

    if not renames:
        print(f"[copy] {src_path}: no renames, just copying", file=sys.stderr)
        shutil.copy(src_path, dst_path)
        return {}

    print(f"[plan] {src_path}: {len(renames)} renames "
          f"(sample: {next(iter(renames))} → {renames[next(iter(renames))]})",
          file=sys.stderr)

    new_header = {}
    for k, v in header.items():
        new_header[renames.get(k, k)] = v

    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(new_header_bytes) % 8) % 8
    new_header_bytes += b" " * pad

    src_size = os.path.getsize(src_path)
    data_size = src_size - data_offset

    print(f"[write] {dst_path}: header={len(new_header_bytes)}B data={data_size/1e9:.2f}GB",
          file=sys.stderr)
    with open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(new_header_bytes)))
        out.write(new_header_bytes)
        with open(src_path, "rb") as f:
            f.seek(data_offset)
            remaining = data_size
            chunk_size = 128 * 1024 * 1024
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk: break
                out.write(chunk)
                remaining -= len(chunk)
    print(f"[done]  {dst_path}", file=sys.stderr)
    return renames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_snapshot", help="HF snapshot dir with model-*.safetensors")
    ap.add_argument("dst_snapshot", help="Output dir (will be created)")
    ap.add_argument("--also-e-score-bias", action="store_true",
                    help="Also rename ffn.gate.e_score_correction_bias → ffn.gate.bias (Card A)")
    args = ap.parse_args()

    src = Path(args.src_snapshot).resolve()
    dst = Path(args.dst_snapshot)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy non-shard files first (config, tokenizer, index)
    for f in src.iterdir():
        if f.is_symlink() or f.is_file():
            real_f = f.resolve()
            target = dst / f.name
            if f.name.endswith(".safetensors"):
                continue  # shards handled separately
            if target.exists():
                target.unlink()
            shutil.copy(real_f, target)
            print(f"[copy] {target.name}", file=sys.stderr)

    # Rewrite shards
    shards = sorted(src.glob("model-*.safetensors"))
    all_renames = {}
    for shard in shards:
        real_shard = shard.resolve()
        dst_shard = dst / shard.name
        renames = rewrite_shard(real_shard, dst_shard,
                                also_e_score_bias=args.also_e_score_bias)
        all_renames.update(renames)

    # Update index
    index_path = dst / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        new_wm = {}
        n_renamed = 0
        for k, v in wm.items():
            if k in all_renames:
                new_wm[all_renames[k]] = v
                n_renamed += 1
            else:
                new_wm[k] = v
        idx["weight_map"] = new_wm
        with open(index_path, "w") as f:
            json.dump(idx, f, indent=2)
        print(f"[index] renamed {n_renamed} entries", file=sys.stderr)

    print(f"[total] {len(all_renames)} keys renamed; output at {dst}", file=sys.stderr)

if __name__ == "__main__":
    main()
