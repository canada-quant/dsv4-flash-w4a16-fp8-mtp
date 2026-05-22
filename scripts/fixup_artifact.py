"""Minimal additive fixup for w4a16-fp8-mtp-smoke artifact.

Three things, all in one atomic pass per shard:
  1. Restore FP32 dtype on 103 keys missed by prior restore script:
     - 41 `*.ffn.gate.bias` (renamed from e_score_correction_bias post-restore)
     - 41 `*.attn.compressor.ape` (renamed from position_bias post-restore)
     - 21 `*.attn.indexer.compressor.ape` (same)
  2. Upcast `head.weight` BF16 -> FP32 to match sibling artifact.
  3. Inject 2 alias keys (Agent B H1 — missing causes 0% MTP acceptance):
     - `mtp.0.head.weight`     = FP32 copy of head.weight (sibling has it FP32)
     - `mtp.0.emb.tok_emb.weight` = BF16 copy of embed.weight (sibling BF16)
     Aliases go in shard 4 (smallest, holds all mtp.0.* already).

Strict atomicity:
  - Read full shard into memory
  - Apply all changes in dict
  - save_file -> .tmp
  - os.replace -> final name
  - No safetensors writes via timeout, no follow-up scripts.

Run --dry-run first to print plan + size deltas without touching disk.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ART = Path("/scratch/weights/w4a16-fp8-mtp-smoke")
SRC = Path("/scratch/weights/bf16-mtp")
ALIAS_SHARD = "model-00004-of-00004.safetensors"  # holds all mtp.0.* keys


def is_fp32_target_post_rename(k: str) -> bool:
    """Predicate over POST-rename key names — catches all 103 misses."""
    return (
        k.endswith(".ffn.gate.bias")
        or k.endswith(".attn.compressor.ape")
        or k.endswith(".attn.indexer.compressor.ape")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    art_idx_p = ART / "model.safetensors.index.json"
    art_idx = json.load(open(art_idx_p))
    art_wm = art_idx["weight_map"]

    src_wm = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]

    # --- Plan: find which shards need work ---
    fp32_targets = [k for k in art_wm if is_fp32_target_post_rename(k)]
    # Group by shard
    fp32_by_shard = {}
    for k in fp32_targets:
        fp32_by_shard.setdefault(art_wm[k], []).append(k)

    # Aliases to inject go into ALIAS_SHARD
    inject_shard = ALIAS_SHARD
    # head.weight upcast lives in whatever shard holds head.weight
    head_shard = art_wm["head.weight"]
    embed_shard = art_wm["embed.weight"]

    shards_to_touch = set(fp32_by_shard.keys()) | {inject_shard, head_shard, embed_shard}

    print(f"=== Fixup plan ===")
    print(f"  artifact: {ART}")
    print(f"  source:   {SRC}")
    print(f"  FP32 restore targets: {len(fp32_targets)}")
    for shard, keys in sorted(fp32_by_shard.items()):
        print(f"    {shard}: {len(keys)} keys")
    print(f"  alias injection shard: {inject_shard}")
    print(f"  head.weight shard: {head_shard} (upcast BF16 -> FP32)")
    print(f"  embed.weight shard: {embed_shard} (no change)")
    print(f"  shards to rewrite: {sorted(shards_to_touch)}")

    # Pre-load head/embed for alias injection. Read from source so we get the
    # canonical numerical content (head is BF16 in source; we will upcast).
    # Actually use artifact's own current values — they should match source already.
    # (Postprocess didn't touch them; the only change vs source is BF16 stays BF16.)
    print(f"\n  Loading head.weight + embed.weight from artifact ...", flush=True)
    with safe_open(ART / head_shard, framework="pt") as f:
        head_bf16 = f.get_tensor("head.weight")
    with safe_open(ART / embed_shard, framework="pt") as f:
        embed_bf16 = f.get_tensor("embed.weight")
    head_fp32 = head_bf16.to(torch.float32).contiguous()
    print(f"    head.weight: {head_bf16.dtype} {tuple(head_bf16.shape)} -> FP32 ({head_fp32.element_size() * head_fp32.numel() / 1e9:.2f} GB)")
    print(f"    embed.weight: {embed_bf16.dtype} {tuple(embed_bf16.shape)} (stays BF16, {embed_bf16.element_size() * embed_bf16.numel() / 1e9:.2f} GB)")

    # --- Pre-load FP32 source tensors for restoration ---
    print(f"\n  Loading {len(fp32_targets)} FP32 source tensors ...", flush=True)
    src_fp32 = {}
    src_by_shard = {}
    for k in fp32_targets:
        if k not in src_wm:
            print(f"    WARN: {k} not in source — will skip", file=sys.stderr)
            continue
        src_by_shard.setdefault(src_wm[k], []).append(k)
    for s, keys in src_by_shard.items():
        with safe_open(SRC / s, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                if t.dtype != torch.float32:
                    print(f"    WARN: {k} src dtype is {t.dtype}, not FP32 — skipping", file=sys.stderr)
                    continue
                src_fp32[k] = t.contiguous()
    print(f"    loaded {len(src_fp32)} FP32 source tensors")

    if args.dry_run:
        print(f"\n=== DRY RUN — no writes ===")
        # Size deltas
        head_delta = (head_fp32.element_size() - head_bf16.element_size()) * head_bf16.numel()
        alias_head_delta = head_fp32.element_size() * head_fp32.numel()
        alias_embed_delta = embed_bf16.element_size() * embed_bf16.numel()
        fp32_delta = sum(
            (src_fp32[k].element_size() - 2) * src_fp32[k].numel()
            for k in src_fp32
            if (src_fp32[k].element_size() - 2) > 0
        )
        print(f"  Size deltas:")
        print(f"    head BF16->FP32 upcast: +{head_delta/1e9:.2f} GB (in {head_shard})")
        print(f"    mtp.0.head.weight FP32 inject: +{alias_head_delta/1e9:.2f} GB (in {inject_shard})")
        print(f"    mtp.0.emb.tok_emb.weight BF16 inject: +{alias_embed_delta/1e9:.2f} GB (in {inject_shard})")
        print(f"    103 FP32 restores (BF16->FP32 in target shards): +{fp32_delta/1e9:.2f} GB total")
        total_delta = head_delta + alias_head_delta + alias_embed_delta + fp32_delta
        print(f"    TOTAL: +{total_delta/1e9:.2f} GB")
        return

    # --- Apply: rewrite each touched shard atomically ---
    print(f"\n=== Applying — rewriting {len(shards_to_touch)} shards atomically ===")
    new_index_adds = {}  # for alias keys
    for shard in sorted(shards_to_touch):
        sp = ART / shard
        print(f"\n  [{shard}] reading...", flush=True)
        tensors = {}
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        orig_n = len(tensors)

        n_fp32_restored = 0
        # 1. FP32 restoration for keys in this shard
        for k in fp32_by_shard.get(shard, []):
            if k in src_fp32 and tensors.get(k) is not None and tensors[k].dtype != torch.float32:
                tensors[k] = src_fp32[k]
                n_fp32_restored += 1

        # 2. head.weight upcast (if this shard holds head.weight)
        upcast_head = False
        if shard == head_shard and "head.weight" in tensors and tensors["head.weight"].dtype != torch.float32:
            tensors["head.weight"] = head_fp32
            upcast_head = True

        # 3. Alias injection (if this is the alias shard)
        aliases_added = []
        if shard == inject_shard:
            if "mtp.0.head.weight" not in tensors:
                tensors["mtp.0.head.weight"] = head_fp32
                aliases_added.append("mtp.0.head.weight")
                new_index_adds["mtp.0.head.weight"] = shard
            if "mtp.0.emb.tok_emb.weight" not in tensors:
                tensors["mtp.0.emb.tok_emb.weight"] = embed_bf16
                aliases_added.append("mtp.0.emb.tok_emb.weight")
                new_index_adds["mtp.0.emb.tok_emb.weight"] = shard

        if n_fp32_restored == 0 and not upcast_head and not aliases_added:
            print(f"    no changes — skipping rewrite")
            continue

        # Write tmp atomically
        tmp_path = sp.with_name(sp.name + ".tmp")
        print(f"    writing {len(tensors)} tensors (was {orig_n}) -> {tmp_path.name}")
        print(f"      FP32 restored: {n_fp32_restored}; head upcast: {upcast_head}; aliases added: {aliases_added}")
        save_file(tensors, str(tmp_path))
        os.replace(str(tmp_path), str(sp))
        print(f"    atomic replace OK -> {sp.name} ({sp.stat().st_size/1e9:.2f} GB)")

    # Update index.json with alias entries
    if new_index_adds:
        for k, shard in new_index_adds.items():
            art_wm[k] = shard
        # Atomic index write
        idx_tmp = art_idx_p.with_name(art_idx_p.name + ".tmp")
        json.dump(art_idx, open(idx_tmp, "w"), indent=2)
        os.replace(str(idx_tmp), str(art_idx_p))
        print(f"\n  index.json updated: {len(new_index_adds)} new entries; total keys: {len(art_wm)}")

    print(f"\n=== Verification ===")
    # Re-read index
    wm2 = json.load(open(art_idx_p))["weight_map"]
    mtp_keys2 = [k for k in wm2 if k.startswith("mtp.")]
    print(f"  total keys: {len(wm2)}")
    print(f"  mtp.* keys: {len(mtp_keys2)} (expected 799)")
    assert "mtp.0.head.weight" in wm2
    assert "mtp.0.emb.tok_emb.weight" in wm2

    # Spot-check the alias tensors
    with safe_open(ART / inject_shard, framework="pt") as f:
        h = f.get_tensor("mtp.0.head.weight")
        e = f.get_tensor("mtp.0.emb.tok_emb.weight")
    print(f"  mtp.0.head.weight: {h.dtype} {tuple(h.shape)}")
    print(f"  mtp.0.emb.tok_emb.weight: {e.dtype} {tuple(e.shape)}")

    # Verify byte-match against sources
    with safe_open(ART / head_shard, framework="pt") as f:
        h_top = f.get_tensor("head.weight")
    with safe_open(ART / embed_shard, framework="pt") as f:
        e_top = f.get_tensor("embed.weight")
    print(f"  head.weight: {h_top.dtype} {tuple(h_top.shape)}")
    print(f"  embed.weight: {e_top.dtype} {tuple(e_top.shape)}")
    assert h_top.dtype == torch.float32, "head.weight must be FP32 to match sibling"
    assert torch.equal(h, h_top), "mtp.0.head.weight must byte-match head.weight"
    assert torch.equal(e, e_top), "mtp.0.emb.tok_emb.weight must byte-match embed.weight"
    print(f"  alias byte-match: OK")

    # Verify FP32 drift fixed
    bad = []
    for k in fp32_targets:
        sh = wm2[k]
        with safe_open(ART / sh, framework="pt") as f:
            t = f.get_tensor(k)
        if t.dtype != torch.float32:
            bad.append((k, t.dtype))
    print(f"  FP32 targets verified: {len(fp32_targets) - len(bad)}/{len(fp32_targets)} FP32")
    if bad:
        print(f"  STILL NOT FP32:")
        for k, d in bad[:10]:
            print(f"    {k}: {d}")

    print(f"\n=== Fixup DONE ===")


if __name__ == "__main__":
    main()
