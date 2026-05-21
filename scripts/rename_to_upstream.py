#!/usr/bin/env python3
"""Rewrite a transformers-saved DSv4 W4A16+FP8+MTP artifact to upstream-style
key naming that vLLM's deepseek_v4 weights mapper expects.

Background:
  transformers' save_pretrained writes DSv4 weights with HF-style names
  (model.layers.X.self_attn.q_a_proj, model.layers.X.mlp.experts.N.gate_proj,
  etc.). vLLM's deepseek_v4 expects upstream-style names instead
  (layers.X.attn.wq_a, layers.X.ffn.experts.N.w1, etc.). The vLLM
  WeightsMapper (in vllm/models/deepseek_v4/nvidia/model.py) prefixes
  `layers.` -> `model.layers.` and renames a few common suffixes, but it
  does NOT undo the per-projection renames that transformers performed.

  Confirmed via the predecessor's published artifact at
  https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8 — all
  100,652 weight_map keys there have NO `model.` prefix and use the
  upstream-style names throughout (`layers.0.attn.wq_a.weight_scale`,
  `layers.0.ffn.experts.1.w1.weight_packed`, etc.).

Renames performed (covering iter 8 / Phase 2 artifact shape):
  1. Strip `model.` prefix from every key.
  2. self_attn -> attn (and per-projection renames):
       q_a_proj -> wq_a, q_b_proj -> wq_b, kv_proj -> wkv,
       o_a_proj -> wo_a, o_b_proj -> wo_b
       q_a_norm -> q_norm, kv_norm -> kv_norm (unchanged)
       sinks -> attn_sink
       compressor.gate_proj -> compressor.wgate
       compressor.kv_proj -> compressor.wkv
       compressor.indexer.q_b_proj -> compressor.indexer.wq_b
  3. input_layernorm -> attn_norm, post_attention_layernorm -> ffn_norm.
  4. mlp -> ffn (and per-projection renames in experts + shared_experts):
       gate_proj -> w1, up_proj -> w3, down_proj -> w2
       mlp.gate -> ffn.gate (unchanged after substr replace)
       mlp.gate.e_score_correction_bias -> ffn.gate.bias
  5. attn_hc.{base,fn,scale} -> hc_attn_{base,fn,scale}.
  6. ffn_hc.{base,fn,scale} -> hc_ffn_{base,fn,scale}.
  7. hc_head.hc_{base,fn,scale} -> hc_head_{base,fn,scale}.
  8. embed_tokens.weight -> embed.weight.
  9. lm_head.weight -> head.weight.
 10. MTP equivalents under `mtp.0.*`.

The script:
  - reads model.safetensors.index.json
  - for each shard, reads the safetensors, applies the rename to every
    key, writes the shard back in place
  - updates the index.json with the new key->shard mapping

Validation: after renaming, run with --check-only to confirm no
HF-style keys remain in the index. (No safetensors read pass; just
checks the names.)
"""
import argparse
import json
import re
import sys
from pathlib import Path

# ----- per-layer rename rules (applied AFTER stripping `model.` prefix) -----

# Per-projection substr replacements applied inside a `layers.N.` key
# (or `mtp.N.`). Order matters: longer/specific patterns first.
_PROJ_SUBSTR_REPLACEMENTS = [
    # self_attn -> attn
    (".self_attn.compressor.indexer.gate_proj.", ".attn.compressor.indexer.wgate."),
    (".self_attn.compressor.indexer.kv_proj.", ".attn.compressor.indexer.wkv."),
    (".self_attn.compressor.indexer.q_b_proj.", ".attn.compressor.indexer.wq_b."),
    (".self_attn.compressor.indexer.kv_norm.", ".attn.compressor.indexer.kv_norm."),
    (".self_attn.compressor.indexer.weights_proj.", ".attn.compressor.indexer.weights_proj."),
    (".self_attn.compressor.indexer.position_bias", ".attn.compressor.indexer.position_bias"),
    (".self_attn.compressor.gate_proj.", ".attn.compressor.wgate."),
    (".self_attn.compressor.kv_proj.", ".attn.compressor.wkv."),
    (".self_attn.compressor.kv_norm.", ".attn.compressor.kv_norm."),
    (".self_attn.compressor.position_bias", ".attn.compressor.position_bias"),
    (".self_attn.q_a_proj.", ".attn.wq_a."),
    (".self_attn.q_b_proj.", ".attn.wq_b."),
    (".self_attn.kv_proj.", ".attn.wkv."),
    (".self_attn.o_a_proj.", ".attn.wo_a."),
    (".self_attn.o_b_proj.", ".attn.wo_b."),
    (".self_attn.q_a_norm.", ".attn.q_norm."),
    (".self_attn.kv_norm.", ".attn.kv_norm."),
    (".self_attn.sinks", ".attn.attn_sink"),
    # mlp -> ffn (experts AFTER gate; gate FIRST or it gets overridden)
    (".mlp.gate.e_score_correction_bias", ".ffn.gate.bias"),
    (".mlp.gate.", ".ffn.gate."),
    (".mlp.shared_experts.gate_proj.", ".ffn.shared_experts.w1."),
    (".mlp.shared_experts.up_proj.", ".ffn.shared_experts.w3."),
    (".mlp.shared_experts.down_proj.", ".ffn.shared_experts.w2."),
    (".mlp.experts.", ".ffn.experts."),  # then per-projection below
    # layernorms
    (".input_layernorm.", ".attn_norm."),
    (".post_attention_layernorm.", ".ffn_norm."),
    # hyperconn
    (".attn_hc.base", ".hc_attn_base"),
    (".attn_hc.fn", ".hc_attn_fn"),
    (".attn_hc.scale", ".hc_attn_scale"),
    (".ffn_hc.base", ".hc_ffn_base"),
    (".ffn_hc.fn", ".hc_ffn_fn"),
    (".ffn_hc.scale", ".hc_ffn_scale"),
]

# For ffn.experts.N.{gate,up,down}_proj -> ffn.experts.N.{w1,w3,w2}
# applied after the .mlp.experts. -> .ffn.experts. swap above.
_EXPERT_PROJ_RE = [
    (re.compile(r"\.ffn\.experts\.(\d+)\.gate_proj\."), r".ffn.experts.\1.w1."),
    (re.compile(r"\.ffn\.experts\.(\d+)\.up_proj\."),   r".ffn.experts.\1.w3."),
    (re.compile(r"\.ffn\.experts\.(\d+)\.down_proj\."), r".ffn.experts.\1.w2."),
]

# Top-level (non-per-layer) substr replacements applied to the key as a whole.
_TOPLEVEL_SUBSTR_REPLACEMENTS = [
    ("hc_head.hc_base",  "hc_head_base"),
    ("hc_head.hc_fn",    "hc_head_fn"),
    ("hc_head.hc_scale", "hc_head_scale"),
]

# Exact key replacements at the very top level (after model. strip).
_TOPLEVEL_EXACT_REPLACEMENTS = {
    "embed_tokens.weight": "embed.weight",
    "lm_head.weight":      "head.weight",
    "norm.weight":         "norm.weight",  # unchanged
}


def rename_key(k: str) -> str:
    # 1) strip model. prefix
    if k.startswith("model."):
        k = k[len("model."):]

    # 2) per-projection substrings (covers layers.N.* and mtp.N.*)
    for old, new in _PROJ_SUBSTR_REPLACEMENTS:
        k = k.replace(old, new)

    # 3) expert .gate_proj./.up_proj./.down_proj. -> .w1./.w3./.w2.
    for pat, repl in _EXPERT_PROJ_RE:
        k = pat.sub(repl, k)

    # 4) top-level (hc_head etc)
    for old, new in _TOPLEVEL_SUBSTR_REPLACEMENTS:
        k = k.replace(old, new)

    # 5) exact key replacements
    if k in _TOPLEVEL_EXACT_REPLACEMENTS:
        k = _TOPLEVEL_EXACT_REPLACEMENTS[k]

    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir", help="path to the artifact directory")
    ap.add_argument(
        "--check-only", action="store_true",
        help="don't touch shards; just print the proposed rename map",
    )
    ap.add_argument(
        "--predecessor-index",
        default="/tmp/predecessor_index.json",
        help="path to predecessor's model.safetensors.index.json for "
             "post-hoc validation that we produce predecessor-compatible "
             "names for the MAIN 43 layers",
    )
    args = ap.parse_args()

    art = Path(args.artifact_dir)
    idx_p = art / "model.safetensors.index.json"
    if not idx_p.exists():
        sys.exit(f"FATAL: {idx_p} not found")

    idx = json.load(open(idx_p))
    wm = idx["weight_map"]

    # Compute renames (and per-shard groupings)
    renames = {}
    collisions = []
    for old in wm.keys():
        new = rename_key(old)
        if new != old:
            renames[old] = new

    # Detect collisions: two different OLD keys mapping to the same NEW key
    new_to_olds: dict[str, list[str]] = {}
    for old, new in renames.items():
        new_to_olds.setdefault(new, []).append(old)
    for new, olds in new_to_olds.items():
        if len(olds) > 1:
            collisions.append((new, olds))

    # Also check NEW keys colliding with existing OLD keys
    existing_unchanged = set(wm.keys()) - set(renames.keys())
    for old, new in renames.items():
        if new in existing_unchanged:
            collisions.append((new, [old, "(unchanged existing key)"]))

    if collisions:
        print(f"!! FATAL: {len(collisions)} collisions detected:")
        for new, olds in collisions[:10]:
            print(f"  {new}  <- {olds}")
        sys.exit(2)

    print(f"Total keys:    {len(wm)}")
    print(f"Renames:       {len(renames)}")
    print(f"Unchanged:     {len(existing_unchanged)}")
    # Sample renames
    print("Sample renames:")
    sample = sorted(renames.items())[:8] + sorted(renames.items())[-8:]
    for old, new in sample:
        print(f"  {old} -> {new}")

    # Predecessor validation: for the main 43 layers, all our new keys
    # (excluding compressor-related extras the predecessor lacks) should
    # be a subset of the predecessor's key set.
    pred_path = Path(args.predecessor_index)
    if pred_path.exists():
        pred_wm = json.load(open(pred_path))["weight_map"]
        pred_keys = set(pred_wm.keys())
        new_keys = set(wm.get(k, None) for k in wm)  # all NEW keys
        new_keys = set()
        for old in wm:
            new_keys.add(renames.get(old, old))

        # Only test layer-N keys not involving the indexer/compressor (which
        # the predecessor may not have)
        layer_keys = {
            k for k in new_keys
            if re.match(r"^layers\.\d+\.", k)
            and "indexer" not in k
        }
        in_pred = layer_keys & pred_keys
        not_in_pred = layer_keys - pred_keys
        print(f"\nPredecessor validation:")
        print(f"  main-layer keys (no indexer): {len(layer_keys)}")
        print(f"  in predecessor:                {len(in_pred)}")
        print(f"  NOT in predecessor:            {len(not_in_pred)}")
        if not_in_pred:
            print(f"  Sample mismatches (5):")
            for k in sorted(not_in_pred)[:5]:
                print(f"    {k}")

    if args.check_only:
        print("\n(--check-only) — not modifying shards")
        return

    # Apply renames to each shard
    from safetensors import safe_open
    from safetensors.torch import save_file

    shards_to_patch: dict[str, list[tuple[str, str]]] = {}
    for old, new in renames.items():
        shards_to_patch.setdefault(wm[old], []).append((old, new))

    print(f"\nApplying renames to {len(shards_to_patch)} shards...")
    for i, (shard, sh_renames) in enumerate(sorted(shards_to_patch.items()), 1):
        sp = art / shard
        print(f"[{i}/{len(shards_to_patch)}] {shard}: {len(sh_renames)} renames",
              flush=True)
        tensors = {}
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        rm = dict(sh_renames)
        new_tensors = {rm.get(k, k): v for k, v in tensors.items()}
        save_file(new_tensors, str(sp))

    # Update index
    new_wm = {}
    for old, shard in wm.items():
        new_wm[renames.get(old, old)] = shard
    idx["weight_map"] = new_wm
    json.dump(idx, open(idx_p, "w"), indent=2)
    print(f"\nUpdated {idx_p}")

    print("DONE — re-run vLLM serve to validate")


if __name__ == "__main__":
    main()
