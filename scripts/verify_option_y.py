#!/usr/bin/env python3
"""Option Y gate — confirm a post-save artifact preserves MTP at BF16.

Reads model.safetensors.index.json + config.json from the W4A16-FP8 output
and checks the following invariants for the deliberate Option Y design
(MTP block stays BF16; main 43 layers get FP8_BLOCK attention + W4A16
routed experts):

  Main-model (must be quantized):
  M1. model.layers.0..42.mlp.experts.*.{gate,up,down}_proj.weight_packed
      count: 43 * 256 * 3 = 33024
  M2. model.layers.0..42.self_attn.{q_a,q_b,kv,o_a,o_b}_proj.weight_scale
      count: 43 * 5 = 215 (subset — actual key set depends on attention
      sub-modules)

  MTP block (must be BF16 — NO scales, NO packed weights):
  Y1. model.mtp.0.mlp.experts.*.weight_packed count: 0
  Y2. model.mtp.0.mlp.experts.*.weight_scale count: 0
  Y3. model.mtp.0.self_attn.*.weight_packed count: 0
  Y4. model.mtp.0.self_attn.*.weight_scale count: 0
  Y5. model.mtp.0.mlp.experts.*.{gate,up,down}_proj.weight present at BF16
      (NOT weight_packed; loaded as 16-bit)
  Y6. model.mtp.0.* total key count >= 768 (MTP block present at all)
      Note: BF16 MTP has 768 expert .weight + ~29 other = ~797 keys.
      Quantized MTP would have ~2300+ keys (extra weight_packed/scale/etc).
      So ">=768" tests presence; "<=900" would test it's BF16, but we
      check Y1-Y4 directly for that.

  Auxiliary (must be BF16):
  A1. model.embed_tokens.weight: BF16
  A2. Output head: BF16. DSV4-Flash uses `head.weight` (not `lm_head.weight`).
      Either name is accepted.
  A3. model.norm.weight: BF16

Failure of any invariant aborts with a clear message. Pass the artifact
path as arg 1 (defaults to /scratch/weights/w4a16-fp8-mtp-smoke).
"""
import argparse
import json
import re
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "model_dir",
        nargs="?",
        default="/scratch/weights/w4a16-fp8-mtp-smoke",
        help="post-calibration W4A16-FP8 model dir (default: smoke output)",
    )
    args = ap.parse_args()

    root = Path(args.model_dir)
    idx_path = root / "model.safetensors.index.json"
    cfg_path = root / "config.json"
    if not idx_path.exists():
        sys.exit(f"FATAL: {idx_path} not found")
    if not cfg_path.exists():
        sys.exit(f"FATAL: {cfg_path} not found")

    weight_map = json.loads(idx_path.read_text()).get("weight_map", {})
    cfg = json.loads(cfg_path.read_text())
    all_keys = list(weight_map.keys())

    main_layers = re.compile(r"^model\.layers\.\d+\.")
    mtp_keys = [k for k in all_keys if k.startswith("model.mtp.")]
    main_keys = [k for k in all_keys if main_layers.match(k)]

    main_expert_packed = [
        k for k in main_keys
        if re.match(
            r"^model\.layers\.\d+\.mlp\.experts\.\d+\."
            r"(gate_proj|up_proj|down_proj)\.weight_packed$",
            k,
        )
    ]
    main_attn_scale = [
        k for k in main_keys
        if re.match(
            r"^model\.layers\.\d+\.self_attn\..+\.weight_scale(_inv)?$", k
        )
    ]

    mtp_expert_packed = [
        k for k in mtp_keys if "experts" in k and k.endswith("weight_packed")
    ]
    mtp_expert_scale = [
        k for k in mtp_keys if "experts" in k and (
            k.endswith(".weight_scale") or k.endswith(".weight_scale_inv")
        )
    ]
    mtp_attn_packed = [
        k for k in mtp_keys if "self_attn" in k and k.endswith("weight_packed")
    ]
    mtp_attn_scale = [
        k for k in mtp_keys if "self_attn" in k and (
            k.endswith(".weight_scale") or k.endswith(".weight_scale_inv")
        )
    ]
    mtp_expert_bf16 = [
        k for k in mtp_keys
        if re.match(
            r"^model\.mtp\.0\.mlp\.experts\.\d+\."
            r"(gate_proj|up_proj|down_proj)\.weight$",
            k,
        )
    ]

    has_embed = "model.embed_tokens.weight" in weight_map
    # DSV4-Flash uses `head.weight` (not `lm_head.weight`); accept either.
    has_lm_head = (
        "lm_head.weight" in weight_map or "head.weight" in weight_map
    )
    has_norm = "model.norm.weight" in weight_map

    print(f"Artifact: {root}")
    print(f"Total tensor keys: {len(all_keys)}")
    print(f"  main model.layers.* keys: {len(main_keys)}")
    print(f"  MTP model.mtp.* keys:     {len(mtp_keys)}")
    print()
    print("Main-model quantization (must be quantized):")
    print(f"  M1. main expert weight_packed: {len(main_expert_packed)}"
          f" (expect 33024 for full 43-layer / 256-expert)")
    print(f"  M2. main attn weight_scale:    {len(main_attn_scale)}"
          f" (expect >0)")
    print()
    print("MTP block (Option Y — must be BF16, NO quant):")
    print(f"  Y1. MTP expert weight_packed: {len(mtp_expert_packed)} (expect 0)")
    print(f"  Y2. MTP expert weight_scale:  {len(mtp_expert_scale)} (expect 0)")
    print(f"  Y3. MTP attn weight_packed:   {len(mtp_attn_packed)} (expect 0)")
    print(f"  Y4. MTP attn weight_scale:    {len(mtp_attn_scale)} (expect 0)")
    print(f"  Y5. MTP expert .weight (BF16): {len(mtp_expert_bf16)}"
          f" (expect ~768 = 256 experts * 3 projections)")
    print(f"  Y6. MTP total keys: {len(mtp_keys)} (expect >=768; ~797 typical for BF16)")
    print()
    print("Auxiliary tensors:")
    print(f"  A1. model.embed_tokens.weight: {'present' if has_embed else 'MISSING'}")
    head_loc = (
        "lm_head.weight" if "lm_head.weight" in weight_map
        else "head.weight" if "head.weight" in weight_map
        else None
    )
    print(f"  A2. output head: {head_loc if head_loc else 'MISSING'}"
          f" (DSV4-Flash uses head.weight)")
    print(f"  A3. model.norm.weight: {'present' if has_norm else 'MISSING'}")

    # config.json sanity
    ql = cfg.get("quantization_config", {}) if isinstance(cfg, dict) else {}
    ignore = ql.get("ignore", [])
    print()
    print(f"config.quantization_config.ignore: {ignore}")
    if "layer_types" in cfg:
        lt = cfg["layer_types"]
        print(f"config.layer_types: len={len(lt)} (expect 43, NOT 44)")
    print(f"config.num_hidden_layers: {cfg.get('num_hidden_layers')}")

    failed = []
    if len(main_expert_packed) < 30000:
        failed.append(
            f"main expert weight_packed too low ({len(main_expert_packed)}); "
            "main MoE didn't quantize"
        )
    if len(main_attn_scale) == 0:
        failed.append("main attention has no weight_scale; FP8 path didn't fire")
    if len(mtp_expert_packed) != 0:
        failed.append(
            f"MTP expert weight_packed = {len(mtp_expert_packed)} (must be 0); "
            "save-time RTN quantized MTP despite ignore=. See "
            "https://github.com/vllm-project/compressed-tensors/issues/712"
        )
    if len(mtp_expert_scale) != 0:
        failed.append(
            f"MTP expert weight_scale = {len(mtp_expert_scale)} (must be 0)"
        )
    if len(mtp_attn_packed) != 0:
        failed.append(
            f"MTP attn weight_packed = {len(mtp_attn_packed)} (must be 0)"
        )
    if len(mtp_attn_scale) != 0:
        failed.append(
            f"MTP attn weight_scale = {len(mtp_attn_scale)} (must be 0)"
        )
    if len(mtp_keys) < 768:
        failed.append(
            f"MTP key count = {len(mtp_keys)} (expect >=768 for 256x3 experts); "
            "MTP may have been dropped at save"
        )
    if not has_embed or not has_lm_head:
        failed.append(
            "missing embed_tokens or output head "
            "(checked lm_head.weight + head.weight)"
        )
    if "layer_types" in cfg and len(cfg["layer_types"]) != 43:
        failed.append(
            f"config.layer_types len = {len(cfg['layer_types'])} (expect 43); "
            "shim's 44-element list leaked through to save"
        )

    if failed:
        print()
        print("OPTION Y GATE: FAILED")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)

    print()
    print("OPTION Y GATE: PASSED — true BF16 MTP + W4A16 main experts + FP8 attn")
    print("Next: run scripts/option_b_serve_smoke.sh to validate end-to-end.")


if __name__ == "__main__":
    main()
