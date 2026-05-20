"""Post-process the calibration artifact for vLLM jasl/dm120 + DeepseekV4 serve.

Applies the transforms validated against savetest2:

  1. quantization_config.config_groups.group_1.targets: rename suffix
     (w1|w2|w3) -> (gate_proj|up_proj|down_proj) so vLLM's
     CompressedTensorsMoEMethod.get_moe_method scheme probe matches.
  2. quantization_config.config_groups.group_0.input_activations = None
     so vLLM picks CompressedTensorsW8A16Fp8 (which auto-renames
     weight_scale -> weight_scale_inv at process_weights_after_loading for
     block strategy).
  3. quantization_config.scale_fmt = "ue8m0" (read by DeepseekV4 model).
  4. Rename mtp.0.embed.weight -> mtp.0.emb.tok_emb.weight so vLLM's MTP
     load_weights -> _remap_weight_name -> "embed_tokens" path matches.
  5. Copy tokenizer files / generation_config.json from bf16-mtp source dir
     if missing (the save path now handles this but be defensive).

Usage:
  python scripts/postprocess_for_vllm.py \\
      --artifact /scratch/weights/w4a16-fp8-mtp \\
      --bf16-source /scratch/weights/bf16-mtp
"""
import argparse
import json
import os
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


MTP_EMBED_OLD = re.compile(r"^mtp\.(\d+)\.embed\.weight$")


def needs_mtp_embed_rename(name: str) -> str | None:
    m = MTP_EMBED_OLD.match(name)
    if m is None:
        return None
    return f"mtp.{m.group(1)}.emb.tok_emb.weight"


def patch_config(artifact_dir: Path) -> None:
    cfg_p = artifact_dir / "config.json"
    cfg = json.load(open(cfg_p))
    qc = cfg.setdefault("quantization_config", {})

    # (1) group_1.targets: rename w1|w2|w3 -> gate_proj|up_proj|down_proj
    g1 = qc.get("config_groups", {}).get("group_1")
    if g1 is not None:
        old = list(g1.get("targets", []))
        new = [
            t.replace("(w1|w2|w3)", "(gate_proj|up_proj|down_proj)")
            for t in old
        ]
        if old != new:
            g1["targets"] = new
            print(f"[cfg] group_1 targets: {old} -> {new}")

    # (2) group_0.input_activations = None so W8A16Fp8 scheme triggers
    g0 = qc.get("config_groups", {}).get("group_0")
    if g0 is not None and g0.get("input_activations") is not None:
        g0["input_activations"] = None
        print("[cfg] group_0.input_activations cleared (W8A16Fp8 path)")

    # (3) scale_fmt
    if not qc.get("scale_fmt"):
        qc["scale_fmt"] = "ue8m0"
        print("[cfg] quantization_config.scale_fmt = ue8m0")

    json.dump(cfg, open(cfg_p, "w"), indent=2)
    print(f"[cfg] wrote {cfg_p}")


def rename_mtp_embed_in_safetensors(artifact_dir: Path) -> None:
    index_p = artifact_dir / "model.safetensors.index.json"
    if not index_p.exists():
        print(f"[mtp-rename] no index at {index_p}, skipping")
        return
    index = json.load(open(index_p))
    wmap = index["weight_map"]

    shards_with_renames: dict[str, list[tuple[str, str]]] = {}
    for k, shard in wmap.items():
        new_name = needs_mtp_embed_rename(k)
        if new_name is None:
            continue
        shards_with_renames.setdefault(shard, []).append((k, new_name))

    if not shards_with_renames:
        print("[mtp-rename] no mtp.N.embed.weight keys to rename")
        return

    for shard, renames in shards_with_renames.items():
        sp = artifact_dir / shard
        print(f"[mtp-rename] {shard}: {renames}")
        tensors = {}
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        rm = dict(renames)
        new_tensors = {rm.get(k, k): v for k, v in tensors.items()}
        save_file(new_tensors, str(sp))

    new_wmap = {}
    for k, shard in wmap.items():
        new_name = needs_mtp_embed_rename(k)
        new_wmap[new_name or k] = shard
    index["weight_map"] = new_wmap
    json.dump(index, open(index_p, "w"), indent=2)
    print(f"[mtp-rename] updated index")


def copy_tokenizer_files(artifact_dir: Path, source_dir: Path) -> None:
    if not source_dir.exists():
        print(f"[tokenizer] source dir {source_dir} missing, skipping")
        return
    for fname in ("tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json",
                  "chat_template.jinja"):
        src = source_dir / fname
        dst = artifact_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[tokenizer] copied {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--bf16-source", default="/scratch/weights/bf16-mtp")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact)
    if not artifact_dir.exists():
        raise FileNotFoundError(artifact_dir)

    print(f"=== Post-processing {artifact_dir} ===")
    patch_config(artifact_dir)
    rename_mtp_embed_in_safetensors(artifact_dir)
    copy_tokenizer_files(artifact_dir, Path(args.bf16_source))
    print("=== DONE ===")


if __name__ == "__main__":
    main()
