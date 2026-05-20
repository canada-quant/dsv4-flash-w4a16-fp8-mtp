#!/usr/bin/env python3
"""DSv4-Flash W4A16 + FP8_BLOCK GPTQ calibration with MTP preserved (Option D).

Based on the predecessor's proven recipe (8 ranks, 14h known-good wall
clock on H200 p5en, see `findings/phase3b-recovery.md` in the predecessor
repo) with these MTP-retention deltas layered in:

1. `install_mtp_shim()` from `scripts/transformers_mtp_shim.py` — adds
   `DeepseekV4NextNPredictor` class to transformers' DSv4 module so the
   `mtp.*` keys have submodules to load into.
2. Runtime extension of `ARCH_TO_2D_MAPPINGS["deepseek_v4"]` to cover
   `mtp.\d+.mlp.experts.*` paths so `linearize_moe`'s `named_modules()`
   walk picks up the MTP block's MoE.
3. Empties `_keys_to_ignore_on_load_unexpected = []` via the existing
   `patches/modeling_deepseek_v4.py.diff` (applied to venv-calib by
   `bootstrap_p5en_h200.sh`).
4. Per-subgraph atomic checkpointing via `scripts/gptq_checkpoint.py` to
   defend against transient mid-run failure.

Launch:
    torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16_mtp.py \
        --input /scratch/weights/bf16-mtp \
        --output /scratch/weights/w4a16-fp8-mtp-gptq \
        --samples 768 --batch-size 4 --max-seq-len 512

Env required (set in `serve` script or shell):
    TORCH_CUDA_ARCH_LIST=9.0a
    NCCL_TIMEOUT=3600
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
    TORCH_NCCL_BLOCKING_WAIT=0
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Apply MTP shim BEFORE any `from transformers import AutoModelForCausalLM`
# downstream — needs to land before `from_pretrained` would otherwise drop
# mtp.* keys.
from scripts.transformers_mtp_shim import (
    install_mtp_shim,
    install_mtp_conversion_mapping_extension,
)

install_mtp_shim()
# This must run BEFORE load_linearized_moe is invoked — load_linearized_moe
# fetches the conversion mapping at from_pretrained time and freezes it for
# the load. Extending after that point is too late.
install_mtp_conversion_mapping_extension()

# MODULE-level imports required for compressed_tensors.offload.load_offloaded_model:
# its _get_caller_frame() walks frame.f_globals to find AutoModelForCausalLM (a
# _BaseAutoModelClass subclass), and patches from_pretrained on it. If the
# import is inside main(), it lives in f_locals and load_offloaded_model can't
# find it → "auto_offload" device_map stays unrecognized → ValueError on load.
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def extend_arch_to_2d_mappings_for_mtp() -> None:
    """Runtime extension of `llmcompressor.modeling.moe.linearize.ARCH_TO_2D_MAPPINGS`
    to cover `mtp.\\d+.mlp.experts.*` paths. Mirrors the diff in
    `patches/llmc_dsv4_mtp_conversion_mappings.diff` (submitted upstream as
    vllm-project/llm-compressor#2739). At our pinned llm-compressor SHA
    `f2aa32e2`, the dict lives in `linearize.py` rather than its own
    `conversion_mappings.py` file (renamed in a later commit on the
    `kylesayrs/transformers-v5` branch).
    """
    import llmcompressor.modeling.moe.linearize as _lz
    from transformers.core_model_loading import WeightRenaming

    experts_cls, existing = _lz.ARCH_TO_2D_MAPPINGS["deepseek_v4"]
    new_entries = [
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.mlp\.experts\.(\d+)\.w1\.",
            target_patterns=r"mtp.\1.mlp.experts.\2.gate_proj.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.mlp\.experts\.(\d+)\.w2\.",
            target_patterns=r"mtp.\1.mlp.experts.\2.down_proj.",
        ),
        WeightRenaming(
            source_patterns=r"^mtp\.(\d+)\.mlp\.experts\.(\d+)\.w3\.",
            target_patterns=r"mtp.\1.mlp.experts.\2.up_proj.",
        ),
    ]
    # Idempotent guard: only add if no MTP entries are already present.
    # `source_patterns` may be a list or string depending on llm-compressor
    # version — fold both into a single string for the membership check.
    def _as_str(p):
        return "|".join(p) if isinstance(p, (list, tuple)) else str(p)
    existing_sources = {_as_str(getattr(e, "source_patterns", "")) for e in existing}
    for entry in new_entries:
        if _as_str(entry.source_patterns) not in existing_sources:
            existing.append(entry)
    _lz.ARCH_TO_2D_MAPPINGS["deepseek_v4"] = (experts_cls, existing)
    print(f"[arch-2d-mtp] extended ARCH_TO_2D_MAPPINGS with {len(new_entries)} "
          f"mtp.\\d+.mlp.experts.* entries", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="BF16 input model dir")
    p.add_argument("--output", required=True, help="W4A16-FP8 output dir")
    p.add_argument("--samples", type=int, default=768,
                   help="number of calibration samples (predecessor pinned 768)")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4,
                   help="per-rank batch size (predecessor pinned 4; bs>=8 OOMs even with offload_hessians)")
    p.add_argument("--offload-dir", default="/scratch/offload")
    p.add_argument("--checkpoint-dir", default="/scratch/weights/checkpoints",
                   help="Per-subgraph checkpoint dir for resume-on-crash")
    return p.parse_args()


def preprocess(example, tokenizer):
    """V4 has no Jinja chat template — encode manually per
    https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding
    """
    BOS = "<｜begin▁of▁sentence｜>"
    EOS = "<｜end▁of▁sentence｜>"
    text = BOS
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text += content
        elif role == "user":
            text += f"<｜User｜>{content}"
        elif role == "assistant":
            text += f"<｜Assistant｜></think>{content}{EOS}"
    return {"text": text}


def main():
    args = parse_args()
    os.environ.setdefault("GPTQ_CKPT_DIR", args.checkpoint_dir)

    # Extend the arch-to-2d mappings BEFORE entering load_linearized_moe.
    extend_arch_to_2d_mappings_for_mtp()

    import torch
    import torch.distributed as dist
    from compressed_tensors.quantization.quant_scheme import (
        FP8_BLOCK,
        W4A16,
        QuantizationScheme,
    )
    from compressed_tensors.offload import init_dist, load_offloaded_model
    from datasets import load_dataset

    from llmcompressor import oneshot
    from llmcompressor.datasets.utils import get_rank_partition
    from llmcompressor.modeling.moe.linearize import load_linearized_moe
    from llmcompressor.modifiers.quantization import GPTQModifier

    from scripts.gptq_checkpoint import (
        install_subgraph_checkpoint_hook,
        list_completed_subgraphs,
    )

    # ---- distributed init with predecessor's NCCL settings ----------------
    init_dist()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main = rank == 0
    if is_main:
        print(f"[dist] world_size={world_size} rank={rank}", flush=True)
        print(f"[quant] input={args.input}")
        print(f"[quant] output={args.output}")
        print(f"[quant] samples={args.samples}  max_seq_len={args.max_seq_len}  batch_size={args.batch_size}")

    # ---- 1. load model with offload + linearized MoE ---------------------
    # Stack two context managers:
    #  - load_offloaded_model: extends accelerate's device_map vocabulary to
    #    include "auto_offload" (the predecessor's path; spills full model to
    #    disk + onloads sequentially as needed). MUST be entered via a direct
    #    `with` statement — load_offloaded_model uses _get_caller_frame() to
    #    find AutoModelForCausalLM in the caller's globals, and indirection
    #    via ExitStack breaks the frame walk.
    #  - load_linearized_moe: patches from_pretrained to swap
    #    DeepseekV4Experts for LinearExperts2D (256 individual Linear modules
    #    per MoE block), and registers the checkpoint conversion mapping that
    #    renames `mlp.experts.X.{w1,w2,w3}` to `mlp.experts.X.{gate,up,down}_proj`.
    if is_main:
        print("[quant] loading model with auto_offload + linearize_moe...", flush=True)
    t0 = time.time()
    with load_offloaded_model():
        with load_linearized_moe():
            model = AutoModelForCausalLM.from_pretrained(
                args.input,
                torch_dtype="auto",
                device_map="auto_offload",
                offload_folder=args.offload_dir,
            )
    if is_main:
        print(f"[quant] model loaded in {time.time()-t0:.1f}s", flush=True)

    # ---- 1.4 MTP tensor-value verification (mandatory) ------------------
    # Catches silent random-init regressions where the module count looks
    # right but the actual weights are random (e.g. wrong layer_type =>
    # empty compressor submodules => _init_weights random-initializes,
    # OR missing conversion mapping for mtp.* paths => keys arrive in
    # upstream form but module is HF-named => silent skip + random-init).
    # See FINDINGS_FOR_SIBLING.md "Bug N1" and "Bug N2" for the full
    # diagnostic write-up. Cost: ~50 ms per launch.
    if is_main:
        import safetensors.torch as st
        from pathlib import Path as _Path
        loaded_w = model.model.mtp[0].self_attn.q_a_proj.weight
        source_w = None
        source_path = None
        for shard in sorted(_Path(args.input).glob("model-*.safetensors")):
            with st.safe_open(shard, framework="pt") as f:
                if "mtp.0.attn.wq_a.weight" in f.keys():
                    source_w = f.get_tensor("mtp.0.attn.wq_a.weight")
                    source_path = str(shard)
                    break
        assert source_w is not None, (
            "could not find mtp.0.attn.wq_a.weight in source — bf16-mtp "
            "checkpoint is missing the MTP block; re-run Phase 1 dequant."
        )
        diff = (loaded_w.detach().cpu().float() - source_w.cpu().float()).abs().max().item()
        print(f"[mtp-verify] source: {source_path}", flush=True)
        print(f"[mtp-verify] loaded:  sum={loaded_w.sum().item():.6f}, "
              f"abs_mean={loaded_w.abs().mean().item():.6f}", flush=True)
        print(f"[mtp-verify] source:  sum={source_w.sum().item():.6f}, "
              f"abs_mean={source_w.abs().mean().item():.6f}", flush=True)
        print(f"[mtp-verify] max_diff: {diff:.2e}", flush=True)
        assert diff < 1e-4, (
            f"MTP weight mismatch! max_diff={diff} on mtp.0.attn.wq_a / "
            f"mtp.0.self_attn.q_a_proj. Symptom of wrong layer_type or "
            f"missing conversion mapping for mtp.* paths. See "
            f"FINDINGS_FOR_SIBLING.md Bugs N1/N2."
        )
        print(f"[mtp-verify] OK — MTP weights loaded correctly", flush=True)

    # ---- 1.5 MTP module walk gate ----------------------------------------
    # Mandatory pre-launch check per the user-approved Option D verification
    # plan: count main vs MTP expert modules. Bail loud if MTP=0.
    main_experts = sum(
        1 for n, _ in model.named_modules()
        if n.count(".mlp.experts.") == 1 and ".layers." in n
        and n.rsplit(".", 1)[1] in {"gate_proj", "up_proj", "down_proj"}
    )
    mtp_experts = sum(
        1 for n, _ in model.named_modules()
        if n.count(".mlp.experts.") == 1 and ".mtp." in n
        and n.rsplit(".", 1)[1] in {"gate_proj", "up_proj", "down_proj"}
    )
    if is_main:
        print(f"[mtp-gate] main expert Linears (layers.*.mlp.experts.*.{{gate,up,down}}_proj): {main_experts}",
              flush=True)
        print(f"[mtp-gate] MTP expert Linears (mtp.*.mlp.experts.*.{{gate,up,down}}_proj):    {mtp_experts}",
              flush=True)
    if mtp_experts == 0:
        if is_main:
            print("[mtp-gate] FATAL: zero MTP expert modules found. The MTP shim "
                  "or ARCH_TO_2D_MAPPINGS extension is not wiring through. "
                  "Aborting before the multi-hour run.", flush=True)
        sys.exit(2)

    # ---- 2. tokenizer + calibration dataset ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.input)
    if is_main:
        print(f"[dataset] loading {args.samples} ultrachat_200k samples", flush=True)
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    ds = load_dataset(
        DATASET_ID,
        split=get_rank_partition(DATASET_SPLIT, args.samples),
    )
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda ex: preprocess(ex, tokenizer))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # ---- 3. GPTQ recipe (predecessor's, verbatim, plus MTP-aware regex) --
    # Predecessor recipe: FP8_BLOCK attention + W4A16 routed experts,
    # everything else BF16. transformers post-rename names (`self_attn.q_a_proj`,
    # `mlp.experts.X.gate_proj`) are what llmcompressor sees during calibration
    # after `load_linearized_moe` + `register_checkpoint_conversion_mapping`
    # have done their renames.
    #
    # MTP-aware: the regexes use `.*` so they match both `model.layers.X.*`
    # and `model.mtp.X.*` paths. No regex changes needed beyond that — the
    # `*` after `re:` is enough to match the MTP prefix too.
    recipe = GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                    r"re:.*self_attn\.compressor\.(gate_proj|kv_proj)$",
                    r"re:.*self_attn\.compressor\.indexer\.(gate_proj|kv_proj|q_b_proj|weights_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "experts": QuantizationScheme(
                targets=[
                    r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
                ],
                **W4A16,
            ),
        },
        ignore=["lm_head"],
        offload_hessians=True,   # required — see predecessor phase3b-recovery.md
        dampening_frac=0.1,
    )

    # ---- 4. per-subgraph checkpoint hook --------------------------------
    completed = list_completed_subgraphs(verbose=is_main)
    install_subgraph_checkpoint_hook(
        rank=rank,
        world_size=world_size,
        completed=completed,
        verbose=is_main,
    )
    if is_main:
        print(f"[ckpt] resuming with {len(completed)} completed subgraphs", flush=True)

    # ---- 5. oneshot --------------------------------------------------------
    if is_main:
        print("[quant] starting oneshot calibration "
              f"(samples={args.samples} batch={args.batch_size} seq={args.max_seq_len})",
              flush=True)
    t_oneshot = time.time()
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
        sequential_targets=["DeepseekV4DecoderLayer", "DeepseekV4NextNPredictor"],
        batch_size=args.batch_size,
    )
    if is_main:
        print(f"[quant] oneshot done in {(time.time()-t_oneshot)/3600:.2f}h", flush=True)

    # ---- 6. save -----------------------------------------------------------
    if is_main:
        print(f"[quant] saving to {args.output}...", flush=True)
        t_save = time.time()
    model.save_pretrained(args.output, save_compressed=True)
    if is_main:
        tokenizer.save_pretrained(args.output)
        print(f"[quant] save done in {time.time()-t_save:.0f}s", flush=True)
        print(f"[quant] DONE. Output at {args.output}", flush=True)


if __name__ == "__main__":
    main()
