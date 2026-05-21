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

# dtype handling: norms in DSv4 ship as fp32 (per HF default
# `_keep_in_fp32_modules_strict` heuristic). Their fp32 output then feeds
# bf16 Linear weights inside the block forward, hitting:
#   RuntimeError: expected mat1 and mat2 to have the same dtype, ...
# The kylesayrs canonical example fixes this by setting
# `_keep_in_fp32_modules_strict = set()` BEFORE from_pretrained — but on
# our transformers 5.8.1 + load_offloaded_model stack, that change triggers
# a completely different device_map inference code path that bypasses our
# MTP shim and conversion mapping (empty `_keep_in_fp32_modules_strict`
# pushes us through `_init_infer_auto_device_map` which loads via a
# different mechanism that doesn't see our patches).
#
# Workaround: leave `_keep_in_fp32_modules_strict` alone during load
# (norms ship as fp32 — load completes correctly with our shim), then
# AFTER from_pretrained returns, cast any fp32 params to bf16 inline.
# This is a one-shot cast over the model's param tree (~few seconds).

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

    # ---- compat shim: update_offload_parameter signature skew ------------
    # llm-compressor f2aa32e2's gptq/base.py:321 calls
    # `update_offload_parameter(module, name, data, source_rank=dist.get_rank())`
    # but compressed-tensors 0.15.1a20260515 only accepts (module, name, data).
    # Wrap to swallow the unknown source_rank kwarg. Carried forward from
    # the previous Option A script (see memory:gptq_dryrun_friction item #3).
    import compressed_tensors.utils.offload as _ct_offload
    import llmcompressor.modifiers.gptq.base as _gptq_base
    _orig_uop = _ct_offload.update_offload_parameter
    def _uop_compat(module, name, data, source_rank=None, **kwargs):
        return _orig_uop(module, name, data)
    _ct_offload.update_offload_parameter = _uop_compat
    # rebind on the gptq module too (it imported the symbol at module load)
    if hasattr(_gptq_base, "update_offload_parameter"):
        _gptq_base.update_offload_parameter = _uop_compat

    from scripts.gptq_checkpoint import (
        install_subgraph_checkpoint_hook,
        list_completed_subgraphs,
    )

    # ---- distributed init with predecessor's NCCL settings ----------------
    # compressed_tensors.offload.init_dist calls dist.init_process_group() with
    # NO `timeout=` arg — uses PyTorch's 10-min default. For DSv4-Flash's
    # load_offloaded_model path, the from_accelerate conversion does a
    # broadcast_object_list AFTER rank 0 finishes the full file read +
    # _init_weights pass. The full load can exceed 10 min, hitting a NCCL
    # watchdog timeout (WorkNCCL OpType=BROADCAST Timeout(ms)=600000).
    # Pre-init the process group with a 2-hour timeout so init_dist's
    # call is a no-op (it skips init if dist.is_initialized()).
    if "TORCHELASTIC_RUN_ID" in os.environ:
        import datetime as _dt
        if not dist.is_initialized():
            _rank = int(os.environ["RANK"])
            _local_rank = int(os.environ["LOCAL_RANK"])
            _world_size = int(os.environ["WORLD_SIZE"])
            _device = torch.device(f"cuda:{_local_rank}")
            torch.cuda.set_device(_device)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=_rank,
                world_size=_world_size,
                device_id=_device,
                timeout=_dt.timedelta(hours=2),
            )
            dist.barrier()
        # NOTE: don't call init_dist() — it would re-init unconditionally
        # and fail with "already initialized". We've done the init above.
    else:
        # Single-rank fallback: let init_dist() do its thing (it'll raise if no env)
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

    # ---- 1.3 dtype unification (norms loaded as fp32, model is bf16) ------
    # See module-level docstring above the imports for why this is done here
    # instead of via `_keep_in_fp32_modules_strict = set()` pre-load. The cast
    # happens in-place on each parameter; offloaded params are onloaded by
    # accelerate's hook system when we touch `.data`, so this is safe even
    # under auto_offload device_map.
    fp32_cast_count = 0
    for name, p in model.named_parameters():
        if p.dtype == torch.float32:
            with torch.no_grad():
                p.data = p.data.to(torch.bfloat16)
            fp32_cast_count += 1
    for name, b in model.named_buffers():
        if b.dtype == torch.float32:
            b.data = b.data.to(torch.bfloat16)
    if is_main:
        print(f"[dtype] cast {fp32_cast_count} fp32 params to bf16", flush=True)

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

    # ---- 3. GPTQ recipe — deliberate mixed-precision with BF16 MTP -------
    # Recipe topology:
    #   Main 43 decoder layers:
    #     - attention projections + compressor/indexer → FP8_BLOCK
    #     - routed-expert MLP (256 experts × 3 projections) → W4A16
    #   MTP draft block (`mtp.0.*`):
    #     - ALL params preserved at BF16 (no quantization)
    #
    # Why MTP stays BF16:
    #
    # MTP is the speculative-decoding draft head. Speculative throughput
    # depends on token-acceptance-rate by the verifier — a small precision
    # delta in the draft directly degrades acceptance. DeepSeek's native
    # release leaves MTP at higher precision than the MXFP4 experts; RedHat
    # dropped MTP entirely. We preserve MTP at full BF16 precision while
    # quantizing the main MoE.
    #
    # Cost: ~10 GB more on disk (MTP ~13.2 GB BF16 vs ~3.3 GB W4A16).
    # Benefit: full MTP acceptance-rate, expected ~1.8× decode speedup at
    # `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.
    # 7% size overhead vs potentially 30%+ throughput loss from degraded
    # acceptance — overwhelmingly worth it.
    #
    # This is a deliberate design choice, not accidental. The `ignore=`
    # entries below make it explicit: any path starting with `mtp.` is
    # excluded from both quantization groups, regardless of whether it
    # would otherwise match the attn or expert target regex.
    # CRITICAL: targets= must be anchored at `model.layers.\d+.` so MTP paths
    # (model.mtp.0.*) are NOT matched. The `ignore=re:.*mtp\..*` below is
    # belt-and-suspenders for GPTQ calibration, but compressed_tensors's
    # save_pretrained wrapper does NOT consult `ignore=` at save time — it
    # only checks targets=. So if targets uses unanchored `.*mlp\.experts\.`
    # patterns (matching both main and MTP), MTP experts get RTN-quantized
    # at save regardless of the ignore list. The first smoke iter 7 confirmed
    # this: subgraph 43 (MTP) was empty during GPTQ (ignore worked), but the
    # saved artifact had `model.mtp.0.mlp.experts.0.down_proj.weight_packed`
    # in int4 anyway. Fix: anchor targets at `model\.layers\.\d+\.`.
    recipe = GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:^model\.layers\.\d+\.self_attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                    r"re:^model\.layers\.\d+\.self_attn\.compressor\.(gate_proj|kv_proj)$",
                    r"re:^model\.layers\.\d+\.self_attn\.compressor\.indexer\.(gate_proj|kv_proj|q_b_proj|weights_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "experts": QuantizationScheme(
                targets=[
                    r"re:^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
                ],
                **W4A16,
            ),
        },
        ignore=[
            "lm_head",
            # MTP draft head: defense-in-depth ignore for GPTQ calibration.
            # The targets= regexes above already anchor at `model.layers.`
            # so MTP is excluded by construction; this entry keeps the
            # design choice visible in the recipe even if targets= ever
            # gets loosened.
            r"re:.*mtp\..*",
        ],
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
    # Pre-save: undo the MTP shim's config-list extension. The shim added one
    # extra entry to layer_types and mlp_layer_types so DeepseekV4Attention
    # and DeepseekV4SparseMoeBlock could index by layer_idx for the MTP block
    # (whose layer_idx == num_hidden_layers). But
    # `transformers.models.deepseek_v4.configuration_deepseek_v4.validate_layer_type`
    # asserts `len(layer_types) == num_hidden_layers`. We truncate back so
    # `config.save_pretrained()` validates clean. The MTP `mtp.0.*` weights
    # are already in the model artifact; the saved config doesn't need to
    # carry a layer_type entry for them (transformers' load path doesn't
    # re-index by layer_idx for MTP — it uses our shim or the upstream PR
    # #46127's `DeepseekV4NextNPredictor` class).
    cfg = model.config
    nhl = cfg.num_hidden_layers
    if getattr(cfg, "layer_types", None) is not None and len(cfg.layer_types) > nhl:
        cfg.layer_types = list(cfg.layer_types)[:nhl]
    if getattr(cfg, "mlp_layer_types", None) is not None and len(cfg.mlp_layer_types) > nhl:
        cfg.mlp_layer_types = list(cfg.mlp_layer_types)[:nhl]
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
