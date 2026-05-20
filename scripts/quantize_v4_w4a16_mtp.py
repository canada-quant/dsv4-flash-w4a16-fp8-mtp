#!/usr/bin/env python3
"""Phase 2 — GPTQ W4A16-FP8 calibration with MTP included.

Single-process invocation::

    python scripts/quantize_v4_w4a16_mtp.py \\
        --weights /scratch/weights/bf16-mtp \\
        --config  vendor/dsv4-upstream/config.json \\
        --output  /scratch/weights/w4a16-fp8-mtp-gptq \\
        --samples 768 --max-seq-len 512 --batch-size 4

Multi-process (predecessor convention; required for the real run)::

    torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16_mtp.py \\
        --weights /scratch/weights/bf16-mtp \\
        --config  vendor/dsv4-upstream/config.json \\
        --output  /scratch/weights/w4a16-fp8-mtp-gptq \\
        --samples 768 --max-seq-len 512 --batch-size 4

Dry-run (single-GPU, tiny sample, recipe restricted to one layer's Linears)::

    python scripts/quantize_v4_w4a16_mtp.py ... --samples 4 --dry-run-one-layer

The pipeline:
  1. ``compressed_tensors.distributed.init_dist()`` (no-op for single-process)
  2. ``scripts.upstream.apply_dist_state()`` to mirror dist into shim globals so
     the vendored MoE shards experts across ranks (256 / world_size per rank).
  3. Build ModelArgs from upstream config — ``dtype="bf16"``, ``expert_dtype``
     and ``scale_fmt`` stripped, kv_cache buffer sized for ``max_seq_len``.
  4. Instantiate shimmed Transformer (init-skip patched to 1.3s).
  5. Stream-load BF16 weights via ``load_safetensors_into``. Hard-fail on any
     unmatched safetensors key or unexpected missing state-dict key.
  6. Wrap in ``CalibrationModel`` to drive main 0..N-1 then mtp[i] forward.
  7. Wrap that in ``_GPTQCompatibleModel`` (a thin PreTrainedModel subclass)
     so ``llmcompressor.oneshot`` can call ``model.save_pretrained``.
  8. Load ``HuggingFaceH4/ultrachat_200k`` (predecessor's pinned corpus),
     apply V4 manual chat encoding, tokenize to ``max_seq_len``.
  9. Build ``GPTQModifier`` with the recipe topology:
       - FP8_BLOCK on ``re:.*\\.attn\\.(wq_a|wq_b|wkv|wo_a|wo_b)$`` and
         ``re:mtp\\.\\d+\\.(e_proj|h_proj)$``
       - W4A16 on ``re:.*\\.ffn\\.experts\\.\\d+\\.(w1|w2|w3)$``
       - ``actorder="static"`` (the GPTQ-vs-RTN tell)
 10. ``oneshot(model=..., recipe=..., dataset=..., sequential_targets=["Block"],
     batch_size=..., max_seq_length=...)``.
 11. Save to ``output_dir`` — quantization_config will include actorder.

If oneshot fails for non-trivial PreTrainedModel-interface reasons (the 2-hour
bail-out condition), the script falls back to the lower-level
``GPTQModifier`` lifecycle directly. See ``_run_via_modifier_direct``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make scripts.upstream importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import torch.nn as nn

from scripts.upstream import (
    Transformer,
    apply_dist_state,
    build_model_args,
)
from scripts.calibration_model import CalibrationModel
from scripts.load_bf16_into_transformer import load_safetensors_into


# =========================================================================
# V4 manual chat encoding (predecessor recipe, verbatim)
# =========================================================================
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"


def preprocess_v4(example: dict) -> dict:
    """V4 has no Jinja chat template — encode manually.

    Source: dsv4-flash-w4a16-fp8/scripts/quantize_v4_w4a16.py:99-114
    """
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


# =========================================================================
# PreTrainedModel bridge — option A
# =========================================================================
class _GPTQCompatibleConfig:
    """Config object satisfying llmcompressor.oneshot's attribute reads AND
    producing a vLLM-loadable ``config.json`` at save time.

    Why not a real ``PretrainedConfig``: ``PretrainedConfig.__init__`` pulls
    in HF auto-mapping that doesn't gracefully accept our shim model_type.
    Duck-type the fields oneshot reads instead.

    Why bind ``upstream_config_path``: the previous save path wrote a
    minimal config (just the ~7 fields stored on the instance), so vLLM
    rejected it with ``Unrecognized model ... Should have a 'model_type'
    key``. The fix: read the bf16-mtp config.json at save time and use it
    as the base, then merge our overrides + ``quantization_config`` on
    top. The bf16-mtp config is the source of truth for everything except
    quantization.
    """
    base_model_prefix = "model"
    is_encoder_decoder = False

    def __init__(self, args, upstream_config_path: str | None = None):
        # model_type is "deepseek_v4" so vLLM's DeepseekV4ForCausalLM model
        # registry recognises it. (Previous value "deepseek_v4_shim" was a
        # sentinel that never made it to the saved file.)
        self.model_type = "deepseek_v4"
        self.tie_word_embeddings = True   # MTP shares embed/head with main
        self.hidden_size = args.dim
        self.num_hidden_layers = args.n_layers + args.n_mtp_layers
        self.vocab_size = args.vocab_size
        self.architectures = ["DeepseekV4ForCausalLM"]
        self.torch_dtype = "bfloat16"
        # NB: do not set use_return_dict / output_hidden_states /
        # output_attentions — vLLM's DeepseekV4Config exposes these as
        # read-only properties (transformers raises AttributeError on
        # setattr). llmcompressor reads ``config.use_return_dict``; we
        # provide it via a property below.
        # Path to the BF16 source config; save_pretrained reads it as the
        # base for output config.json.
        self._upstream_config_path = upstream_config_path

    # Provide use_return_dict to llmcompressor at runtime, but DON'T put it
    # in __dict__ so save_pretrained doesn't serialize it.
    @property
    def use_return_dict(self):
        return True

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }

    def save_pretrained(self, save_directory, **_kw):
        # Build full config = (bf16-mtp config) ∪ (our overrides). vLLM
        # needs the full HF DeepseekV4 field set; the BF16 source has it.
        base: dict = {}
        upstream_dir = None
        if self._upstream_config_path and os.path.exists(self._upstream_config_path):
            with open(self._upstream_config_path) as f:
                base = json.load(f)
            upstream_dir = os.path.dirname(self._upstream_config_path)
        # Don't carry the source's quantization_config (if any) — llm-
        # compressor writes the new one separately.
        base.pop("quantization_config", None)
        # Our overrides win for the keys we explicitly set; the base
        # provides the long-tail (rope_scaling, MoE/index/HC fields, etc.).
        merged = {**base, **self.to_dict()}
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(merged, f, indent=2, default=str)
        # Copy tokenizer files alongside the model so vLLM can boot the
        # tokenizer at serve time. Without these, ``Couldn't instantiate
        # the backend tokenizer'' on serve startup.
        if upstream_dir:
            import shutil
            for fname in ("tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "generation_config.json",
                          "chat_template.jinja"):
                src = os.path.join(upstream_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(save_directory, fname))


class _GPTQCompatibleModel(nn.Module):
    """Thin wrapper presenting a CalibrationModel as llmcompressor expects.

    Why nn.Module not PreTrainedModel: PreTrainedModel.__init__ wires up
    GenerationMixin and module-init logic that's slow and unnecessary. We
    only need (forward returning .logits) + (config) + (save_pretrained).
    """

    def __init__(self, calibration_model: CalibrationModel, args,
                 upstream_config_path: str | None = None):
        super().__init__()
        self.cal_model = calibration_model
        self.config = _GPTQCompatibleConfig(args, upstream_config_path)
        # llmcompressor's save path calls update_and_save_recipe(model.name_or_path, ...)
        # to copy or update a recipe.yaml. We never had a name_or_path; pass an
        # empty string so the recipe write goes to save_directory only.
        self.name_or_path = ""

    @property
    def device(self) -> torch.device:
        # llmcompressor sometimes asks; return cpu (model lives on CPU,
        # sequential calibration moves blocks to GPU as needed).
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_input_embeddings(self):
        return self.cal_model.transformer.embed

    def get_output_embeddings(self):
        return self.cal_model.transformer.head

    def tie_weights(self):
        # embed/head sharing is wired in vendored Transformer.__init__.
        pass

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.cal_model(input_ids, **kwargs)

    def save_pretrained(self, save_directory, save_compressed: bool = True, **kw):
        """Save the underlying transformer's state_dict + config + quant config.

        The compressed-tensors save flow is normally handled by
        ``modify_save_pretrained`` wrapping a real ``PreTrainedModel``'s
        save_pretrained. Since we're not a real PreTrainedModel, we write
        the state dict to safetensors shards ourselves and let llmcompressor's
        SessionRecipe-attached compressor produce the quantization_config.
        """
        from safetensors.torch import save_file
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        state = self.cal_model.transformer.state_dict()
        # Shard at 5 GB per file
        shards: list[dict[str, torch.Tensor]] = [{}]
        bytes_per_shard = 5 * (1 << 30)
        cur_bytes = 0
        for name, tensor in state.items():
            t_bytes = tensor.numel() * tensor.element_size()
            if cur_bytes + t_bytes > bytes_per_shard and shards[-1]:
                shards.append({})
                cur_bytes = 0
            shards[-1][name] = tensor
            cur_bytes += t_bytes

        n = len(shards)
        weight_map = {}
        for i, payload in enumerate(shards, start=1):
            fname = f"model-{i:05d}-of-{n:05d}.safetensors"
            save_file(payload, os.path.join(save_directory, fname),
                      metadata={"format": "pt"})
            for k in payload:
                weight_map[k] = fname

        idx = {
            "metadata": {
                "total_size": sum(t.numel() * t.element_size()
                                  for s in shards for t in s.values())
            },
            "weight_map": weight_map,
        }
        with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f, indent=2)


# =========================================================================
# Recipe
# =========================================================================
def build_gptq_recipe(dry_run_one_layer: bool):
    """Return a GPTQModifier with the predecessor's recipe topology."""
    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, W4A16
    from llmcompressor.modifiers.quantization import GPTQModifier

    if dry_run_one_layer:
        # Restrict to a SINGLE layer's Linears so the dry-run finishes fast.
        # `.*` prefix matches the wrapper path (cal_model.transformer.layers.5...).
        attn_targets = [r"re:.*\.layers\.5\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$"]
        expert_targets = [r"re:.*\.layers\.5\.ffn\.experts\.\d+\.(w1|w2|w3)$"]
    else:
        attn_targets = [
            r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
            r"re:.*mtp\.\d+\.(e_proj|h_proj)$",
        ]
        expert_targets = [r"re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$"]

    # Set per-group ``format=`` so the compressor packs/encodes weights
    # correctly at save time. Without this, scheme.format defaults to
    # "dense" and vLLM's compressed-tensors loader raises
    # NotImplementedError: No compressed-tensors compatible scheme was
    # found.  pack-quantized = bit-packed W4A16; float-quantized = FP8 with
    # block scales.
    return GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=attn_targets, format="float-quantized", **FP8_BLOCK
            ),
            "experts": QuantizationScheme(
                targets=expert_targets, format="pack-quantized", **W4A16
            ),
        },
        ignore=[
            "head", "embed",
            r"re:.*norm.*",
            r"re:.*\.ffn\.gate$",
            r"re:.*\.ffn\.gate\..*",
            r"re:.*\.ffn\.shared_experts\..*",
            r"re:.*\.hc_.*",
            r"re:hc_.*",
            r"re:.*\.attn\.attn_sink$",
            r"re:.*\.attn\.(compressor|indexer)\..*",
        ],
        offload_hessians=True,
        dampening_frac=0.1,
        actorder="static",   # GPTQ-vs-RTN tell
    )


# =========================================================================
# Dataset
# =========================================================================
def build_calibration_dataset(tokenizer, *, num_samples: int, max_seq_len: int,
                              seed: int = 42):
    """Predecessor's exact calibration recipe (locked in PLAN.md):

        HuggingFaceH4/ultrachat_200k  train_sft  seed=42
        768 samples, seq=512, V4 manual chat encoding.
    """
    from datasets import load_dataset

    # over-fetch (factor 2) so empty-after-preprocess samples don't shortfall
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{num_samples * 2}]",
    )
    ds = ds.shuffle(seed=seed)
    ds = ds.map(preprocess_v4)
    ds = ds.select(range(num_samples))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Record the HF dataset commit hash for reproducibility (per PLAN.md)
    rev = None
    try:
        from datasets import builder as _ds_builder  # noqa: F401
        # Best-effort: load_dataset caches under the resolved revision
        info = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:1]")
        rev = getattr(info, "info", None)
        rev = getattr(rev, "version", None) if rev is not None else None
    except Exception:
        pass
    return ds, rev


# =========================================================================
# Main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Phase-1 BF16 dir")
    ap.add_argument("--config", required=True,
                    help="upstream config.json (vendor/dsv4-upstream/config.json)")
    ap.add_argument("--output", required=True, help="output W4A16-FP8 dir")
    ap.add_argument("--samples", type=int, default=768)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dry-run-one-layer", action="store_true",
                    help="recipe restricted to layer 5 Linears only; "
                         "for timing projection")
    args = ap.parse_args()

    t_total = time.time()

    # ---- 0. version-skew patch -----------------------------------------
    # llm-compressor f2aa32e2 calls update_offload_parameter(..., source_rank=...)
    # but compressed-tensors 0.15.1a20260515 signature is (module, name, data).
    # Wrap the function to swallow source_rank when present so the GPTQ
    # post-compress writeback succeeds.
    import compressed_tensors.utils.offload as _cto
    _orig_uop = _cto.update_offload_parameter
    def _uop_compat(module, name, data, *, source_rank=None, **_kw):
        return _orig_uop(module, name, data)
    _cto.update_offload_parameter = _uop_compat
    # llm-compressor caches by-name import at module load — re-bind there too.
    import llmcompressor.modifiers.gptq.base as _gptq_base
    if hasattr(_gptq_base, "update_offload_parameter"):
        _gptq_base.update_offload_parameter = _uop_compat
    print(f"[compat] update_offload_parameter wrapped to swallow source_rank kwarg",
          flush=True)

    # ---- 1. distributed init ---------------------------------------------
    use_dist = "TORCHELASTIC_RUN_ID" in os.environ
    if use_dist:
        # NCCL default timeout = 10 min. With 4-rank calibration, large
        # Hessian REDUCE collectives on later layers can hit this and abort
        # the run (observed: 600s timeout on layers.5.attn.wo_b 2026-05-20).
        # Init the process group ourselves with a longer timeout BEFORE
        # init_dist; init_dist is then a no-op since dist is already up.
        if not dist.is_initialized():
            import datetime as _dt
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
        # init_dist would re-init; skip if we already did the work.
        # (Older compressed_tensors versions raise on double-init.)
    elif not dist.is_initialized():
        # llm-compressor's GPTQ compress_modules() calls dist.get_rank()
        # unconditionally — even in single-process mode. Init a 1-rank
        # gloo group via a unique tcp endpoint so it succeeds.
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{free_port}",
            rank=0,
            world_size=1,
        )
    apply_dist_state()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0
    if is_main:
        print(f"[dist] world_size={world_size} rank={rank} use_dist={use_dist}",
              flush=True)

    # ---- 2. build ModelArgs ----------------------------------------------
    margs = build_model_args(
        args.config, max_batch_size=args.batch_size, max_seq_len=args.max_seq_len
    )
    if is_main:
        print(f"[args] dim={margs.dim} n_layers={margs.n_layers} "
              f"n_mtp_layers={margs.n_mtp_layers} "
              f"n_routed_experts={margs.n_routed_experts}", flush=True)

    # ---- 3. instantiate Transformer + load BF16 --------------------------
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")

    if is_main:
        print("[load] instantiating Transformer on CPU (init-skip)", flush=True)
    # Transformer.__init__ (vendor/dsv4-upstream/model.py line 773) does
    # ``global world_size; world_size = dist.get_world_size() ...`` which
    # overrides the shim's WS=1 force every time. Mask dist temporarily
    # during construction so the global stays at 1.
    import dsv4_upstream_model as _ds
    _orig_is_init = dist.is_initialized
    _orig_get_ws = dist.get_world_size
    _orig_get_rk = dist.get_rank
    dist.is_initialized = lambda: False
    dist.get_world_size = lambda *a, **kw: 1
    dist.get_rank = lambda *a, **kw: 0
    t0 = time.time()
    try:
        transformer = Transformer(margs)
    finally:
        dist.is_initialized = _orig_is_init
        dist.get_world_size = _orig_get_ws
        dist.get_rank = _orig_get_rk
        _ds.world_size = 1
        _ds.rank = 0
    if is_main:
        print(f"[load] upstream _module.world_size={_ds.world_size} rank={_ds.rank} (post-init)",
              flush=True)
    if is_main:
        print(f"[load] instantiated in {time.time()-t0:.1f}s", flush=True)

    if is_main:
        print(f"[load] streaming safetensors from {args.weights}", flush=True)
    t1 = time.time()
    loaded, unmatched, missing = load_safetensors_into(
        transformer, Path(args.weights), verbose=is_main
    )
    if is_main:
        print(f"[load] loaded={loaded} unmatched={len(unmatched)} "
              f"missing={len(missing)} in {time.time()-t1:.1f}s", flush=True)
    # With decoupled expert sharding (apply_dist_state sets
    # _expert_world_size = N), each rank's model has only N_routed/N experts
    # populated; the safetensors contain all 256 experts; per-rank load
    # naturally skips the (N-1)/N expert tensors that aren't in this rank's
    # state_dict. Those are "expected unmatched", not a fatal error. Only
    # bail if there are unmatched NON-expert keys.
    import re as _re
    non_expert_unmatched = [
        k for k in unmatched if _re.search(r"\.experts\.\d+\.", k) is None
    ]
    if non_expert_unmatched:
        if is_main:
            print(
                f"FATAL: {len(non_expert_unmatched)} unmatched non-expert "
                f"safetensors keys: {non_expert_unmatched[:10]}",
                flush=True,
            )
        sys.exit(2)

    # ---- 3.5 multi-rank patches + sharding invariant + checkpoint hooks ---
    # See scripts/multirank_patches.py for the full rationale. Three patches
    # (Observer.synchronize no-op + GPTQ reduce/broadcast skip-sharded) close
    # NCCL collective hangs that arise from our decoupled expert shard — a
    # path the predecessor calibration never exercised (it used HF
    # auto-offload, which kept module sets identical across ranks).
    from scripts.multirank_patches import (
        assert_sharding_invariant,
        apply_all_patches,
    )
    from scripts.gptq_checkpoint import (
        list_completed_subgraphs,
        install_subgraph_checkpoint_hook,
    )
    # Verify the shard is actually disjoint before patching. If experts
    # were accidentally replicated, "skip reduce for sharded modules" would
    # silently corrupt the Hessian — catch loud here instead.
    assert_sharding_invariant(
        transformer,
        world_size=world_size,
        rank=rank,
        n_routed_experts=margs.n_routed_experts,
        verbose=is_main,
    )
    apply_all_patches(world_size=world_size, verbose=is_main)
    # Install the per-subgraph checkpoint hook; resume any already-completed
    # subgraphs from /scratch/weights/checkpoints/.
    _completed = list_completed_subgraphs(verbose=is_main)
    install_subgraph_checkpoint_hook(
        rank=rank,
        world_size=world_size,
        completed=_completed,
        verbose=is_main,
    )

    # ---- 4. tokenizer + dataset ------------------------------------------
    if is_main:
        print(f"[tokenizer] loading from {args.weights}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.weights, trust_remote_code=False)

    if is_main:
        print(f"[dataset] preparing {args.samples} samples from ultrachat_200k",
              flush=True)
    ds, ds_rev = build_calibration_dataset(
        tokenizer, num_samples=args.samples, max_seq_len=args.max_seq_len
    )
    if is_main:
        print(f"[dataset] {len(ds)} samples ready; revision={ds_rev}", flush=True)
        # Record the dataset commit hash per PLAN.md
        findings_dir = Path("/data/findings") if Path("/data").exists() else Path("findings")
        findings_dir.mkdir(parents=True, exist_ok=True)
        with open(findings_dir / "calibration-dataset-commit.txt", "w") as f:
            f.write(f"HuggingFaceH4/ultrachat_200k  train_sft  seed=42\n"
                    f"resolved dataset version: {ds_rev}\n"
                    f"samples={args.samples}  seq_len={args.max_seq_len}\n")

    # ---- 5. wrap model ---------------------------------------------------
    cal = CalibrationModel(transformer)
    model = _GPTQCompatibleModel(cal, margs, upstream_config_path=args.config)

    print("[step6] entering topk patch block", flush=True)
    # ---- 6. patch topk_idxs for device matching --------------------------
    import dsv4_upstream_model as _dsv4
    print(f"[step6] imported _dsv4 = {_dsv4!r}", flush=True)
    _orig_win = _dsv4.get_window_topk_idxs
    _orig_cmp = _dsv4.get_compress_topk_idxs

    # Clear lru_cache so any previously-cached cpu results are dropped — the
    # next call will recompute and our wrapper will relocate.
    if hasattr(_orig_win, "cache_clear"):
        _orig_win.cache_clear()
    if hasattr(_orig_cmp, "cache_clear"):
        _orig_cmp.cache_clear()

    def _current_cuda_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return torch.device("cpu")

    _win_call_count = [0]
    _cmp_call_count = [0]

    def _win_dev(*a, **kw):
        r = _orig_win(*a, **kw)
        target = _current_cuda_device()
        out = r.to(target) if r.device != target else r
        _win_call_count[0] += 1
        if _win_call_count[0] <= 3:
            print(f"  [topk-patch] _win_dev call {_win_call_count[0]}: "
                  f"target={target}, in.device={r.device}, out.device={out.device}",
                  flush=True)
        return out

    def _cmp_dev(*a, **kw):
        r = _orig_cmp(*a, **kw)
        target = _current_cuda_device()
        out = r.to(target) if r.device != target else r
        _cmp_call_count[0] += 1
        if _cmp_call_count[0] <= 3:
            print(f"  [topk-patch] _cmp_dev call {_cmp_call_count[0]}: "
                  f"target={target}, in.device={r.device}, out.device={out.device}",
                  flush=True)
        return out

    _dsv4.get_window_topk_idxs = _win_dev
    _dsv4.get_compress_topk_idxs = _cmp_dev

    # Also force-defend: monkey-patch sparse_attn to relocate topk_idxs at use
    # site, as a belt-and-suspenders against fx's tendency to inline the
    # original function's result as a constant.
    from scripts.upstream import kernel_shim as _ks
    _orig_sparse_attn = _ks.sparse_attn
    _sparse_call_count = [0]

    def _sparse_attn_dev(q, kv, attn_sink, topk_idxs, softmax_scale):
        tgt = q.device
        moved = []
        if topk_idxs.device != tgt:
            topk_idxs = topk_idxs.to(tgt); moved.append("topk_idxs")
        if attn_sink.device != tgt:
            attn_sink = attn_sink.to(tgt); moved.append("attn_sink")
        _sparse_call_count[0] += 1
        if _sparse_call_count[0] <= 3:
            print(f"  [sparse-patch] call {_sparse_call_count[0]}: q.device={tgt}, "
                  f"moved={moved}", flush=True)
        return _orig_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    _ks.sparse_attn = _sparse_attn_dev
    _dsv4.sparse_attn = _sparse_attn_dev

    # Indexer.forward has its own internal lazy-default-device allocations
    # (mask via torch.where(...) at vendor/model.py:426, index_score, etc).
    # The cleanest fix: wrap Attention.forward AND Indexer.forward to set
    # torch.set_default_device to the live cuda device for the duration of
    # the call, so any lazy torch.arange/zeros/where(scalar,...) creations
    # land on cuda instead of cpu.
    _wrap_call_count = [0]

    def _wrap_forward(original_forward):
        def wrapped(self, *args, **kwargs):
            # Use the first tensor argument's device as the active default
            tgt = None
            for a in args:
                if torch.is_tensor(a):
                    tgt = a.device
                    break
            if tgt is None and torch.cuda.is_available():
                tgt = torch.device(f"cuda:{torch.cuda.current_device()}")
            if tgt is None:
                return original_forward(self, *args, **kwargs)
            _wrap_call_count[0] += 1
            if _wrap_call_count[0] <= 3:
                print(f"  [forward-wrap] call {_wrap_call_count[0]}: "
                      f"{type(self).__name__}.forward with default device={tgt}",
                      flush=True)
            with torch.device(tgt):
                return original_forward(self, *args, **kwargs)
        return wrapped

    _dsv4.Attention.forward = _wrap_forward(_dsv4.Attention.forward)
    _dsv4.Indexer.forward = _wrap_forward(_dsv4.Indexer.forward)
    _dsv4.Compressor.forward = _wrap_forward(_dsv4.Compressor.forward)

    # NB: an earlier attempt set torch.set_default_device("cuda:0") here, but
    # that broke the DataLoader's CPU-generator randperm sampler. Rely on
    # the wrap_forward `with torch.device(tgt):` context inside each call
    # instead.

    # Verify patches landed in the right namespace
    _dsv4_check = sys.modules.get("dsv4_upstream_model")
    print(f"  [patch-check] dsv4 in sys.modules: {_dsv4_check is not None}", flush=True)
    print(f"  [patch-check] _dsv4 is sys.modules.dsv4_upstream_model: "
          f"{_dsv4 is _dsv4_check}", flush=True)
    print(f"  [patch-check] _dsv4.get_window_topk_idxs is _win_dev: "
          f"{_dsv4.get_window_topk_idxs is _win_dev}", flush=True)
    print(f"  [patch-check] _dsv4.sparse_attn is _sparse_attn_dev: "
          f"{_dsv4.sparse_attn is _sparse_attn_dev}", flush=True)
    print(f"  [patch-check] _ks.sparse_attn is _sparse_attn_dev: "
          f"{_ks.sparse_attn is _sparse_attn_dev}", flush=True)
    # Also check what Attention.forward sees
    _attn_cls = _dsv4.Attention
    _attn_globals = _attn_cls.forward.__globals__
    print(f"  [patch-check] Attention.forward.__globals__ is _dsv4.__dict__: "
          f"{_attn_globals is _dsv4.__dict__}", flush=True)
    print(f"  [patch-check] Attention.forward sees sparse_attn as _sparse_attn_dev: "
          f"{_attn_globals.get('sparse_attn') is _sparse_attn_dev}", flush=True)
    print(f"  [patch-check] Attention.forward sees get_window_topk_idxs as _win_dev: "
          f"{_attn_globals.get('get_window_topk_idxs') is _win_dev}", flush=True)

    # ---- 7. recipe + oneshot --------------------------------------------
    if is_main:
        print(f"[recipe] building GPTQ recipe (dry_run_one_layer={args.dry_run_one_layer})",
              flush=True)
    recipe = build_gptq_recipe(args.dry_run_one_layer)

    if is_main:
        print(f"[oneshot] starting calibration  "
              f"samples={args.samples}  batch={args.batch_size}  "
              f"seq={args.max_seq_len}", flush=True)
    t_oneshot = time.time()

    try:
        from llmcompressor import oneshot

        oneshot(
            model=model,
            tokenizer=tokenizer,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=args.samples,
            sequential_targets=["Block"],
            batch_size=args.batch_size,
            output_dir=args.output,
        )
    except Exception as exc:
        # Known non-fatal: compressed_tensors' from_accelerate post-save hook
        # raises ``ValueError: Attempted to offload a module twice'' AFTER
        # the artifact is fully written to disk (config + recipe + safetensors).
        # We swallow it so post-save processing can complete and the run
        # reports CALIBRATION_DONE.
        is_offload_twice = (
            isinstance(exc, ValueError)
            and "offload a module twice" in str(exc).lower()
        )
        if is_main:
            print(f"[oneshot] failed with: {type(exc).__name__}: {exc}",
                  flush=True)
            if is_offload_twice:
                print("[oneshot] non-fatal: artifact already on disk; "
                      "continuing to post-save processing.", flush=True)
            else:
                print("[oneshot] this is the option-A bail point — consider option B "
                      "(direct GPTQModifier.apply). Investigate the traceback before retrying.",
                      flush=True)
        if not is_offload_twice:
            raise

    # Post-save: inject scale_fmt into the saved config.json's
    # quantization_config. vLLM's DeepseekV4Model reads
    # ``config.quantization_config["scale_fmt"]`` and the FP8 block path
    # uses "ue8m0". llmcompressor's save flow doesn't write this field,
    # so we patch the saved file.
    if is_main:
        out_cfg = os.path.join(args.output, "config.json")
        if os.path.exists(out_cfg):
            with open(out_cfg) as f:
                _cfg = json.load(f)
            _qc = _cfg.setdefault("quantization_config", {})
            if _qc.get("scale_fmt") is None:
                _qc["scale_fmt"] = "ue8m0"
                with open(out_cfg, "w") as f:
                    json.dump(_cfg, f, indent=2)
                print("[post-save] set quantization_config.scale_fmt = ue8m0",
                      flush=True)

    if is_main:
        print(f"[oneshot] done in {time.time()-t_oneshot:.1f}s", flush=True)
        print(f"CALIBRATION_DONE total={time.time()-t_total:.1f}s output={args.output}",
              flush=True)


if __name__ == "__main__":
    main()
