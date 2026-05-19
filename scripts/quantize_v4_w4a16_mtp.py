#!/usr/bin/env python3
"""Phase 2 — GPTQ W4A16-FP8 calibration of DeepSeek-V4-Flash, MTP included.

Loads the Phase-1 BF16 dequant into an adapted upstream Transformer
(``scripts.upstream``), wraps it so calibration forward flows through both
the main 43 layers AND the MTP block (so GPTQ hooks fire on every Linear
in mtp.0.*), then invokes ``llmcompressor.oneshot`` with the recipe
specified in PHASE2_DESIGN.md.

Recipe summary (DeepSeek-internal naming throughout):

  attention   FP8_BLOCK  re:.*\\.attn\\.(wq_a|wq_b|wkv|wo_a|wo_b)$
                          re:mtp\\.\\d+\\.(e_proj|h_proj)$
  experts     W4A16      re:.*\\.ffn\\.experts\\.\\d+\\.(w1|w2|w3)$
  ignore                  lm_head, embeddings, *norm*, *gate*, *shared_experts*,
                          *hc_*, *attn_sink*, *compressor*, *indexer*

CLI::

    python scripts/quantize_v4_w4a16_mtp.py \\
        --input  /scratch/weights/bf16-mtp \\
        --output /scratch/weights/w4a16-fp8-mtp \\
        --config vendor/dsv4-upstream/config.json \\
        --samples 768 --batch-size 4 --max-seq-len 512

Smoke test (fast, validates the pipeline before the 8-12h full run)::

    python scripts/quantize_v4_w4a16_mtp.py ... --samples 4 --batch-size 1

Notes on memory + cost:
  - The BF16 model is ~568 GB. Loads into CPU RAM (4 TB on p6-b300.48xlarge).
  - llmcompressor's sequential GPTQ moves one Block at a time to GPU,
    keeping per-Block residency at ~13 GB. With ``offload_hessians=True``
    Hessians stream back to CPU between Linears.
  - The full 768-sample run is 8-12 hours per PLAN.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

# Path setup so scripts.upstream resolves regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from scripts.upstream import Transformer, build_model_args
from scripts.load_bf16_into_transformer import load_safetensors_into


# ----------------------------- model wrapper -----------------------------


class CalibrationModel(nn.Module):
    """Adapter that drives both main + MTP layers from a single forward.

    Upstream Transformer.forward only flows through ``self.layers``; the MTP
    block at ``self.mtp[0]`` is unused at inference time and would therefore
    receive zero calibration activations under a plain forward.

    This wrapper replicates the main forward path explicitly and then *also*
    invokes the MTP block on the final main-layer hidden states. The MTP
    logits are discarded — we only need the activations to flow through MTP
    Linears so GPTQ hooks fire.

    The forward signature accepts ``input_ids=...`` and returns an object
    with ``.logits`` to match what ``llmcompressor.oneshot`` expects from an
    HF-style PreTrainedModel.
    """

    def __init__(self, transformer: Transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids: torch.Tensor, **_unused) -> "_LogitsOut":
        t = self.transformer
        h = t.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, t.hc_mult, 1)
        for layer in t.layers:
            h = layer(h, 0, input_ids)
        # Drive MTP — discard its logits, but its forward fires GPTQ hooks
        # on e_proj, h_proj, attn.*, ffn.experts.*.
        for mtp_layer in t.mtp:
            _ = mtp_layer(h, 0, input_ids)
        logits = t.head(h, t.hc_head_fn, t.hc_head_scale, t.hc_head_base, t.norm)
        return _LogitsOut(logits)


class _LogitsOut:
    __slots__ = ("logits",)

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


# ----------------------------- dataset -----------------------------

# V4 has no Jinja chat template; manual encoding per
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"


def preprocess_v4(example: dict) -> dict:
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


def build_calibration_dataset(
    tokenizer, *, num_samples: int, max_seq_len: int, seed: int = 42
):
    """Load + tokenize HuggingFaceH4/ultrachat_200k with the V4 manual encoding."""
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{num_samples * 2}]",  # over-fetch; filter after preprocess
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
    return ds


# ----------------------------- recipe -----------------------------


def build_recipe():
    """GPTQ recipe: FP8_BLOCK attention (incl. MTP e_proj/h_proj) + W4A16 experts."""
    from compressed_tensors.quantization.quant_scheme import (
        FP8_BLOCK,
        W4A16,
        QuantizationScheme,
    )
    from llmcompressor.modifiers.quantization import GPTQModifier

    return GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
                    r"re:mtp\.\d+\.(e_proj|h_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "experts": QuantizationScheme(
                targets=[
                    r"re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$",
                ],
                **W4A16,
            ),
        },
        ignore=[
            "head",
            "embed",
            r"re:.*norm.*",
            r"re:.*\.ffn\.gate$",
            r"re:.*\.ffn\.gate\..*",
            r"re:.*\.ffn\.shared_experts\..*",
            r"re:.*\.hc_.*",
            r"re:hc_.*",
            r"re:.*\.attn\.attn_sink",
            r"re:.*\.attn\.(compressor|indexer)\..*",
        ],
        offload_hessians=True,
        dampening_frac=0.1,
    )


# ----------------------------- main -----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Phase-1 BF16 dir")
    ap.add_argument("--output", required=True, help="output W4A16-FP8 dir")
    ap.add_argument("--config", required=True, help="vendor/dsv4-upstream/config.json")
    ap.add_argument("--samples", type=int, default=768,
                    help="calibration samples (use 4 for smoke test)")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="overrides --samples=4 --batch-size=1; aborts after one oneshot batch")
    args = ap.parse_args()

    if args.smoke:
        args.samples = 4
        args.batch_size = 1
        print("[smoke] samples=4 batch_size=1")

    # ---- model load ----
    print(f"[args] config={args.config}")
    margs = build_model_args(
        args.config,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    print(f"[args] dim={margs.dim} n_layers={margs.n_layers} "
          f"n_routed_experts={margs.n_routed_experts} n_mtp_layers={margs.n_mtp_layers}")

    print("[load] instantiating Transformer on CPU")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")
    transformer = Transformer(margs)

    print(f"[load] copying BF16 safetensors from {args.input}")
    loaded, unmatched, missing = load_safetensors_into(
        transformer, Path(args.input), verbose=True
    )
    print(f"[load] loaded={loaded} unmatched={len(unmatched)} missing={len(missing)}")
    if unmatched:
        print(f"[load] FATAL: {len(unmatched)} safetensors keys did not map to model params")
        for k in unmatched[:10]:
            print(f"  - {k}")
        sys.exit(2)

    # ---- tokenizer ----
    from transformers import AutoTokenizer

    print(f"[tokenizer] loading from {args.input}")
    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=False)
    print(f"[tokenizer] vocab_size={tokenizer.vocab_size}")

    # ---- dataset ----
    print(f"[dataset] preparing {args.samples} calibration samples")
    ds = build_calibration_dataset(
        tokenizer, num_samples=args.samples, max_seq_len=args.max_seq_len
    )
    print(f"[dataset] ready: {len(ds)} samples")

    # ---- wrap for oneshot ----
    model = CalibrationModel(transformer)

    # ---- recipe + oneshot ----
    print("[recipe] building GPTQModifier (FP8_BLOCK attn + W4A16 experts)")
    recipe = build_recipe()

    from llmcompressor import oneshot

    print("[oneshot] starting sequential GPTQ calibration over Block targets")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
        sequential_targets=["Block", "MTPBlock"],
        batch_size=args.batch_size,
    )

    # ---- save ----
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[save] writing quantized model to {out}")

    # llmcompressor save_pretrained / save_compressed expects a PreTrainedModel.
    # Our wrapper isn't one; save raw state_dict shards via safetensors.
    from safetensors.torch import save_file
    state = model.transformer.state_dict()
    # Naive sharding: 5 GB per shard
    shards: dict[str, dict[str, torch.Tensor]] = {}
    cur_bytes = 0
    cur_idx = 1
    for name, tensor in state.items():
        if cur_bytes > 5 * (1 << 30):
            cur_idx += 1
            cur_bytes = 0
        sname = f"model-{cur_idx:05d}-of-?????.safetensors"
        shards.setdefault(sname, {})[name] = tensor
        cur_bytes += tensor.numel() * tensor.element_size()
    n = cur_idx
    final = {}
    for s, payload in shards.items():
        new_name = s.replace("?????", f"{n:05d}")
        save_file(payload, str(out / new_name))
        final[new_name] = list(payload.keys())
    print(f"[save] wrote {n} shards to {out}")
    print("CALIBRATION_DONE")


if __name__ == "__main__":
    main()
