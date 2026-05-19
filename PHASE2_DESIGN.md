# Phase 2 design — MTP-inclusive GPTQ calibration

**Status:** scaffolded, not implemented. Picks up after `scripts/dequant_mtp.py`
produced `/scratch/weights/bf16-mtp` (verified by `verify_mtp_keys.py`).

## The blocker, restated

transformers 5.8.1's `deepseek_v4` package has the architecture but **no MTP
module class**. With `_keys_to_ignore_on_load_unexpected` patched to `[]`,
`from_pretrained` deserializes the 1,575 `mtp.0.*` tensors but they cannot
attach to any `nn.Module` — they sit in the state dict as orphans and are
absent from `model.parameters()`. GPTQ via `llmcompressor.oneshot` traverses
`model.named_modules()` to find Linear targets, so it cannot calibrate them.

## Three approaches (chosen: option A')

| | Pros | Cons |
|---|---|---|
| **A. In-script shim** — define a faithful `DeepSeekV4MTPLayer` `nn.Module` in `quantize_v4_w4a16_mtp.py` and inject it as `model.model.mtp = shim` before oneshot | Calibration runs in a single oneshot pass like predecessor | 500+ LOC reimplementing Block + Attention + MoE + RMSNorm + hc_pre/hc_post + Compressor + Indexer; risk of subtle numerical drift vs upstream |
| **A'. Vendor upstream + adapt** — copy `inference/model.py` + `kernel.py` from the HF repo into `vendor/dsv4-upstream/`, swap their custom `Linear` for `nn.Linear`, replace kernel.py imports with reference PyTorch (`F.scaled_dot_product_attention` for `sparse_attn`, hand-rolled `hc_split_sinkhorn`, no `act_quant` since W4A16 doesn't quantize activations), instantiate `MTPBlock` from the BF16 safetensors, attach to model | Reference-correct forward; minimal new code | Need adapter shim for kernel deps; debugging subtle distributed/cache-aware code paths even with `world_size=1` |
| **B. Two-pass calibration** — pass 1 calibrates layers 0-42 only (predecessor recipe with naming fixed); capture `hidden_states_layer_42_output` from each calibration batch; pass 2 runs a standalone `MTPBlock` forward on the captured states + input_ids and triggers GPTQ on its leaf Linears | Cleanest decoupling; main-model calibration is the proven path | Still need the same MTPBlock shim for pass 2; doubles calibration wall-clock; activation distribution drift if pass-1's quantized layers aren't faithfully simulated when capturing |
| **C. Skip MTP** | Trivial; produces a working model | Misses the entire point of the repo — restores the predecessor's MTP-less state |

A' is chosen: the upstream code is 827 LOC of model.py + 536 LOC of kernel.py, and we vendor both verbatim at `vendor/dsv4-upstream/`. The adaptation surface is contained to:

1. **`Linear` swap.** Upstream's `Linear` is FP4/FP8/BF16-aware (auto-selects scale param at __init__). For calibration we need plain `nn.Linear` so llmcompressor's GPTQ hooks fire natively. Either monkey-patch `vendor.dsv4_upstream.model.Linear = nn.Linear` before instantiation, or fork the file and replace.

2. **`kernel.py` shim.** Replace the six imports
   ```python
   from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
   ```
   with a `vendor/dsv4_upstream/kernel_shim.py` that provides PyTorch reference impls:
   - `sparse_attn(q, kv, attn_sink, topk_idxs, scale)` → reduce to `F.scaled_dot_product_attention` over the topk-selected positions. Calibration only needs *some* attention output; sliding-window sparsity is implementation detail.
   - `hc_split_sinkhorn` → reference implementation using `softmax` + a few Sinkhorn iterations (~30 LOC).
   - `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm` → no-ops or BF16 fallbacks. W4A16 recipe does not quantize activations, so these never trigger their fast paths during calibration.

3. **Distributed bypass.** `world_size = 1`, `rank = 0`, no `dist.all_reduce` calls trigger.

4. **`max_batch_size` / `kv_cache` sizing.** Set tiny (e.g., max_batch_size=1, max_seq_len=512 matching `--max-seq-len` arg) so the kv_cache buffer doesn't OOM during calibration.

5. **Weight loader.** Upstream's `Transformer.load_weights` expects DeepSeek-internal naming, which matches `/scratch/weights/bf16-mtp` exactly (see `memory:dsv4_naming_convention`). Iterate the BF16 safetensors, copy each tensor into the Transformer's state dict. The Transformer's MTPBlock owns `e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`, and inherits attn/ffn/norms from Block.

## Wiring into llmcompressor.oneshot

Once the upstream `Transformer` is loaded with BF16 weights, monkey-patch its `forward` to expose the structure transformers' `PreTrainedModel` expects (just enough that `oneshot(model=...)` works). Or skip transformers entirely — `oneshot` accepts any nn.Module with a forward; the transformers integration is for tokenizer / dataset / `from_pretrained` convenience, not a hard requirement.

Recipe (names corrected for internal convention — see PLAN.md Phase 2):

```python
recipe = GPTQModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[r"re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
                     r"re:mtp\.\d+\.(e_proj|h_proj)$"],
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[r"re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$"],
            **W4A16,
        ),
    },
    ignore=[
        "lm_head", "embed_tokens",
        r"re:.*norm.*",
        r"re:.*\.ffn\.gate$",
        r"re:.*\.ffn\.shared_experts\..*",
        r"re:.*\.hc_.*",
        r"re:.*\.attn\.attn_sink",
        r"re:.*\.attn\.(compressor|indexer)\..*",
    ],
    offload_hessians=True,
    dampening_frac=0.1,
)
```

`sequential_targets=["Block"]` covers both regular and MTP blocks since MTPBlock inherits Block. Set `n_layers=43` and `n_mtp_layers=1` in ModelArgs so the Transformer instantiates 43 Blocks + 1 MTPBlock matching the upstream checkpoint.

## Steps (in order) for the next session

1. `cp vendor/dsv4-upstream/{model.py,kernel.py} scripts/upstream/` and create `kernel_shim.py` replacing the six kernel imports.
2. Patch `Linear` -> `nn.Linear` in the vendored `model.py`.
3. Write a small loader: takes `/scratch/weights/bf16-mtp` + ModelArgs (read from upstream config.json), instantiates `Transformer`, populates state_dict from the safetensors shards.
4. Sanity check: do one full forward on a tiny prompt; verify the output is finite (not NaN/Inf).
5. Replace the `main()` stub in `scripts/quantize_v4_w4a16_mtp.py` with the actual oneshot invocation using the recipe above, `sequential_targets=["Block"]`, the calibration dataset path from the predecessor.
6. Launch with `torchrun --nproc-per-node 8` (will need TP-aware tweaks in the vendored MoE — set `world_size=8`, distribute experts across ranks).
7. Run `verify_mtp_quantized.py` on the output.

## Estimated effort

3-4 focused hours of coding + GPU debugging. The numerical correctness of `hc_split_sinkhorn` and the `sparse_attn` reduction is the highest-risk piece — recommend comparing calibration activations against a known-good run before scaling to the full 768-sample sweep.
