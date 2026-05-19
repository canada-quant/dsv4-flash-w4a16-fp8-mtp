# dsv4-flash-w4a16-fp8-mtp

Re-quantization of [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) to W4A16-FP8 with the **MTP (multi-token-prediction) layer correctly included**, targeting AWS `p6-b300.48xlarge` (8× B300, Blackwell DC SM 10.0).

The predecessor quant at [`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/pastapaul/DeepSeek-V4-Flash-W4A16-FP8) shipped without the MTP block — `transformers` 5.8.1's `DeepseekV4PreTrainedModel._keys_to_ignore_on_load_unexpected` silently drops every `mtp.*` key on `from_pretrained`. This repo isolates and fixes that.

Sibling, non-overlapping scope to [`dsv4-flash-reasoning-agent`](https://github.com/pasta-paul/dsv4-flash-reasoning-agent) (which adds SFT + GRPO on top of the quant). This repo is **quant-only**.

## Status

| Phase | What | Outcome | Wall clock |
|---|---|---|---|
| 0 | Instance bring-up, `/data` mount, `/scratch` symlink, `venv-calib` with patches | ✓ done | ~1 h |
| 1 | Dequant FP4/FP8 → BF16 preserving MTP | ✓ **543 GB BF16, 797 MTP tensors** at `/scratch/weights/bf16-mtp` | 8m 33s |
| 2 (RTN fallback, **SUPERSEDED**) | model_free RTN W4A16-FP8 + MTP — kept as a fallback at `/scratch/weights/w4a16-fp8-mtp-rtn-fallback`, **not shipping** | ⚠ superseded — RTN does not match predecessor GPTQ quality | 7 min |
| 3 (config cleanup for RTN, **SUPERSEDED**) | clean_ignore_list.py removed pass-1↔pass-2 overlap on the RTN config | ⚠ superseded with Phase 2 | <1 s |
| **2 (real)** | **GPTQ calibration of layers 0-42 + mtp.0 in one oneshot pass** — vendored upstream `Transformer` patched per `PHASE2_DESIGN.md` option A', forward wrapped to flow through MTP, HuggingFaceH4/ultrachat_200k 768 samples (predecessor's corpus) | ⏳ scaffolded; delta in `PHASE2_GPTQ_DELTA.md` — **gated on user approval before code lands** | est 8-12 h calibration + 1 day adapter work |
| **3 (real)** | post-calibration config + GPTQ-signature verification (per-expert scale spread, `actorder` present) | ⏳ waiting on Phase 2 | <5 min |
| 4 | Install CUDA toolkit, build vLLM, apply 2 patches | ✓ done — vllm-0.1.dev1+g3424fba51 installed, both `packed_modules_mapping` attributes verified | ~1 h |
| 5 | Smoke serve TP=2 with `--speculative-config method=mtp num_speculative_tokens=2` on the **GPTQ** artifact | ⏳ waiting on Phase 2/3 | — |
| 6 | Benchmarks (chat-smoke, toolcall15, GSM8K, HumanEval, NIAH, MTP-acceptance) | ⏳ next | ~4 h |
| 7 | 4× TP=2 instances pinned to GPU pairs | ⏳ next | ~2 h |
| 8 | HF release as `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` | ⏳ permission-gated | ~1 h |

See [`PLAN.md`](PLAN.md) for the full per-phase plan and [`patches/VERSIONS.md`](patches/VERSIONS.md) for patch provenance.

## Key findings (root causes documented in memory)

1. **transformers 5.8.1 silently drops `mtp.*` keys.** The `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` regex on `DeepseekV4PreTrainedModel` is the actual mechanism. Patched in `patches/modeling_deepseek_v4.py.diff` (hunk 1).
2. **Upstream uses DeepSeek-internal naming throughout** (`layers.X.attn.wq_a` not `model.layers.X.self_attn.q_a_proj`, scale suffix `.scale` not `.weight_scale_inv`). The predecessor's regexes would silently match zero modules.
3. **`llmcompressor.entrypoints.model_free.model_free_ptq`** operates directly on safetensors with no `PreTrainedModel` required. Bypassed the MTP-class integration block that would otherwise need a 500+ LOC adapter. Trade-off is RTN instead of GPTQ — fine for FP8 (essentially lossless) and acceptable for MTP draft layer (forgiving metric).
4. **DLAMI gotchas:** (a) `/opt/pytorch`'s Python 3.13 venv with `--system-site-packages` pulls 3.12-only wheels from `/usr/lib/python3/dist-packages` causing pyo3 panics — don't use `--system-site-packages`; (b) the bundled CUDA at `/opt/pytorch/cuda` lacks `lib64/` symlink and unversioned `.so` files — needs `apt install cuda-toolkit-13-0` for source builds.

## Recipe

Same topology as the predecessor quant — FP8_BLOCK 128×128 attention + W4A16 INT4 g=128 sym routed experts — extended to **also cover the MTP block** (layer 43, named `mtp.0.*` in upstream):

- `re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$` → **W4A16** (matches main 43 layers + mtp.0)
- `re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$` → **FP8_BLOCK**
- `re:mtp\.\d+\.(e_proj|h_proj)$` → **FP8_BLOCK** (MTP-specific entry projections)
- everything else (norms, gates, shared experts, hc_*, attn_sink, compressor/indexer aux) → **BF16 passthrough**

## Target output

- HF model: `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` (Phase 8, permission-gated)
- Decode target: **>85.52 tok/s @ 524K** on 8× B300 with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
- Eval bar: GSM8K ≥94.5%, HumanEval pass@1 ≥77%, toolcall15 ≥26/30, NIAH 5/5 @ 524K

## Reproduce

```bash
# Phase 0 — bootstrap (on a fresh p6-b300.48xlarge with ami-02e9fc7da15a197f9)
sudo apt-get install -y cuda-toolkit-13-0     # see memory:dlami_cuda_toolkit_incomplete
bash scripts/bootstrap_p6_b300.sh             # mounts /data, sets up venv-calib + venv-serve

# Phase 1 — dequant
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir /data/weights/upstream
python scripts/dequant_mtp.py --input /data/weights/upstream --output /scratch/weights/bf16-mtp
python scripts/verify_mtp_keys.py /scratch/weights/bf16-mtp

# Phase 2 — model_free RTN quantization
python scripts/quantize_v4_model_free.py \
    --input /scratch/weights/bf16-mtp \
    --output /scratch/weights/w4a16-fp8-mtp \
    --device cuda:0

# Phase 3 — clean ignore list
python scripts/clean_ignore_list.py --config /scratch/weights/w4a16-fp8-mtp/config.json

# Phase 4 — vLLM patches
source /data/venv-serve/bin/activate
python scripts/patch_v4_forcausal_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_mtp_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"

# Phase 5 — serve
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_b300_tp2.sh /scratch/weights/w4a16-fp8-mtp 8000
```

## Repo layout

```
.
├── PLAN.md                 # full phase-by-phase execution plan
├── PHASE2_DESIGN.md        # MTP-shim design for the GPTQ refinement path
├── CLAUDE.md               # session notes for Claude Code agents
├── README.md               # this file
├── patches/
│   ├── modeling_deepseek_v4.py.diff   # transformers 5.8.1: empty mtp-ignore + cache fix
│   ├── helpers.py.diff                # llm-compressor: Cache tracer
│   └── VERSIONS.md
├── scripts/
│   ├── bootstrap_p6_b300.sh           # Phase 0
│   ├── dequant_mtp.py                 # Phase 1
│   ├── verify_mtp_keys.py
│   ├── quantize_v4_model_free.py      # Phase 2 (RTN, ships)
│   ├── verify_mtp_quantized.py
│   ├── clean_ignore_list.py           # Phase 3
│   ├── patch_v4_forcausal_packed_mapping.py   # Phase 4 vLLM patches
│   ├── patch_mtp_packed_mapping.py
│   ├── serve_b300_tp2.sh              # Phase 5
│   ├── upstream/                      # Phase 2 GPTQ refinement scaffold
│   │   ├── __init__.py                # vendor model.py adapter
│   │   └── kernel_shim.py             # PyTorch refs for tilelang kernels
│   ├── load_bf16_into_transformer.py  # GPTQ-path loader
│   ├── smoke_test_adapter.py          # adapter smoke test (passes)
│   └── quantize_v4_w4a16_mtp.py       # GPTQ entry (scaffold; needs PreTrainedModel shim)
└── vendor/
    └── dsv4-upstream/                 # verbatim from deepseek-ai/DeepSeek-V4-Flash/inference/
```

## License

Apache-2.0, inherited from the base model (which is MIT). Each vendored file under `vendor/` retains its original upstream license.
