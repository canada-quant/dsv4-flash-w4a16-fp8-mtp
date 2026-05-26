# dsv4-flash-w4a16-fp8-mtp

Reproduction repo for [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) — W4A16 INT4 routed experts + FP8 block 128×128 attention + **BF16 Multi-Token Prediction (MTP) draft head retained** on DeepSeek-V4-Flash. First V4-Flash quant with working speculative decoding — 1.49× decode speedup at bs=1, k=1.

Full model card with TL;DR, benchmarks, throughput, cost-per-token, and honest limitations lives on the [HF page](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP); this README is the operator/reproduction tour.

## Family / related repos

| Repo | HF model card | Role |
|---|---|---|
| **this repo** (`dsv4-flash-w4a16-fp8-mtp`) | [W4A16-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP) | W4A16 + FP8 + BF16 MTP retained; 1.49× spec-decode at bs=1 |
| [`canada-quant/dsv4-flash-w4a16-fp8`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8) | [W4A16-FP8](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) | predecessor — same recipe without MTP (broadest hardware compatibility) |
| [`canada-quant/dsv4-flash-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp) | [NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) | sibling — NVFP4 routed experts (Blackwell-native) + MTP. **Use this sibling for batched thinking-mode on SM 12.0 / RTX PRO 6000** — this W4A16 repo's Marlin MoE decode path has a kernel race under concurrent thinking-mode load (see [`docs/findings/sm12x_token_corruption_2026_05_24.md`](docs/findings/sm12x_token_corruption_2026_05_24.md) and [`jasl/vllm#12`](https://github.com/jasl/vllm/issues/12)). |
| [`canada-quant/dsv4-pro-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-pro-nvfp4-fp8-mtp) | [Pro NVFP4-FP8-MTP](https://huggingface.co/canada-quant/DeepSeek-V4-Pro-NVFP4-FP8-MTP) | larger sibling — V4-Pro NVFP4 + MTP, B300-only |

## Quickstart

### H200 calibration + serve (8× H200 / `p5en.48xlarge`)

```bash
# Phase 0 — bootstrap (venv-calib + venv-serve + vendor + apply patches)
bash scripts/bootstrap_p5en_h200.sh

# Phase 1 — download upstream + dequant to BF16-MTP source (~660 GB, ~30 min)
bash scripts/phase1_dequant.sh

# Phase 2 — GPTQ calibration (8 ranks, ~15h wall)
bash scripts/run_phase2.sh

# Phase 3 — postprocess (rename + config patch + FP32 restore + MTP aliases)
bash scripts/postprocess_phase2.sh

# Phase 4 — verify
python scripts/verify_option_y.py /scratch/weights/w4a16-fp8-mtp-gptq

# Phase 5 — serve
vllm serve /scratch/weights/w4a16-fp8-mtp-gptq \
    --tensor-parallel-size 2 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --trust-remote-code
```

### RTX PRO 6000 Blackwell deployment (Brev `g7e.24xlarge` or equivalent)

```bash
# 1) Bootstrap vLLM source build (~25 min)
sudo apt-get update && sudo apt-get install -y git
sudo ln -sfn /opt/dlami/nvme /scratch && sudo chown -h "$USER:$USER" /scratch
cd /scratch
git clone https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp.git
cd dsv4-flash-w4a16-fp8-mtp
bash scripts/bootstrap_rtx6000pro.sh

# 2) Extra pins
source ~/venv-serve/bin/activate
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
pip install --quiet setuptools-rust
pip install --quiet git+https://github.com/inclusionAI/humming.git
pip install --quiet "flashinfer-python==0.6.8.post1" "flashinfer-cubin==0.6.8.post1" \
    "numba==0.65.0" "tilelang==0.1.9" "apache-tvm-ffi==0.1.9" "fastsafetensors>=0.2.2"

# 3) Apply patches
python scripts/patch_v4_forcausal_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_mtp_packed_mapping.py        "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_nvidia_attn_scale.py         "$(python -c 'import vllm; print(vllm.__path__[0])')"
bash   scripts/patch_wo_a_bf16_path.sh             "$(python -c 'import vllm; print(vllm.__path__[0])')"

# 4) Download artifact (159 GiB) — already dequant'd in-artifact as of 2026-05-24,
#    no local preprocessing step required
pip install --user --quiet huggingface_hub hf-transfer
export PATH="$HOME/.local/bin:$PATH"
hf download canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP \
    --local-dir /scratch/weights/w4a16-fp8-mtp-gptq

# 5) Serve TP=2 (or TP=4 with 0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/w4a16-fp8-mtp-gptq 8000 2

# 6) Smoke + bench
bash scripts/chat_smoke.sh http://localhost:8000
bash scripts/bench_rtx6000pro_suite.sh http://localhost:8000 2 1
```

Full RTX PRO 6000 recipe (patch rationale, debug notes): [`RECIPE_RTX6000PRO.md`](RECIPE_RTX6000PRO.md).

## Headline validation

| Hardware | TP | bs=1 output tok/s | bs=1 TPOT | bs=16 output tok/s | MTP acceptance @ bs=1 |
|---|---|---|---|---|---|
| 8× H200 | 2 | 88.35 | **6.02 ms** | 367.13 | 89% calibrated / 70% random |
| 4× RTX PRO 6000 box | TP=2 (per replica, 2 replicas fit) | **98.83** | 8.55 ms | 482.61 | 71% |
| 4× RTX PRO 6000 box | TP=4 (single replica) | **107.32** | 7.77 ms | **584.04** | 68% |

Quality (same artifact, all hardware): GSM8K 93.71% (8-shot strict), MMLU 86.88%, HumanEval pass@1 84.76%, AIME 2024 30.0% (thinking=high). Spec-decode speedup: **1.49× at bs=1, k=1** (TPOT 6.02 ms vs 8.93 ms, same artifact w/ and w/o spec). Full numbers + methodology footnotes on the [HF model card](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP).

## What's in this repo

| Path | What |
|---|---|
| [`scripts/bootstrap_p5en_h200.sh`](scripts/bootstrap_p5en_h200.sh) | H200 calibration environment (venv-calib + venv-serve + vendor + apply patches) |
| [`scripts/bootstrap_rtx6000pro.sh`](scripts/bootstrap_rtx6000pro.sh) | RTX PRO 6000 vLLM source build (~25 min) |
| [`scripts/phase1_dequant.sh`](scripts/phase1_dequant.sh) | Download upstream + dequant to BF16-MTP source |
| [`scripts/quantize_v4_w4a16_mtp.py`](scripts/quantize_v4_w4a16_mtp.py) | GPTQ calibration entry point (8 ranks) |
| [`scripts/postprocess_phase2.sh`](scripts/postprocess_phase2.sh) | rename + config patch + FP32 restore + MTP head/embed aliases |
| [`scripts/fixup_artifact.py`](scripts/fixup_artifact.py) | FP32 restore (workaround for `transformers.save_pretrained` silent downcast) + MTP alias injection |
| [`scripts/verify_option_y.py`](scripts/verify_option_y.py) | Verify MTP block present and unquantized in saved artifact |
| [`scripts/dequant_compressor.py`](scripts/dequant_compressor.py) | Historical one-time compressor/indexer dequant. As of 2026-05-24 the dequant'd weights are baked into the published HF artifact, so this script is no longer needed for a fresh deploy. Kept for re-quant builds. |
| [`scripts/serve_rtx6000pro.sh`](scripts/serve_rtx6000pro.sh) | RTX PRO 6000 serve helper with all required env vars |
| [`scripts/patch_*.{py,sh}`](scripts/) | vLLM in-place patches for `packed_modules_mapping`, attn scale, BF16 `wo_a` |
| [`benchmarks/rtx6000pro/`](benchmarks/rtx6000pro/) | Raw `vllm bench serve` JSONs + summary |
| [`FINDINGS_FOR_SIBLING.md`](FINDINGS_FOR_SIBLING.md) | Upstream contributions filed during this work |
| [`SUPERVISION_RULES.md`](SUPERVISION_RULES.md) | Discipline notes (atomic safetensors writes, verify-before-acting on summaries, subagent briefing standards) |
| [`RECIPE_RTX6000PRO.md`](RECIPE_RTX6000PRO.md) | Full RTX PRO 6000 recipe + patch rationale |

## Upstream contributions filed during this work

| Contribution | Description | Status |
|---|---|---|
| `transformers.save_pretrained` silent FP32 → BF16 downcast | 417 tensors specified as FP32 (HC plumbing, gate bias, attn_sink, indexer/compressor `ape`) silently written as BF16 when `torch_dtype` is BF16. Workaround in [`scripts/fixup_artifact.py`](scripts/fixup_artifact.py). | local; upstream filing pending |
| vLLM MTP loader silently skips top-level `head.weight` + `embed.weight` | `DeepSeekV4MTP.load_weights` no-ops on non-`mtp.0.*` keys → uninitialized `shared_head.head` / `embed_tokens` → 0% MTP acceptance with no load-time error. Workaround: postprocess injects `mtp.0.head.weight` and `mtp.0.emb.tok_emb.weight`. | local; upstream filing pending |
| DeepGemm `paged_mqa_logits` asserts on `num_speculative_tokens > 1` | `smxx_fp8_fp4_paged_mqa_logits.hpp:233` enforces `next_n == 1 or next_n == 2`. With `next_n = k+1`, practical k cap is 1. | upstream (DeepGemm) — filing pending |
| [`vllm-project/vllm#43248`](https://github.com/vllm-project/vllm/pull/43248) | `bool()` wrap on `is_static_input_scheme` | open |
| [`vllm-project/vllm#43288`](https://github.com/vllm-project/vllm/pull/43288) | `scale_fmt` defensive `.get()` + BF16 `getattr` wrap | open |
| [`vllm-project/vllm#43290`](https://github.com/vllm-project/vllm/pull/43290) | `weight_scale_inv`-or-`weight_scale` fallback | open |
| [`vllm-project/vllm#43319`](https://github.com/vllm-project/vllm/pull/43319) | MTP-quant-detect from safetensors header + BF16 `wo_a` fallback path | open |
| [`vllm-project/vllm#40923`](https://github.com/vllm-project/vllm/pull/40923) (tonyliu312) | Marlin MoE: include SM 12.x in default arch list — eliminates JIT-PTX corruption on Blackwell consumer/server SKUs | open, member-approved, blocked on core-maintainer SM120 policy review; canada-quant evidence [posted](https://github.com/vllm-project/vllm/pull/40923#issuecomment-4530927937) 2026-05-25 |
| [`jasl/vllm#12`](https://github.com/jasl/vllm/issues/12) | Token-stream corruption under concurrent thinking-mode on SM 12.0 W4A16 Marlin MoE | open, [updated](https://github.com/jasl/vllm/issues/12#issuecomment-4530929146) 2026-05-25 with PR #40923 result (corruption gone, but second kernel race surfaces as crash) |

## vLLM patch series — [`vllm-patches/`](vllm-patches/)

The minimum patch series we apply on top of `jasl/vllm@a02a3778f` to serve this artifact on RTX PRO 6000 (SM 12.0). Each patch is documented with its upstream PR link, status, and rationale in [`vllm-patches/README.md`](vllm-patches/README.md):

| Patch | Purpose | Upstream |
|---|---|---|
| `0001_marlin_moe_archs_40923.patch` | Build native sm_120a Marlin MoE cubins (eliminates JIT-PTX corruption on Blackwell) | [PR #40923](https://github.com/vllm-project/vllm/pull/40923) (open) |
| `0002_marlin_moe_workspace_4x.patch` | Oversize Marlin MoE lock-array workspace 4× (defensive) | (to file as follow-up to #40923) |
| `0003_marlin_moe_c_tmp_36889.patch` | Drop `min()` clamp on `c_tmp` FP32 reduce buffer (block-decode safety) | [PR #36889](https://github.com/vllm-project/vllm/pull/36889) (closed, re-file candidate) |

Patches 0002 + 0003 were built and verified to compile cleanly against `jasl/vllm@a02a3778f + #40923` (new `_moe_C.abi3.so` = 181,697,240 bytes, +1,784 vs unpatched). Functional verification on this artifact is currently blocked by a separate safetensors naming issue documented below.

## Investigation findings (RTX PRO 6000 SM 12.0)

The W4A16-MTP path on this artifact has known issues on RTX PRO 6000 + current vLLM that the team has been investigating, all documented in [`docs/findings/`](docs/findings/):

1. **Compressor/indexer FP8 shipping bug** ([`session_summary_2026_05_24.md`](docs/findings/session_summary_2026_05_24.md)) — fixed 2026-05-24 by dequantizing those weights in-artifact to BF16. Artifact now loads cleanly on modern vLLM via the older `jasl/vllm@c79225692` build that was installed at the time.

2. **Marlin MoE concurrent-decode kernel race** ([`sm12x_token_corruption_2026_05_24.md`](docs/findings/sm12x_token_corruption_2026_05_24.md)) under thinking-mode at c≥2. Partial fix from upstream [`vllm-project/vllm#40923`](https://github.com/vllm-project/vllm/pull/40923) eliminates the JIT-PTX-fallback corruption (14/30 → 0/30 on AIME c=4), but a second race in the W4A16 Marlin MoE decode path on SM 12.0 surfaces as `CUDA illegal memory access` under sustained concurrent thinking-mode load. **Workaround**: use the NVFP4 sibling [`canada-quant/dsv4-flash-nvfp4-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp) for batched thinking-mode on SM 12.0. This W4A16-MTP artifact works cleanly for sequential thinking-mode (c=1) and any batched chat-mode (no thinking) workload.

3. **Marlin c_tmp + workspace 4× patches built** ([`cardd_marlin_patches_built_artifact_blocker_2026_05_25.md`](docs/findings/cardd_marlin_patches_built_artifact_blocker_2026_05_25.md)) — patches 0002 + 0003 in `vllm-patches/` were cherry-picked (PR #36889 `c_tmp` clamp removal + workspace 4× oversize). They compile cleanly, produce a new `_moe_C.abi3.so` (+1,784 bytes). Functional verification is blocked by item (4) below.

4. **Safetensors `.weight_scale` naming blocker** ([`cardd_artifact_weight_scale_naming_blocker_2026_05_25.md`](docs/findings/cardd_artifact_weight_scale_naming_blocker_2026_05_25.md)) — all 33,239 quantized scale tensors in this artifact's safetensors are named `<module>.weight_scale` rather than the canonical `<module>.weight_scale_inv`. Mathematical content is identical (the `llmcompressor.model_free_ptq` path emits this naming), but current upstream vLLM's loader is strict and rejects the artifact. Filed as [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564) with a proposed two-line defensive `getattr` patch + tagged `@kylesayrs`. **The fix can land via either**: (a) a vLLM PR that accepts both naming conventions, or (b) renaming `.weight_scale` → `.weight_scale_inv` in this artifact's safetensors header (a metadata-only rewrite — see [`scripts/rename_weight_scale_to_inv.py`](scripts/rename_weight_scale_to_inv.py); no re-quantization).

## License

MIT, inherited from upstream `deepseek-ai/DeepSeek-V4-Flash`.

## Acknowledgments

- DeepSeek for the base model + MTP architecture + inference reference.
- jasl ([`jasl/vllm`](https://github.com/jasl/vllm) and [`jasl/vllm-ds4-sm120-harness`](https://github.com/jasl/vllm-ds4-sm120-harness)) for the vLLM build pins and benchmark harness.
- [`canada-quant/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8) (predecessor) for the recipe topology this artifact extends with MTP.
- [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) (sibling) for the alias-injection pattern and MTP acceptance methodology.
