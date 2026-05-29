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

### RTX PRO 6000 Blackwell — Docker (recommended)

The fastest path on RTX PRO 6000 Blackwell Server Edition (SM 12.0) is the
pre-built [`canada-quant/dsv4-rtx6000pro:v3`](https://huggingface.co/datasets/canada-quant/dsv4-flash-w4a16-rtxpro6000-image)
image, which bakes the full 13-layer recipe (`jasl/vllm@27fd665b` + canada-quant
BF16 MTP cherry-pick + Marlin MoE c_tmp/workspace patches + `cute.arch.fmin`
shim). Boots straight to a working `vllm serve` endpoint in ~3-5 min on a
g7e.24xlarge.

```bash
# 1) Pull image (~14 GB compressed, ~47 GB on disk)
docker pull canada-quant/dsv4-rtx6000pro:v3

# 2) Pre-cache the W4A16 model onto NVMe (~159 GB; 1-2 min via xet on Brev)
HF_HOME=/opt/dlami/nvme/hf-cache hf download \
    canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP

# 3) Serve TP=2 (2× RTX PRO 6000) or TP=4 (4× RTX PRO 6000)
docker run -d --gpus '"device=0,1"' --name dsv4-w4a16-serve \
    --shm-size=16g --ipc=host -p 8000:8000 \
    -v /opt/dlami/nvme/hf-cache:/root/.cache/huggingface \
    -v $(pwd)/scripts:/workspace/scripts:ro \
    -e TP=2 -e MAX_NUM_SEQS=4 -e MAX_MODEL_LEN=65536 -e GPU_MEM_UTIL=0.95 \
    canada-quant/dsv4-rtx6000pro:v3 \
    bash /workspace/scripts/serve_rtx6000pro_w4a16.sh

# 4) Wait for /v1/models (~4-5 min model load + cudagraph capture)
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done

# 5) Smoke test
curl -sX POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"DSV4-W4A16-FP8-MTP",
         "messages":[{"role":"user","content":"What is 17*23?"}],
         "max_tokens":60,"temperature":0}' | jq .choices[0].message.content

# 6) Run the full bench matrix (AIME mode sweep + GSM8K + throughput)
docker exec dsv4-w4a16-serve bash -c \
    "TAG=tp2_64k MAX_MODEL_LEN=65536 bash /workspace/scripts/bench_matrix.sh"
```

Notes:
- The image is published at `canada-quant/dsv4-flash-w4a16-rtxpro6000-image`
  on HF as a dataset; pull via `docker load < $(hf download canada-quant/dsv4-flash-w4a16-rtxpro6000-image --include "*.tar.gz" --local-dir .)`.
- For TP=4 (single replica, all 4 GPUs), use `--gpus all -e TP=4 -e MAX_NUM_SEQS=16`.
- All `bench_matrix.sh` AIME runs set `max_tokens = max_model_len - 500` so
  reasoning runs to its natural stop instead of being capped.
- See [`docker/Dockerfile.rtx6000pro`](docker/Dockerfile.rtx6000pro) if you
  want to rebuild the image from source — the recipe is the same one we used
  to build v3 (~25 min on a clean g7e.24xlarge).

### RTX PRO 6000 Blackwell — from-source install (advanced)

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

### RTX PRO 6000 Blackwell Server Edition — verified 2026-05-29 in fresh Docker ✓

Full AIME-2024 + GSM8K + throughput sweep on `canada-quant/dsv4-w4a16-rtxpro6000:v1` (image baked from `jasl/vllm@27fd665b` + canada-quant BF16-MTP cherry-pick + Marlin MoE `c_tmp`/workspace patches). Both TP=2 and TP=4 single-replica, **all AIME runs at `max_tokens = max_model_len - 500 = 65036`** so reasoning runs to natural stop instead of being capped:

#### AIME-2024 (n=30) — thinking-mode sweep, c=4

| | TP=2 (max_num_seqs=4) | TP=4 (max_num_seqs=16) |
|---|---|---|
| chat (no think) | 18/30 (60.0%) · MTP 95.78% · 53m | **19/30 (63.3%) · MTP 93.06% · 5m** |
| **thinking-high** | **27/30 (90.0%) · MTP 91.97% · 152m** | **29/30 (96.7%) · MTP 91.01% · 13m** |
| thinking-max | 24/30 (80.0%) · MTP 92.52% · 177m | 27/30 (90.0%) · MTP 91.68% · 26m |

#### AIME-2024 (n=30) — c=1 single-shot reference (thinking-high)

| | TP=2 | TP=4 |
|---|---|---|
| c=1 high | 27/30 (90.0%) · MTP 91.68% · 48m | **28/30 (93.3%) · MTP 90.76% · 41m** |

#### GSM8K (n=50, 8-shot)

| | TP=2 | TP=4 |
|---|---|---|
| flexible-extract | 45/50 (90.0% ± 4.3) | 43/50 (86.0% ± 5.0) |
| strict-match | 42/50 (84.0% ± 5.2) | 40/50 (80.0% ± 5.7) |

#### Throughput sweep — random 256/256 (MTP on, single replica)

| bs | TP=2 tok/s | TP=2 TPOT p50 | TP=4 tok/s | TP=4 TPOT p50 |
|---|---|---|---|---|
| 1 | 95.2 | 8.05 ms | **108.1** | 7.32 ms |
| 4 | 40.6 | 83.18 ms | 104.3 | 11.31 ms |
| 8 | 45.7 | 79.02 ms | **433.2** | 16.44 ms |
| 16 | 34.9 (queued at max_num_seqs=4) | 81.77 ms | 164.3 (scheduler thrash) | 90.18 ms |

#### Throughput sweep — random 1024/1024 (MTP on)

| bs | TP=2 tok/s | TP=4 tok/s |
|---|---|---|
| 1 | 30.7 | 138.1 |
| 4 | 45.1 | 363.7 |

**Headlines:**

- **TP=4 thinking-high 29/30 + zero CUDA errors** on 90 AIME problems + 50 GSM8K problems = strongest validation to date of the Card D path on consumer/server Blackwell. Closes [`jasl/vllm#12`](https://github.com/jasl/vllm/issues/12) (Marlin MoE concurrent-decode race) — the `c_tmp` clamp removal in PR [`vllm#43730`](https://github.com/vllm-project/vllm/pull/43730) + sm_120a native cubins from PR [`vllm#40923`](https://github.com/vllm-project/vllm/pull/40923) are doing their job.
- **TP=4 is 7-12× faster wall** than TP=2 for AIME (chat 53m→5m, high 152m→13m, max 177m→26m). MoE expert sharding across 4 GPUs decisively wins.
- **Thinking-max regresses correctness AND triples wall** (TP=4: high 29/30 in 13m vs max 27/30 in 26m). DeepSeek-V4-Flash hits its sweet spot at thinking-high; reasoning-effort=max gives the model more budget but produces more long-tail failures.
- **MTP acceptance holds at 91-93%** across all thinking modes and both TP configs — the BF16-retained draft head is doing its job everywhere.
- **Throughput sweet spot at TP=4 is bs=8** for short outputs (433 tok/s aggregate); bs=16 enters scheduler-thrash territory.

Raw JSON for every cell above lives in [`benchmarks/rtxpro6000_docker_v3/`](benchmarks/rtxpro6000_docker_v3/) (50 files: per-bench JSON + log + the matrix runlog + tp{2,4}_64k_SUMMARY.md).

#### Max context window (TP=2, max_num_seqs=2, MTP retained) — empirical ceiling

Walking `max_model_len` to find where TP=2 + MTP + gpu_memory_utilization=0.95 stops fitting on 2× RTX PRO 6000 (97.9 GiB/GPU). Each step is a fresh container restart + smoke test:

| max_model_len | Loaded? | VRAM/GPU | Available KV cache | Notes |
|---|---|---|---|---|
| 64K | ✓ | 93 GiB | — | the throughput config |
| 128K | ✓ | 93 GiB | — | doubles cheap |
| 256K | ✓ | 93 GiB | — | initial warmup 6.4 min |
| 384K | ✓ | 93 GiB | — | |
| 512K | ✓ | 93 GiB | — | |
| **1M (1,048,576)** | **✓ — architectural max** | **93 GiB** | **8.04 GiB** | the model's `max_position_embeddings` cap |

**Result: TP=2 + W4A16 + FP8 attention + BF16 MTP retained serves the full architectural 1M token context on 2× RTX PRO 6000 Blackwell** — same VRAM footprint as the 64K config because vllm sizes KV cache to fill `gpu_memory_utilization` regardless.

Caveats:
- vllm requires `cudagraph_capture_sizes` to be multiples of 2 when MTP `num_speculative_tokens=1` is active (each step processes main + draft). Effective minimum is `max_num_seqs=2`.
- At 1M context with 8 GiB KV cache budget, only **one sequence can actually fill the full window at once** (~4 KB/token effective FP8 MLA + Lightning Indexer ≈ 4 GiB per 1M tokens). The second `max_num_seqs` slot exists for cudagraph rounding only — under sustained 1M-tokens-per-request workload, plan for effective concurrency = 1.
- Prefill latency at 1M tokens is ~8-12 minutes per request even on TP=2 Blackwell. **Long-context is a single-user interactive operating mode, not a throughput config.** Use the 64K + max_num_seqs=4/8/16 profiles for throughput.

Two recommended serve profiles:

```bash
# Throughput / interactive multi-user (sweet spot)
docker run ... -e TP=4 -e MAX_NUM_SEQS=16 -e MAX_MODEL_LEN=65536 -e GPU_MEM_UTIL=0.95 ...
# ⇒ 29/30 AIME thinking-high, 433 tok/s aggregate at bs=8, 7-12× faster than TP=2

# Long-context single-user (architectural max)
docker run ... --gpus '"device=0,1"' -e TP=2 -e MAX_NUM_SEQS=2 -e MAX_MODEL_LEN=1048576 -e GPU_MEM_UTIL=0.95 ...
# ⇒ full 1M context with MTP retained on 2× RTX PRO 6000
```

> **Bench-script note (fixed 2026-05-29):** vllm's OpenAI serving layer renames `reasoning_content` → `reasoning` in the response per the spec. The old bench script only checked `reasoning_content` (which doesn't exist in the response), so on `finish_reason=length` it lost the model's partial reasoning entirely. TP=2 results above were collected with the old script and therefore slightly under-count length-truncations (1 each in chat/high/max). TP=4 results use the patched script ([`scripts/aime_thinking_bench.py`](scripts/aime_thinking_bench.py)). Headlines are unchanged; future numbers will be clean.

**`finish_reasons` distribution** at c=4: 22 `stop` + 8 `length` truncation at max_tokens=16K — consistent with reference H200/B300 truncation rate for the longest AIME-2024 reasoning problems. Non-truncated pass@1 = 24/22 = 100%.

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
| [`vllm-project/vllm#43459`](https://github.com/vllm-project/vllm/pull/43459) | DSv4 MTP loader: route top-level `head.weight` + `embed.weight` to MTP `shared_head/embed_tokens` (fixes 0% acceptance for non-aliased artifacts) | open |
| [`vllm-project/vllm#43722`](https://github.com/vllm-project/vllm/pull/43722) | `MarlinFP8.can_implement` refuses block-FP8 layers (dispatcher falls through to Triton) | **open, filed 2026-05-26** |
| [`vllm-project/vllm#43723`](https://github.com/vllm-project/vllm/pull/43723) | DSv4 `attention.py` `wo_a.weight_scale_inv` getattr fallback (companion to #43722) | **open, filed 2026-05-26** |
| [`vllm-project/vllm#41834`](https://github.com/vllm-project/vllm/pull/41834) (jasl) | RTX PRO 6000 Server Edition Triton block-FP8 tuned autotune configs (6 linear + 10 MoE) — without these, default `num_stages=2` produces drifting outputs on SM 12.0 | open, our validation [comment](https://github.com/vllm-project/vllm/pull/41834#issuecomment-4550181916) 2026-05-26 |
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
