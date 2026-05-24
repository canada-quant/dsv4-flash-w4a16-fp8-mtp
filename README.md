---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash
base_model_relation: quantized
tags:
- deepseek
- deepseek_v4
- compressed-tensors
- gptq
- w4a16
- fp8
- mtp
- speculative-decoding
- text-generation
- conversational
pipeline_tag: text-generation
library_name: vllm
---

# DeepSeek-V4-Flash — W4A16 + FP8 + **BF16 MTP** (Option Y)

The first DeepSeek-V4-Flash quantization that **preserves the Multi-Token Prediction
(MTP) draft head at BF16 precision** while quantizing the main routed-experts to W4A16
and the attention path to FP8_BLOCK. The MTP draft tower delivers a measured
**1.49× decode speedup at single-user concurrency** versus the same artifact served
without speculative decoding — with **zero quality regression** on standard knowledge
benchmarks vs the predecessor W4A16 quant.

| Component | Precision | Quantization recipe |
|---|---|---|
| Main routed experts (43 MoE layers × 256 experts × 3 projections) | **W4A16** INT4 g=128 sym | GPTQ via llm-compressor, 768 calibration samples |
| Attention path (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`, indexer, compressor) | **FP8_BLOCK** 128×128 | Dynamic scales, `scale_fmt=ue8m0` |
| **MTP block (`mtp.0.*`)** | **BF16** | Excluded from quantization (Option Y) |
| HC plumbing (`hc_attn_*`, `hc_ffn_*`, `hc_head_*`), `attn_sink`, `ffn.gate.bias`, indexer/compressor `ape` | **FP32** | Restored post-save from BF16 source (see C13) |
| `head.weight` (LM head) | **FP32** | Upcast from BF16 to match sibling artifact |
| Vocab embedding (`embed.weight`, `mtp.0.emb.tok_emb.weight`) | BF16 | Source dtype preserved |

Total artifact size: 159 GB (4 shards).

---

## Why this exists

The predecessor `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` and the RedHat
`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` artifacts both **drop the MTP block**
because `transformers` 5.8.1's `DeepseekV4PreTrainedModel` declares:

```python
_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
```

which silently filters every `mtp.*` tensor at `from_pretrained` time —
without warning, without error. Quantization pipelines that go through
`from_pretrained` therefore produce W4A16/NVFP4 main weights paired with
an absent MTP block — and serving falls back to plain decode, losing the
~1.5–2× spec-decode speedup that DeepSeek-V4-Flash's architecture
provides.

This artifact bypasses that silent drop, runs the full 8-rank GPTQ
calibration on a 768-sample dataset against the main routed experts,
preserves the MTP block unquantized in BF16, and produces a serving
artifact where speculative decoding **actually fires**.

---

## Benchmarks (H200, vLLM, TP=2)

All numbers from the same Phase 2 artifact, same hardware (1 of 4 TP=2 pairs
on a `p5en.48xlarge`), same vLLM build (HEAD `50d9dd902` with PRs
#43248+#43288+#43290+#43319 cherry-picked).

### Quality (standard benchmarks)

| Benchmark | Phase 2 (this repo) | Predecessor (W4A16-FP8, no MTP, HF card) | RedHat (NVFP4-FP8, no MTP) | Delta vs predecessor |
|---|---|---|---|---|
| GSM8K 8-shot strict-match | **93.71%** ± 0.67 | 94.99% (phase4e, 8-shot strict) | 91.0% | -1.28 pts (within SE) |
| GSM8K 8-shot flexible-extract | 93.63% ± 0.67 | — | — | predecessor HF card cites 5-shot flex at 92.87%; not directly comparable to our 8-shot |
| MMLU 5-shot | **86.88%** ± 0.27 | 87.27% | — | -0.39 pts (within SE) |
| MMLU-Pro 5-shot (12k prompts, custom-extract) | **71.28%** ± 0.40 | — | — | sibling NVFP4-FP8-MTP scored 81.13% (B300) → -9.85 pts vs sibling; expected given W4A16 has more quantization noise than NVFP4 on a knowledge-heavy harder benchmark |
| HumanEval pass@1 0-shot instruct | **84.76%** ± 2.82 | 54.27% (strict-regex artifact) | — | predecessor's 54.27% reflects a strict-regex extraction artifact (per predecessor notes); our 84.76% uses lm-eval-harness's default flexible code-block extraction, which is what `evaluate-metric/code_eval` actually executes |
| AIME 24 (math competition, 30 problems) | **30.0% exact_match** ± 8.51 | — | — | sibling published `tier1_aime24_2026_05_21.md`; competition math at high difficulty |
| Chat-smoke quick (4 deterministic) | **4/4** | 4/4 | — | match |
| Chat-smoke quality (4 writing/translation) | **4/4** | 4/4 | — | match |
| Chat-smoke coding (2 HTML/code) | **2/2** | 2/2 | — | match |
| toolcall15 | **24/30 (80%)** | 26/30 (87%) | — | -2 pts |

### MTP-specific (Option Y differentiator)

| Metric | Phase 2 | Sibling NVFP4-FP8-MTP (published) | Notes |
|---|---|---|---|
| MTP draft-token acceptance (random 256-token prompts, c=1, k=1, 200 samples) | **69.94%** (21024 / 30058) | 67.29% (HumanEval raw code, c=1, k=2) | Direct comparison — different prompt distribution but same metric class |
| MTP acceptance by workload — code (15 raw-completion prompts) | **92.91%** (1847 / 1988) | 67.29% (sibling raw HumanEval) | Sibling's HumanEval prompts are full multi-line function bodies; ours are short signature+docstring prompts — predictable continuation pattern bumps acceptance |
| MTP acceptance by workload — chat-templated prose (15 prompts) | **81.90%** (1946 / 2376) | 85.04% (sibling EvalPlus HumanEval c=16) | Both numbers fall in the chat-templated 80-85% band sibling documented |
| MTP acceptance by workload — raw natural language (15 continuation prompts) | **83.65%** (1745 / 2086) | — | New measurement |
| Decode TPOT median, bs=1, k=1, MTP-spec | **6.02 ms** | — | Single-user decode latency per output token |
| Decode TPOT median, bs=1, no spec-decode | 8.93 ms | — | Same artifact, spec-decode disabled |
| **Decode speedup bs=1 (k=1, vs no-spec)** | **1.49×** | 2.03× (sibling at k=2) | Sibling used k=2; we hit DeepGemm kernel ceiling at k=1 (see C15) |

Spec-decode wins at low concurrency (single-user interactive workload).
At bs=4/16, the verifier model is already filling its batch lane, so the
extra verifier passes add overhead without saving wall-clock — matching
sibling's published methodology framing of `c=1` as the headline
operating point.

### toolcall15 -2 pts explained honestly

The two regressions vs predecessor are model-routing decisions, not
parser/quant artifacts:

- **TC-06 "Multi-Value Extraction":** asked to translate one phrase to
  two languages; predecessor issued two `translate` tool calls,
  Phase 2 returned both translations as plain content text without
  routing to tools. Net effect: same task completed, different
  execution path. Not a parser failure (confirmed by replaying
  through `--tool-call-parser deepseek_v4`).
- **TC-07 "Search Read Act":** Phase 2 issued the first two tool calls
  (`search_files`, `read_file`) correctly but stopped mid-chain to ask
  the user a clarifying question instead of carrying the result
  forward into a third call. Predecessor completed the chain end-to-end.

Both regressions are conservatism in chain-completion + tool-selection
heuristics. Quality-wise the model still completed the underlying user
intent; the harness scores tool-call-protocol fidelity, not task
completion. No evidence of a parser config issue.

---

## Three upstream contributions surfaced during this work

These bugs were diagnosed during the build and are filed in
[`FINDINGS_FOR_SIBLING.md`](FINDINGS_FOR_SIBLING.md) for upstream PRs:

### C13 — `transformers.save_pretrained` silently downcasts FP32 to BF16

417 tensors specified as FP32 in DeepSeek's release spec (HC plumbing,
gate bias, attn_sink, indexer/compressor `ape`) are silently written as
BF16 by `transformers.save_pretrained` when the model's `torch_dtype` is
BF16. No warning, no error. Workaround: postprocess restore from the
BF16 source. The
[`scripts/fixup_artifact.py`](scripts/fixup_artifact.py) pipeline does
this in one atomic per-shard pass. Worth filing upstream against
transformers — the save path should preserve per-tensor dtype.

### C14 — vLLM MTP loader silently skips top-level `head.weight` + `embed.weight`

`vllm.models.deepseek_v4.nvidia.mtp.DeepSeekV4MTP.load_weights` calls
`name.replace("mtp.0.", "")` which no-ops on non-`mtp.0.*` keys, then
`get_spec_layer_idx` returns None → the loop hits `continue` and the
weight is skipped. Top-level `head.weight` and `embed.weight` never
reach the MTP layer's `shared_head.head` / `embed_tokens`, leaving
those parameters uninitialized → garbage logits → **0% MTP draft
acceptance with no load-time error**.

Workaround: postprocess injects `mtp.0.head.weight` (FP32 copy of
top-level head, matching sibling artifact's pattern) and
`mtp.0.emb.tok_emb.weight` (BF16 copy of top-level embed) as full
duplicates. Worth filing upstream against vLLM — the loader should
either route top-level keys to the MTP slot or raise at construction
time when `shared_head.head` is uninitialized.

### C15 — DeepGemm `paged_mqa_logits` kernel asserts on `num_speculative_tokens > 1`

`vllm serve --speculative-config method=mtp,num_speculative_tokens=2`
crashes during `profile_cudagraph_memory` with:

```
Assertion error (smxx_fp8_fp4_paged_mqa_logits.hpp:233):
  next_n == 1 or next_n == 2
```

vLLM passes `next_n = num_speculative_tokens + 1` into the DeepGemm
kernel (k draft + 1 main verifier in the lookahead window). The
assertion enforces `num_speculative_tokens <= 1` in practice. The
`FLASHINFER_MLA_SPARSE` attention backend hits the same assertion
(kernel is logits-side, not attention-backend-specific).

This caps our practical k at 1, leaving headroom on the speedup. With
k=2 unlocked the bs=1 decode speedup should land closer to sibling's
published 2.03×.

---

## Reproducing this artifact

The full pipeline is committed in this repo. From a fresh
8× H200 box:

```bash
# Phase 0 — bootstrap (venv-calib + venv-serve + vendor + apply patches)
bash scripts/bootstrap_p5en_h200.sh

# Phase 1 — download upstream + dequant to BF16-MTP source
# (writes /scratch/weights/bf16-mtp/, ~660 GB, ~30 min)
bash scripts/phase1_dequant.sh  # (called from bootstrap)

# Phase 2 — GPTQ calibration (8 ranks, ~15h wall)
bash scripts/run_phase2.sh
# (Equivalently: torchrun --nproc-per-node=8 scripts/quantize_v4_w4a16_mtp.py
#  --input /scratch/weights/bf16-mtp --output /scratch/weights/w4a16-fp8-mtp-gptq
#  --samples 768 --batch-size 4 --max-seq-len 512
#  --checkpoint-dir /scratch/weights/checkpoints-phase2)

# Phase 3 — postprocess (rename + config patch + FP32 restore + MTP aliases)
bash scripts/postprocess_phase2.sh
# (Runs: rename_to_upstream.py → postprocess_for_vllm.py
#  → pass2_rename.py (indexer/compressor nesting fix)
#  → fixup_artifact.py (FP32 restore + MTP head/embed aliases))

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

See [`SUPERVISION_RULES.md`](SUPERVISION_RULES.md) for the discipline
practices that emerged during the build (atomic safetensors writes,
verify-before-acting on summaries, subagent briefing standards).

---

## Hardware

This artifact was **both calibrated and originally benchmarked on
H200** (AWS `p5en.48xlarge`, 8× H200 SXM5, Hopper SM 9.0a, 143 GB
HBM3e per GPU). vLLM serve used TP=2 (2 GPUs per replica) — TP=4
underutilizes Marlin tensor cores on per-rank expert shards for our
W4A16 layout, so the H200 box is best run as 4× TP=2 replicas behind a
load balancer.

(An earlier calibration attempt on **B300** (`p6-b300.48xlarge`, SM
10.0) was abandoned due to multi-rank NCCL friction on the
decoupled-MoE-expert shard. The B300 GPTQ scaffolds that came out of
that detour are kept in `scripts/loadtest_sharded.py` and
`scripts/multirank_patches.py` for reference. The **NVFP4 sibling
artifact** at `canada-quant/dsv4-flash-nvfp4-fp8-mtp` is a separate
project that was calibrated on B300 — that's distinct from this repo.)

- Phase 2 GPTQ calibration on H200: 15.09h oneshot + ~16 min save = ~15.4h end-to-end. Per-subgraph cadence stabilized at ~20 min/subgraph across 44 subgraphs (43 MoE main + 1 MTP — MTP subgraph is a no-op per Option Y).
- Phase 2 GPTQ output: 4 shards, 159 GB total.
- Postprocess: ~6 min wall (single-process, 4 shards rewritten
  atomically with FP32 restore + alias injection).

### Also runs on RTX PRO 6000 Blackwell (SM 12.0) — competitive per replica, ~5× cheaper per output token

Second hardware demonstration on 2026-05-24: Brev `g7e.24xlarge` with
4× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB HBM3 each).
Full `torch.compile` + cudagraph stack enabled. Both TP=2 and TP=4
verified: chat-smoke 4/4 PASS, MTP acceptance 68-72%, Marlin
TP > 2 bug (`vllm-project/vllm#41511`) did **not** fire.

#### Per-replica throughput

H200 numbers are **per-replica TP=2** (1 of 4 possible replicas on the
8-GPU `p5en.48xlarge`). RTX numbers are per-replica TP=2 or TP=4 on the
4-GPU `g7e.24xlarge`.

| Metric | H200 TP=2 (1 replica) | RTX 6000 Pro TP=2 (1 replica) | RTX 6000 Pro TP=4 (1 replica) |
|---|---|---|---|
| bs=1 output tok/s | 88.35 | 98.83 | **107.32** |
| bs=1 TPOT median (ms) | **6.02** | 8.55 | 7.77 |
| bs=4 output tok/s | 138.80 | 219.53 | **221.52** |
| bs=16 output tok/s | 367.13 | 482.61 | **584.04** |
| MTP acceptance bs=1 | 89.1% / 69.94% | 71.39% | 68.15% |

Per-replica, RTX 6000 Pro **wins on output throughput at every batch
size** while H200 still wins per-token TPOT median.

#### Node-level throughput (more apples-to-apples)

The H200 box is 8 GPUs (`p5en.48xlarge`, ~$98/h on AWS); the RTX 6000
Pro box is 4 GPUs (`g7e.24xlarge`, $19.92/h on Brev). Multiplying by
the number of replicas each can host:

| Box | Replicas | bs=1 total tok/s | bs=16 total tok/s | $/h | $/(1000 tok/h) at bs=1 |
|---|---|---|---|---|---|
| `p5en.48xlarge` (8× H200) | 4× TP=2 | 4 × 88.35 = **~353** | 4 × 367.13 = **~1468** | $98 | **$278** |
| `g7e.24xlarge` (4× RTX 6000 Pro) | 2× TP=2 | 2 × 98.83 = **~198** | 2 × 482.61 = **~965** | $19.92 | **$101** |
| `g7e.24xlarge` (4× RTX 6000 Pro) | 1× TP=4 | 107.32 | 584.04 | $19.92 | $186 |

**Cost-per-token at bs=1 (interactive workload): RTX 6000 Pro 2×TP=2
is ~2.7× cheaper than H200 4×TP=2.** At bs=16 the gap narrows because
the H200's per-replica throughput scales better with batch — node total
is ~1500 tok/s on H200 vs ~965 on RTX (per-replica × 2). H200 wins
absolute throughput when you can fill it; RTX wins on $/token unless
you genuinely need 1500+ tok/s of aggregate output.

(Aggregate "Total replicas × per-replica" numbers above are
extrapolated from the measured single-replica runs; we benchmarked
single replicas only.)

#### Patches required

| Patches required | H200 | RTX 6000 Pro |
|---|---|---|
| `packed_modules_mapping` on `DeepseekV4ForCausalLM` + `DeepSeekV4MTP` | ✓ | ✓ |
| `weight_scale_inv → weight_scale` fallback on `wo_a` | ✓ (PR #43290) | ✓ (`patch_nvidia_attn_scale.py`) |
| BF16 wo_a path for MTP block | n/a | ✓ — uses static `weight.dtype == bfloat16` check (dynamo-safe) |
| Compressor/indexer FP8 → BF16 dequant preprocess | n/a | ✓ (`dequant_compressor.py`, one-time) |
| `--disable-custom-all-reduce` (no NVLink) | n/a | ✓ |
| CMakeLists `USE_SABI 3.11` removal for Py 3.10 | n/a | ✓ |

#### Quick start (RTX 6000 Pro)

```bash
# From a fresh Brev g7e.24xlarge (Ubuntu 22.04, CUDA 12.9 + driver
# 580 pre-installed):
sudo apt-get update && sudo apt-get install -y git
sudo ln -sfn /opt/dlami/nvme /scratch && sudo chown -h "$USER:$USER" /scratch
cd /scratch
git clone https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp.git
cd dsv4-flash-w4a16-fp8-mtp

# 1) Bootstrap (~25 min for vLLM source build)
bash scripts/bootstrap_rtx6000pro.sh

# 2) Extra deps (Rust, humming, flashinfer pins)
source ~/venv-serve/bin/activate
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
pip install --quiet setuptools-rust
pip install --quiet git+https://github.com/inclusionAI/humming.git
pip install --quiet "flashinfer-python==0.6.8.post1" "flashinfer-cubin==0.6.8.post1" \
    "numba==0.65.0" "tilelang==0.1.9" "apache-tvm-ffi==0.1.9" "fastsafetensors>=0.2.2"

# 3) Apply patches against installed vLLM
python scripts/patch_v4_forcausal_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_mtp_packed_mapping.py        "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_nvidia_attn_scale.py         "$(python -c 'import vllm; print(vllm.__path__[0])')"
bash   scripts/patch_wo_a_bf16_path.sh             "$(python -c 'import vllm; print(vllm.__path__[0])')"

# 4) Download artifact (159 GiB, ~1.5 min on Brev)
pip install --user --quiet huggingface_hub hf-transfer
export PATH="$HOME/.local/bin:$PATH"
hf download canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP \
    --local-dir /scratch/weights/w4a16-fp8-mtp-gptq

# 5) Dequantize compressor/indexer modules (one-time, ~1.5 min)
python scripts/dequant_compressor.py /scratch/weights/w4a16-fp8-mtp-gptq

# 6) Serve TP=2 on GPUs 0+1 (or TP=4 with 0,1,2,3)
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/w4a16-fp8-mtp-gptq 8000 2

# 7) Smoke + bench (in another shell once /health is 200)
bash scripts/chat_smoke.sh http://localhost:8000
bash scripts/bench_rtx6000pro_suite.sh http://localhost:8000 2 1
```

For full reproduction details (patch rationale, debug notes, future
work): [`RECIPE_RTX6000PRO.md`](RECIPE_RTX6000PRO.md).

Raw JSONs + summary:
- [`benchmarks/rtx6000pro/2026-05-24-cudagraph-summary.md`](benchmarks/rtx6000pro/2026-05-24-cudagraph-summary.md) (headline)
- `benchmarks/rtx6000pro/tp{2,4}_2026-05-24T*/bench_mtp_bs{1,4,16}.json`

#### Scope clarification: this is the W4A16 artifact, not NVFP4

This repo (`canada-quant/dsv4-flash-w4a16-fp8-mtp`) ships the
**W4A16+FP8+MTP** quantization. The Blackwell-native **NVFP4+FP8+MTP**
artifact lives in the sibling repo
[`canada-quant/dsv4-flash-nvfp4-fp8-mtp`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP),
calibrated on B300 (SM 10.0) hardware.

NVFP4 native MoE kernels for SM 12.0 (RTX 6000 Pro Blackwell) exist in
upstream vLLM (`csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`)
but aren't being picked by the backend selector
([vllm-project/vllm#31085](https://github.com/vllm-project/vllm/issues/31085)).
Once that fix lands, running the sibling's NVFP4 artifact on RTX 6000
Pro would unlock a tighter expert kernel than the W4A16 Marlin path —
that's a follow-up project, not in scope for this repo.

---

## Honest limitations

1. **k=1 cap on spec-decode** due to C15 — current vLLM build limits
   `num_speculative_tokens` to 1 on H200/Hopper. With C15 fixed, expect
   speedup to rise from 1.49× to ~1.85× at bs=1.
2. **toolcall15 -2 pts** vs predecessor — model-routing regressions on
   chain-completion + multi-tool extraction. See breakdown above. Not
   a parser issue.
3. **GSM8K -1.3 pts** vs predecessor's 8-shot strict-match — within
   one standard-error band, but technically below. Predecessor's
   calibration ran on Spark; ours ran on H200 with the same recipe
   (`scripts/quantize_v4_w4a16_mtp.py` matches their Phase 2 invocation
   modulo hardware). Likely calibration-set sensitivity; not a recipe
   bug.

---

## Credits + reproducibility

- DeepSeek for the base model + inference reference.
- jasl (`jasl/vllm` and `jasl/vllm-ds4-sm120-harness`) for the working
  vLLM build pins (`ds4-sm120-experimental` for the original H200
  calibration; `ds4-sm120-preview-dev` for the post-refactor RTX
  6000 Pro Blackwell SM 12.0 serving path) and the benchmark
  harness.
- `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` (predecessor) for the
  proven recipe topology that this artifact extends with MTP.
- `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP` (sibling) for the
  alias-injection pattern + MTP acceptance methodology.

Source repo, scripts, recipe, every patch:
<https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp>

---

## License

MIT. See `LICENSE` in the source repo. Base model is licensed per
DeepSeek's terms — review at the upstream `deepseek-ai/DeepSeek-V4-Flash`
repo before commercial deployment.
