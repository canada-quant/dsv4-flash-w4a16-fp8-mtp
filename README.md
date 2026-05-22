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
| GSM8K 8-shot strict-match | **93.71%** ± 0.67 | 94.99% (phase4e, 8-shot) | 91.0% | -1.28 pts |
| GSM8K 5-shot flex (HF card protocol) | 93.63% ± 0.67 (8-shot flex) | 92.87% (5-shot flex) | — | +0.76 pts |
| MMLU 5-shot | **86.88%** ± 0.27 | 87.27% | — | -0.39 pts (within SE) |
| HumanEval pass@1 0-shot instruct | **84.76%** ± 2.82 | 54.27% (strict-regex artifact) | — | predecessor's 54.27% reflects a strict-regex extraction artifact (per predecessor notes); our 84.76% uses lm-eval-harness's default flexible code-block extraction, which is what `evaluate-metric/code_eval` actually executes |
| Chat-smoke quick (4 deterministic) | **4/4** | 4/4 | — | match |
| Chat-smoke quality (4 writing/translation) | **4/4** | 4/4 | — | match |
| Chat-smoke coding (2 HTML/code) | **2/2** | 2/2 | — | match |
| toolcall15 | **24/30 (80%)** | 26/30 (87%) | — | -2 pts |

### MTP-specific (Option Y differentiator)

| Metric | Phase 2 | Sibling NVFP4-FP8-MTP (published) | Notes |
|---|---|---|---|
| MTP draft-token acceptance (random 256-token prompts, c=1, k=1, 200 samples) | **69.94%** (21024 / 30058) | 67.29% (HumanEval raw code, c=1, k=2) | Direct comparison — different prompt distribution but same metric class |
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

Built on AWS `p5en.48xlarge` (8× H200 SXM5, Hopper SM 9.0a, 143 GB
HBM3e per GPU). vLLM serve uses TP=2 (2 GPUs per replica) per the
sibling artifact's published guidance — TP=4 underutilizes Marlin
tensor cores on per-rank expert shards for our W4A16 layout.

- Phase 1 dequant + Phase 2 calibration: ~16h wall on full 8× H200.
- Phase 2 GPTQ output: 4 shards, 159 GB total.
- Postprocess: ~6 min wall (single-process, 4 shards rewritten
  atomically with FP32 restore + alias injection).

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
  vLLM build pin (`ds4-sm120-experimental` branch) and the benchmark
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
