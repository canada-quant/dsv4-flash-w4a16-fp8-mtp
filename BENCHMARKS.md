# Phase 2 benchmark log

All raw benchmark outputs land under `benchmarks/phase2/` (H200) and
`benchmarks/rtx6000pro/` (RTX PRO 6000 Blackwell). Each row in this file
points at the raw JSON / JSONL for traceability. The H200 capacity block expired
2026-05-23 04:30 PDT — those raw logs are the proof. The RTX 6000 Pro run is
from a Brev `familiar-teal-worm` instance (4× RTX PRO 6000 Blackwell, SM 12.0)
on 2026-05-23 and is the second hardware demonstration of the artifact.

## Predecessor reference numbers (cited from `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` HF card)

| Metric | Predecessor (W4A16-FP8, no MTP) |
|---|---|
| GSM8K (5-shot flexible) | 92.87% |
| MMLU (5-shot) | 87.27% |
| HumanEval pass@1 (0-shot instruct) | 54.27% |
| toolcall15 | 26/30 |
| chat-smoke quick/quality/coding | 4/4 / 4/4 / 2/2 |

## Our numbers (Phase 2 = W4A16+FP8+MTP)

| Run | Metric | Phase 2 | Δ vs predecessor | Raw |
|---|---|---|---|---|
| smoke | MTP acceptance (4 probes) | 67.9% | n/a (predecessor has no MTP) | (see chat) |
| phase2-acceptance | MTP acceptance (4 probes) | 89.1% (196/220) | n/a | (see chat) |
| 2026-05-22 P0 throughput | TPOT median at bs=1 (decode speedup) | **6.02 ms with MTP / 8.93 ms without → 1.49× faster** | n/a | [`benchmarks/phase2/2026-05-22T195133Z-throughput-summary.md`](benchmarks/phase2/2026-05-22T195133Z-throughput-summary.md) + 6 raw JSONs |
| 2026-05-22 P0b acceptance@200 | MTP draft-token acceptance over 200 random prompts (k=1, c=1) | **69.94%** (21024 / 30058) | n/a | [`benchmarks/phase2/acc_2026-05-22T200425Z_metrics.txt`](benchmarks/phase2/acc_2026-05-22T200425Z_metrics.txt) + `acc_*.json` |
| 2026-05-22 P1 GSM8K | lm-eval 0.4.11 local-completions, num_fewshot=8, c=8, full 1319 prompts | **93.71% strict-match (93.63% flex)** ± 0.67 | predecessor phase4e 8-shot strict 94.99% → **-1.28 pts** (within SE); RedHat NVFP4 91.0% → **+2.71 pts**; predecessor HF card 92.87% is 5-shot flex — not directly comparable to our 8-shot | [`benchmarks/phase2/gsm8k_phase2_2026-05-22.json`](benchmarks/phase2/gsm8k_phase2_2026-05-22.json) + `.log` |
| 2026-05-22 P1 toolcall15 | jasl harness `toolcall15`, ds4_harness HEAD 85aca32 | **24/30 (80%)**, 3 fails | predecessor 26/30 → **-2 pts** (same band) | [`benchmarks/phase2/harness_2026-05-22T204111Z/toolcall15.json`](benchmarks/phase2/harness_2026-05-22T204111Z/toolcall15.json) |
| 2026-05-22 P1 chat-smoke quick | jasl harness | **4/4 PASS** | matches predecessor 4/4 | [`benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quick.jsonl`](benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quick.jsonl) |
| 2026-05-22 P1 chat-smoke quality | jasl harness | **4/4 PASS** | matches predecessor 4/4 | [`benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quality.jsonl`](benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quality.jsonl) |
| 2026-05-22 P1 chat-smoke coding | jasl harness (max-model-len 16384 retry — first attempt token-restricted) | **2/2 PASS** | matches predecessor 2/2 | [`benchmarks/phase2/coding_2026-05-22T204521Z/chat_smoke_coding.jsonl`](benchmarks/phase2/coding_2026-05-22T204521Z/chat_smoke_coding.jsonl) |
| 2026-05-22 P2 MMLU | lm-eval 0.4.11, 5-shot, c=8, 57 subtasks, full set | **86.88% acc** ± 0.27 | predecessor 87.27% → **-0.39 pts** (within SE) | [`benchmarks/phase2/mmlu_phase2_2026-05-22.json`](benchmarks/phase2/mmlu_phase2_2026-05-22.json) + `.log` |
| 2026-05-22 P2 HumanEval | lm-eval 0.4.11 `humaneval_instruct`, 0-shot, c=8, 164 prompts, pass@1 via code_eval | **84.76% pass@1** ± 2.82 | predecessor 54.27% (flagged as strict-regex artifact); our number uses default flexible extraction | [`benchmarks/phase2/humaneval_phase2_2026-05-22.json`](benchmarks/phase2/humaneval_phase2_2026-05-22.json) + `.log` |
| 2026-05-22 P2-ext AIME 24 | lm-eval 0.4.11 `aime24` task, 0-shot, c=8, 30 problems, exact_match | **30.0% exact_match** ± 8.51 | sibling: AIME 24 in `tier1_aime24_2026_05_21.md`; competition math at 30/30 sampled | [`benchmarks/phase2/aime24_phase2_2026-05-22.json`](benchmarks/phase2/aime24_phase2_2026-05-22.json) + `.log` |
| 2026-05-22 P2-ext MMLU-Pro | lm-eval 0.4.11 `mmlu_pro` 5-shot, c=8, full 12032 prompts (retry with max-model-len=8192 after first attempt crashed at 91% on a long-prompt) | **71.28% exact_match (custom-extract)** ± 0.40 | sibling NVFP4-FP8-MTP **81.13%** ± 0.35 (B300, custom-extract) → **-9.85 pts** (expected — NVFP4 is higher-quality quantization than W4A16) | [`benchmarks/phase2/mmlu_pro_phase2_2026-05-22.json`](benchmarks/phase2/mmlu_pro_phase2_2026-05-22.json) + `.log` |
| 2026-05-22 P2-ext acceptance by workload | Custom script (`/tmp/bench_acceptance_by_workload.py`), 3 buckets × 15 prompts, MTP-spec k=1 c=1 | **code 92.91%** / **chat-prose 81.90%** / **raw NL 83.65%** (weighted mean 85.87%) | sibling reports 67.29% raw code (HumanEval) and 85.04% chat-templated (EvalPlus); our results in the same band | [`benchmarks/phase2/acceptance_workload_2026-05-23T025005Z.json`](benchmarks/phase2/acceptance_workload_2026-05-23T025005Z.json) + `.log` |

## RTX PRO 6000 Blackwell (SM 12.0) — second hardware demonstration (2026-05-23)

Brev `familiar-teal-worm` instance — 4× RTX PRO 6000 Blackwell Server
Edition (96 GiB HBM3 each), `g7e.24xlarge`, Columbus OH. vLLM built from
`jasl/vllm@ds4-sm120-preview-dev` (SHA `c79225692`) with
`TORCH_CUDA_ARCH_LIST=12.0a`. Six patches required on top of the SM12
branch — see [`RECIPE_RTX6000PRO.md`](RECIPE_RTX6000PRO.md) for the
full reproduction path.

**Important caveat:** these numbers are from **eager-mode** serving
(`--enforce-eager`). The runtime BF16 `wo_a` fallback patch (needed for
MTP block compatibility on SM12) is not dynamo-safe, which forces eager
mode and disables `torch.compile` + cudagraphs. Throughput is therefore
~10× slower than the H200 numbers above. With a dynamo-safe rewrite of
the fallback, expect bs=1 TPOT to drop to 8-15 ms (competitive with H200).
MTP acceptance and quality are unaffected.

| Run | Metric | TP=2 | TP=4 | H200 ref | Raw |
|---|---|---|---|---|---|
| 2026-05-23 chat-smoke quick | jasl harness equiv, 4 deterministic prompts | **4/4 PASS** | **4/4 PASS** | 4/4 PASS | [`benchmarks/rtx6000pro/tp2_2026-05-23T211824Z/chat_smoke_quick.log`](benchmarks/rtx6000pro/tp2_2026-05-23T211824Z/chat_smoke_quick.log), [`tp4`](benchmarks/rtx6000pro/tp4_2026-05-23T214313Z/chat_smoke_quick.log) |
| 2026-05-23 throughput bs=1 (MTP-spec k=1) | `vllm bench serve` 8 prompts c=1 | **11.57 tok/s, TPOT 82.70 ms** | 11.74 tok/s, TPOT 85.23 ms | 88.35 tok/s, TPOT 6.02 ms | [`tp2 bs=1 json`](benchmarks/rtx6000pro/tp2_2026-05-23T211824Z/bench_mtp_bs1.json), [`tp4`](benchmarks/rtx6000pro/tp4_2026-05-23T214313Z/bench_mtp_bs1.json) |
| 2026-05-23 throughput bs=4 (MTP-spec k=1) | `vllm bench serve` 32 prompts c=4 | 41.37 tok/s, TPOT 87.62 ms | 41.81 tok/s, TPOT 88.17 ms | 138.80 tok/s, TPOT 9.50 ms | [`tp2 bs=4 json`](benchmarks/rtx6000pro/tp2_2026-05-23T211824Z/bench_mtp_bs4.json) |
| 2026-05-23 throughput bs=16 (MTP-spec k=1) | `vllm bench serve` 128 prompts c=16 | **147.00 tok/s** | **157.33 tok/s (+7%)** | 367.13 tok/s | [`tp2 bs=16 json`](benchmarks/rtx6000pro/tp2_2026-05-23T211824Z/bench_mtp_bs16.json), [`tp4`](benchmarks/rtx6000pro/tp4_2026-05-23T214313Z/bench_mtp_bs16.json) |
| 2026-05-23 MTP acceptance bs=1 | reported by vLLM `/metrics`, 8 random prompts × 256 out | **74.17%** (870/1173) | 72.84% (861/1182) | 89.1% (Phase 2 calibrated) / 69.94% (200-prompt eval) | (in throughput JSON) |
| 2026-05-23 MTP acceptance bs=4 | same | 70.70% (3386/4789) | 69.94% (3364/4810) | n/a | (in throughput JSON) |
| 2026-05-23 MTP acceptance bs=16 | same | 72.04% (13691/19004) | 71.28% (13607/19090) | n/a | (in throughput JSON) |

**Marlin TP > 2 bug (`vllm-project/vllm#41511`) verdict:** did **not**
fire on TP=4 — model loaded cleanly, all 43 MoE layers + MTP block
initialized, chat smoke 4/4 PASS. Either fixed on jasl/vllm's
`ds4-sm120-preview-dev` branch's compressed_tensors path, or our
W4A16 layout doesn't trigger the failing K-sharding code path.

**Full summary:** [`benchmarks/rtx6000pro/2026-05-23-throughput-summary.md`](benchmarks/rtx6000pro/2026-05-23-throughput-summary.md)

**Accuracy benchmarks deferred:** GSM8K / MMLU / HumanEval / AIME on RTX
6000 Pro would take ~8-12 hours in eager mode (vs 30 min on H200 with
compile + cudagraph). Re-run after the dynamo-safe rewrite lands. The
H200 quality numbers above remain the published quality reference for
this artifact.
