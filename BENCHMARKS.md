# Phase 2 benchmark log

All raw benchmark outputs land under `benchmarks/phase2/`. Each row in this file
points at the raw JSON / JSONL for traceability. The H200 capacity block expires
2026-05-23 04:30 PDT — these raw logs are the proof.

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
| 2026-05-22 P1 GSM8K | lm-eval 0.4.11 local-completions, num_fewshot=8, c=8, full 1319 prompts | **93.71% strict-match (93.63% flex)** ± 0.67 | predecessor HF card 92.87% (5-shot flex) → **+0.84 pts**; predecessor phase4e 8-shot 94.99% strict → **-1.28 pts**; RedHat NVFP4 91.0% → **+2.71 pts** | [`benchmarks/phase2/gsm8k_phase2_2026-05-22.json`](benchmarks/phase2/gsm8k_phase2_2026-05-22.json) + `.log` |
| 2026-05-22 P1 toolcall15 | jasl harness `toolcall15`, ds4_harness HEAD 85aca32 | **24/30 (80%)**, 3 fails | predecessor 26/30 → **-2 pts** (same band) | [`benchmarks/phase2/harness_2026-05-22T204111Z/toolcall15.json`](benchmarks/phase2/harness_2026-05-22T204111Z/toolcall15.json) |
| 2026-05-22 P1 chat-smoke quick | jasl harness | **4/4 PASS** | matches predecessor 4/4 | [`benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quick.jsonl`](benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quick.jsonl) |
| 2026-05-22 P1 chat-smoke quality | jasl harness | **4/4 PASS** | matches predecessor 4/4 | [`benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quality.jsonl`](benchmarks/phase2/harness_2026-05-22T204111Z/chat_smoke_quality.jsonl) |
| 2026-05-22 P1 chat-smoke coding | jasl harness (max-model-len 16384 retry — first attempt token-restricted) | **2/2 PASS** | matches predecessor 2/2 | [`benchmarks/phase2/coding_2026-05-22T204521Z/chat_smoke_coding.jsonl`](benchmarks/phase2/coding_2026-05-22T204521Z/chat_smoke_coding.jsonl) |
| 2026-05-22 P2 MMLU | lm-eval 0.4.11, 5-shot, c=8, 57 subtasks, full set | **86.88% acc** ± 0.27 | predecessor 87.27% → **-0.39 pts** (within SE) | [`benchmarks/phase2/mmlu_phase2_2026-05-22.json`](benchmarks/phase2/mmlu_phase2_2026-05-22.json) + `.log` |
| 2026-05-22 P2 HumanEval | lm-eval 0.4.11 `humaneval_instruct`, 0-shot, c=8, 164 prompts, pass@1 via code_eval | **84.76% pass@1** ± 2.82 | predecessor 54.27% (flagged as strict-regex artifact); our number uses default flexible extraction | [`benchmarks/phase2/humaneval_phase2_2026-05-22.json`](benchmarks/phase2/humaneval_phase2_2026-05-22.json) + `.log` |
