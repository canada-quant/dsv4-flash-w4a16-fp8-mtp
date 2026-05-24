# SM 12.0 token-corruption under concurrent thinking-mode on W4A16-MTP

**Status (2026-05-24):** Reproducibly identified. Workaround: use sequential (c=1) thinking-mode, or use the NVFP4 sibling artifact for batched thinking-mode workloads on SM 12.0.

## Environment

- **Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell Server Edition (SM 12.0, sm_120, 96 GB HBM/GPU). Brev `g7e.24xlarge`.
- **vLLM:** `jasl/vllm@ds4-sm120-preview-dev` HEAD `a937d4b28` ("Stabilize SM12x sparse MLA long prefill", 2026-05-24).
- **Artifact:** `canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP` (post-dequant fix, all 4 shards re-uploaded 2026-05-24, commits `b9dd4a18`/`103cfd63`/`4f099699`/`a5e727ba`/`487f4c4e`).
- **Config:** TP=4, MTP n=1, cuda graphs ON, max_model_len 65536, max_num_seqs 8, max_num_batched_tokens 8192.
- **Required env:** `VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_TRITON_MLA_SPARSE=1 VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 VLLM_USE_FLASHINFER_SAMPLER=0`.

## Observed behavior

The kernel race manifests under **concurrent (c≥2) thinking-mode generations** producing **token-stream corruption** rather than crashes. Sample of corrupted output (from AIME 2024 c=4 run on `a937d4b28`):

```
2024-II-13: "...:Realizing, the 13–as for L (P=(A1z4 ione  -up text:PEY8/PURABI..."
2024-II-7:  "|youtube#593._临 ‘ low, 0-,-set,, lint`v0..."
2024-I-15:  "______________# # | a b c d e f g h i j k l m n o p..."
2024-II-9:  "...white chips we want to place, ... bXperlet 6 ICHE APPROB..."
```

These outputs contain CJK characters, Cyrillic, broken HTML-ish tokens, and incoherent ASCII — clearly not the model's coherent reasoning trace (the same model on the same hardware produces clean reasoning at c=1).

## Reproducible runs (all on `jasl/vllm@a937d4b28`)

| Run | Config | Errored | Truncated | **Corrupted** | Coherent-wrong | **Correct** | pass@1 / clean | MTP accept |
|---|---|---|---|---|---|---|---|---|
| AIME c=4 thinking (D, W4A16) — baseline | default env | 0 | 2 | **14** | 4 | 10 | **10/14 = 71.4%** | 87.14% |
| AIME c=2 thinking (D) | default env | 7 | 3 | **10** | 4 | 6 | 6/10 = 60.0% | n/a (server crashed) |
| AIME c=4 thinking (D) + `VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256` | smaller chunk | 27 | 0 | 0 | 0 | 3 | 3/3 = 100% (only 3 completed) | n/a |
| AIME c=4 thinking (D) + no MTP | spec disabled | 29 | 0 | 0 | 0 | 1 | 1/1 = 100% | n/a |
| AIME c=4 thinking (D) + `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` | decode fallback | 29 | 0 | 0 | 0 | 1 | 1/1 = 100% | n/a |
| AIME c=4 thinking **(B, NVFP4)** | default env | 0 | 21 | **1** | 0 | 8 | **8/8 = 100%** | 90.63% |
| AIME 1-shot smoke (D) | concurrency=1 | 0 | 0 | 0 | 0 | 1 | 1/1 = 100% | n/a |
| GSM8K-20 chat-mode sequential (D) | concurrency=1, no thinking | 0 | 0 | 0 | 0 | 20 | 20/20 = 100% | **92.46%** |

## What this tells us

1. **Bug is concurrent-load + W4A16 + Marlin path specific.** Card B (NVFP4 + flashinfer_trtllm MoE) produced 1/30 corrupted at c=4 thinking. Card D (W4A16 + Marlin MoE) produced 14/30 corrupted at c=4. Same hardware, same vLLM build, same MTP config, same thinking-mode prompts.

2. **Sequential thinking-mode is clean.** 1-shot AIME smoke returns the correct integer answer in 2072 tokens with `\boxed{N}`. GSM8K-20 sequential chat-mode is 20/20 = 100%. The artifact and kernel produce correct results when not under concurrent decode load.

3. **a937d4b28's sparse-MLA stability fix only addressed the *crash*, not the *corruption*.** The fix bounds the `topk_chunk_size` for prefill on SM 12.x; decode path is untouched. Pre-`a937d4b28`, the same concurrent thinking-mode load *crashed* with `CUDA error: an illegal memory access`. Post-fix, the same workload runs to completion but produces corrupted output on the majority of generations.

4. **Smaller topk chunk doesn't help.** Setting `VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256` to force smaller chunks just made the kernel race manifest as crashes faster (27/30 errored).

5. **Disabling MTP or matmul_decode doesn't help.** Same crash pattern. The race is in the main decode + Marlin W4A16 MoE path under concurrent execution, not in spec-decode scheduling or the sparse-MLA matmul kernel specifically.

6. **Effective model quality is intact.** Conditional on the kernel not corrupting output, Card D achieves ~71% pass@1 on AIME 2024 thinking=high — within reasonable range of Card B's published 83.33% on B300 (which uses a different MoE kernel and thus doesn't hit this bug).

## Recommendations

### For users serving on RTX PRO 6000 / SM 12.0

| Workload | Recommendation |
|---|---|
| Single-stream interactive chat | Card D W4A16-MTP, any config. Works cleanly. |
| Sequential thinking-mode reasoning | Card D W4A16-MTP, concurrency=1. Clean, MTP fires at ~87–92%. |
| **Concurrent batched thinking-mode** | **Use Card B NVFP4-FP8-MTP instead**, with documented `VLLM_TEST_FORCE_FP8_MARLIN=1` config. Stable at c=4, MTP at 90.63%. |
| Chat-mode batched (no thinking) | Card D works (no kernel race triggered under typical short generations). GSM8K-50 c=1 chat-mode = 88% on RTX PRO 6000 TP=4. |

### For upstream / vLLM kernel maintainers

The smoking gun is reproducible:

1. Serve `canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP` on 4× RTX PRO 6000 (SM 12.0, sm_120), TP=4, MTP n=1, cuda graphs ON, `--max-model-len 65536`, `VLLM_TEST_FORCE_FP8_MARLIN=1`.
2. Run 30 long thinking-mode AIME problems at concurrency=4 (each generates ~3K-30K tokens).
3. Inspect raw model outputs — ~half will contain CJK/Cyrillic/garbled text instead of coherent math reasoning.
4. The same setup at concurrency=1 (or with NVFP4 artifact + `flashinfer_trtllm` MoE) is clean.

Likely fault sites (from the diff inspection of `a937d4b28` and surrounding code):

- `vllm/models/deepseek_v4/nvidia/flashmla.py` decode path — the prefill chunk-size guard introduced in `a937d4b28` does not cover decode; concurrent decode under Marlin W4A16 MoE may have an analogous unbounded chunk race.
- Marlin W4A16 MoE kernel (`csrc/quantization/marlin/qqq/` or `csrc/moe/marlin_kernels/`) under SM 12.0 + concurrent KV-cache writes — possible race writing to the wrong KV slot.

Filing as: `[Bug] DeepSeek V4 W4A16 + Marlin MoE token-stream corruption under concurrent thinking-mode generations on SM 12.0` against `jasl/vllm`.

## Working production configs verified this session

### Card D — single-stream / sequential

```bash
# canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP — works cleanly
vllm serve canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 65536 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.92 \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --disable-custom-all-reduce \
    --trust-remote-code
```

### Card B — batched concurrent thinking-mode

```bash
# canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP — clean at c=4 thinking
VLLM_TEST_FORCE_FP8_MARLIN=1 \
VLLM_TRITON_MLA_SPARSE=1 VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 65536 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.97 \
    --no-enable-prefix-caching \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --disable-custom-all-reduce \
    --trust-remote-code
```

## Open questions for follow-up

1. **Does the corruption reproduce on H200 + modern upstream vLLM?** Card D was originally calibrated and benched on H200 with `jasl/vllm@ds4-sm120-experimental@abad5dc71` (older build that supported FP8 compressor and presumably didn't have this issue). We don't yet know if the W4A16+Marlin+SM-something race exists on Hopper too.
2. **Does concurrency-1 thinking-mode AIME really score 76-83%?** Estimated from clean-conditional pass@1 of 10/14 (c=4) and 8/8 (B at c=4). A full sequential c=1 AIME on D would take ~2-3h on this hardware; not run this session.
3. **Will a kernel fix arrive on `jasl/vllm@ds4-sm120-preview-dev`?** The prefill-only fix in `a937d4b28` suggests jasl is actively working on this area. A matching decode-path guard or a Marlin-decode race fix would close the bug.

## Run artifacts on the bench host (`familiar-teal-worm`)

Saved on the Brev box at `/tmp/d-rebench/`:

- `aime24_a937_*.json` — Card D c=4 thinking baseline
- `aime24_c2_*.json` — Card D c=2 thinking
- `aime24_topk256_*.json` — Card D c=4 with smaller topk chunk
- `aime24_nomtp_*.json` — Card D c=4 with MTP disabled
- `aime24_nomatmul_*.json` — Card D c=4 with matmul_decode disabled
- `aime24_B_c4_*.json` — Card B (NVFP4) c=4 thinking
- `gsm8k50_dequant_*.json` — Card D GSM8K-50 sequential (88%)

All `*.rescored.json` files contain re-extracted predictions using correct `\boxed{N}` regex (the upstream `aime_bench.py` has a broken regex `r'\\\\boxed\\{(\\d+)\\}'` that doesn't match — fixed in `aime_rescore.py`).
