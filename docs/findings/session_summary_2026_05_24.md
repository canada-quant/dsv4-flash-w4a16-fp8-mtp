# Canada Quant session summary — 2026-05-24 → 2026-05-25 (RTX PRO 6000 leg)

## What shipped this session

### Artifact-level fixes
- **Card D shipping bug fixed.** Compressor/indexer FP8 → BF16 dequant'd in-artifact and re-uploaded ([5 commits on HF starting `b9dd4a18`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP/commits/main)). Loads cleanly on modern vLLM with no preprocessing. GSM8K-50 chat-mode sequential = 44/50 = 88% post-fix.
- **Card B IFEval on-disk evidence.** Re-bench on RTX PRO 6000 TP=4 produced verified JSON ([`benchmarks/rtxpro6000/ifeval_2026_05_24.json`](https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp/blob/main/benchmarks/rtxpro6000/ifeval_2026_05_24.json)): prompt-strict 0.8429 (-1.1 pp vs published B300 0.8540). Closes the no-evidence audit gap.

### Card README updates
- **Card D HF** (3c53eb3d, 8321b2e2, a05c9b6c): documents shipping fix + AIME methodology correction + RTX PRO 6000 concurrent-thinking limitation + PR #40923 application result.
- **Card D GH** (cda989dd, 00c5e602): same content + investigation-findings section.
- **Card B HF** (96f5c651, c26d63e1): cross-link to D's limitation + IFEval evidence row with link to JSON.
- **Card B GH** (afb77eed): cross-link to D's limitation.

### Repo-level hygiene
- **`requirements-bench.txt`** pushed to all 4 reproduction repos (pins `sentencepiece`, `langdetect`, `immutabledict`, `nltk`, `lm-eval`, `evalplus`, `datasets`, `openai`, `hf-transfer`). Future clean-box bench runs no longer hit the missing-dep failures we hit this session.
- **`docs/findings/sm12x_token_corruption_2026_05_24.md`** committed to `canada-quant/dsv4-flash-w4a16-fp8-mtp`: full debug log with 7+ controlled tests, sample corrupted output, working production configs.
- **Bench tooling committed**: `scripts/aime_rescore.py`, `aime_smoke.py`, `gsm8k50_sanity.py`, `d_production_verify.py`.
- **Bench JSON evidence**: `benchmarks/rtxpro6000/gsm8k50_chat_seq_dequant_20260524.json` (and IFEval JSON on Card B repo).

### Upstream contributions
- **[`jasl/vllm#12`](https://github.com/jasl/vllm/issues/12)** filed: SM 12.0 W4A16 + Marlin MoE concurrent-thinking corruption bug + full repro + cross-card NVFP4 control. Updated 2026-05-25 with PR #40923 result.
- **[`vllm-project/vllm#40923` comment](https://github.com/vllm-project/vllm/pull/40923#issuecomment-4530927937)**: canada-quant repro evidence + cross-card NVFP4-vs-W4A16 control + bench-JSON links. PR is OPEN, member-approved, blocked on core-maintainer SM120 policy review — our evidence may help unblock it.

## What was investigated and definitively isolated

The RTX PRO 6000 concurrent-thinking-mode corruption on Card D was hypothesized through 7 controlled tests:

| Hypothesis | Test | Outcome |
|---|---|---|
| sparse-MLA topk-chunk size | `VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256` | Crash earlier; not the cause |
| MTP scheduler race | Serve without `--speculative-config` | Same crash; MTP isn't the cause |
| matmul_decode kernel race | `VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0` | Same crash; not the cause |
| cuda graphs replay race | `--enforce-eager` | Survives longer (~11 min) before crashing; not the root cause |
| Concurrency sweep | c=2 / c=4 / c=8 | Corruption proportional to concurrency |
| NVFP4 vs W4A16 path | Card B (NVFP4) c=4 thinking | 1/30 corrupted vs Card D's 14/30 → **bug isolated to W4A16/Marlin path** |
| Native vs JIT-PTX Marlin | Apply [PR #40923](https://github.com/vllm-project/vllm/pull/40923) + rebuild | Corruption 14/30 → 0/30; second race surfaces as crash 29/30 |

The bug is in the **W4A16 Marlin MoE decode path on SM 12.0 under concurrent thinking-mode load**. PR #40923 fixes the JIT-PTX-fallback symptom but a second race in the same code path persists.

## Card status matrix (RTX PRO 6000)

| Card | Hardware | Single-stream thinking | Batched chat (no thinking) | Concurrent thinking (c≥2) |
|---|---|---|---|---|
| A — W4A16 (no MTP) | n/a (current vLLM | n/a (no MTP) | ✅ historical | ❌ architecture-drift KeyError on current vLLM ([details](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/issues/2)) |
| B — NVFP4-MTP | ✅ verified | ✅ verified | ✅ verified | ✅ **verified — 1/30 corrupted at c=4 thinking (essentially clean)** |
| C — Pro NVFP4-MTP | n/a (B300-only, 913 GiB) | — | — | — |
| D — W4A16-MTP | ✅ verified post-dequant | ✅ verified (GSM8K-50 = 88%) | ✅ MTP fires 92.46% | ❌ kernel race — use Card B sibling instead |

## What's still pending (for tomorrow's H200 / B300 session)

| Item | Hardware needed | Notes |
|---|---|---|
| Card A architecture-drift issue (`e_score_correction_bias` missing) | H200 with historical `jasl/vllm@ds4-sm120-experimental@abad5dc71`, or RTX PRO 6000 with a defensive `.get()` patch | Surfaced today; Card A on current vLLM hits a second shipping issue beyond the FP8 compressor dequant. Either re-calibrate or apply a vLLM loader patch |
| Card D smoke test on H200 with current vLLM + post-fix artifact | H200 (~$50, 1h) | Belt-and-suspenders verification of the lossless-math claim |
| Card D NIAH long-context smoke on patched build | RTX PRO 6000 (still up) | A's published NIAH numbers used SM 12.0 — should re-verify post-patch |
| Card B IFEval on B300 (the platform of the published 0.8540) | B300 | Confirm the -1.1pp RTX-vs-B300 delta is hardware-driven |
| Card A `e_score_correction_bias` loader patch upstream | None for filing | Investigate where vLLM's loader expects this tensor and propose a defensive change |
| Per-hardware one-shot scripts (H200 TP=2/4, RTX PRO 6000 TP=2/4) | Drafted but not validated | Validate against each platform tomorrow |

## State of the Brev box

`familiar-teal-worm` ($19.92/hr) **left running** per session instructions. Card B serve last loaded at TP=4. Patched vLLM with native SM 12.0a Marlin MoE cubins ready to use.

If you want to extend, the most useful next thing on this box is **`e_score_correction_bias` loader patch investigation for Card A** — the missing-tensor error is the only remaining shipping-class issue this session surfaced, and the box has all 4 artifacts cached.

## Out of scope (deferred to tomorrow)

- H200 verification of Card D post-format-change
- H200 smoke test of Card B
- B300 re-verification of Card C
- Any new card creation or major restructuring
