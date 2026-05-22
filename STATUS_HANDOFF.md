# Status handoff — autonomous run summary (2026-05-21)

User left for sleep around 06:30 UTC on 2026-05-21 authorizing autonomous
progression to Phase 2 launch. Working through 2026-05-21 ~12:30 UTC
(~6h autonomous). This is the situation when you wake up.

## ⚠️ HANDOFF INACCURACY — 2026-05-22 corrected

The pre-compaction summary in this file's prior revision claimed the iter 9
artifact at `/scratch/weights/w4a16-fp8-mtp-smoke` was **corrupted** ("4 shards
have incomplete metadata headers from interrupted rename_e_score.py").
**This was wrong.** Verified 2026-05-22 ~01:00 UTC: all 4 shards open clean
with `safetensors.safe_open`, 101,449 total keys, no truncated headers.

Built a 3h re-smoke plan partly on this false claim. Saved by running the
verify-open check first thing on resume. Lesson: pre-compaction summary
claims are NOT facts — verify before acting. Especially "X is corrupted /
broken / failed" claims; those are exactly the ones that drive expensive
recovery decisions.

The real artifact state on 2026-05-22 ~01:30 UTC was: shards intact, both
rename passes applied (797 mtp.* keys, 0 layers.43.* keys, no
e_score_correction_bias), but 103 FP32 dtypes drifted (not just 41 — also
41 `attn.compressor.ape` + 21 `attn.indexer.compressor.ape` which the original
restore predicate didn't cover post-rename), and 2 alias keys missing
(`mtp.0.head.weight` + `mtp.0.emb.tok_emb.weight`) which is why MTP returns
garbage logits and 0% acceptance (vLLM's `DeepSeekV4MTP.load_weights`
silently skips top-level head/embed for the MTP slot — sibling artifact
adds these as full FP32 duplicates of head.weight and BF16 duplicates of
embed.weight in postprocess).

## ⚠️ HARD CHECKPOINT — read this BEFORE debugging any error

**Rule (added 2026-05-21 17:50 UTC after this rule was violated 3×
across sessions):** if you hit an error on an artifact that descends
from predecessor `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` work, the
FIRST THREE ACTIONS are:

1. **Read predecessor's [README](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/blob/main/README.md) + [`patches/VERSIONS.md`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8/blob/main/patches/VERSIONS.md)** in full. The predecessor's repo documents the EXACT vLLM build pin + vendored patches that shipped a working serve. Don't skip this.

2. **Compare predecessor's build pin to ours.** Predecessor uses
`jasl/vllm@ds4-sm120-experimental` (2026-05-06, production-validated).
If we're on something else (e.g. `ds4-sm120-preview-dev`), call out the
delta explicitly.

3. **Report the delta to the user before iterating.** Don't start
patching the artifact or the build until the user confirms the path
forward.

**Why this rule exists:** on 2026-05-21 morning the H200 agent spent ~4
hours debugging serve smoke against `preview-dev` (bleeding edge) when
predecessor's documented build (`ds4-sm120-experimental`) was the
known-good base. The probe was run too late; predecessor-repo-read was
skipped three times before user prompted it. Cost: 4 hours of patches
against a broken build, one Option-Y violation that was rolled back.

The rule applies to BOTH the W4A16 repo (this one) and the NVFP4 sibling
(`canada-quant/dsv4-flash-nvfp4-fp8-mtp`) — both descend from the same
predecessor recipe topology.

## Bottom line

Phase 1 + iter 8 calibration smoke **succeeded** — first DSv4-Flash quant
artifact with **BF16 MTP preserved (Option Y verified)**. Phase 2 launch
is **NOT** kicked off because the serve smoke is **BLOCKED** on a vLLM
compressed-tensors scale-fusion issue that needs your eyes.

I made one strategic call you should sanity-check: I did NOT auto-launch
Phase 2 ($1300+, 14h) because serve smoke is unresolved, and shipping
needs serve. The artifact data is sound; the path-to-serve is what's stuck.

## What worked

1. **Smoke iter 8** completed in ~3h on H200 8-GPU (load 33min →
   calibrate 44 subgraphs × 112s → save 932s → 156 GB artifact).
   `[quant] DONE. Output at /scratch/weights/w4a16-fp8-mtp-smoke`.

2. **Option Y verification** PASSED (`scripts/verify_option_y.py`):
   - M1: 33024 main expert weight_packed (exact 43×256×3)
   - M2: 381 main attn weight_scale
   - **Y1-Y4: ALL ZERO** — MTP block has no weight_packed/weight_scale
     (the goal — true BF16 MTP achieved for the first time)
   - Y5: 768 BF16 MTP expert .weight (exact 256×3)
   - A1-A3: embed_tokens / head.weight / model.norm all present
   - config.layer_types: 43 (shim truncation worked)

3. **Filed upstream C10**:
   [`vllm-project/compressed-tensors#712`](https://github.com/vllm-project/compressed-tensors/issues/712) —
   `ignore=` honored at calibration but NOT at save. The asymmetry that
   broke iter 7; the targets=-anchor workaround in our recipe is what
   made iter 8 produce a clean BF16 MTP.

4. **FINDINGS_FOR_SIBLING.md** updated with Bug N3 (ignore=/save) and
   Bug N4 (targets= vs vLLM internal naming) for the B300 NVFP4 sibling.

## What's stuck — serve smoke

I ran 9 iterations of `scripts/option_b_serve_smoke.sh` against the iter
8 artifact. Each iteration surfaced a NEW root cause when fixed:

| # | Failure | Root cause | Fix applied |
|---|---|---|---|
| 1 | /health timeout 5min | deadline too short | extended to 15min |
| 2 | AttributeError compress_ratios | save_pretrained stripped source-only config keys | restore from source |
| 3 | KeyError scale_fmt | recipe writes ue8m0; vLLM requires it | re-added |
| 4 | OOM 4GB on 1GB free | targets= used HF names; vLLM uses upstream; fell into UnquantizedFusedMoE allocating BF16 | rewrite targets in postprocess |
| 5 | OOM (same) | input_activations missing for FP8 path | restore predecessor dict |
| 6 | KeyError hc_head.hc_base | save_pretrained nested under submodule; vLLM expects flat hc_head_base | rename in safetensors |
| 7 | KeyError layers.0.attn_hc.base | systemic transformers HF naming vs vLLM upstream naming | **comprehensive rename script** (`scripts/rename_to_upstream.py`) — 101,448 renames across 4 shards |
| 8 | KeyError compressor.indexer.kv_norm | indexer nested under compressor in transformers; predecessor has indexer.compressor and norm/ape | pass-2 rename (292 keys) |
| 9 | **KeyError fused_wkv_wgate.weight_scale** | **vLLM constructs fused linear but can't load fused weight_scale from per-shard scales** | **stuck here** |

After all 8 fixes, our artifact's layer 10 key set is **99.96%
identical to predecessor's published artifact** (`pastapaul/DeepSeek-V4-Flash-W4A16-FP8`).
Only difference: `ffn.gate.bias` vs `ffn.gate.e_score_correction_bias`
(predecessor's form). Tiny, but easy to fix as a third rename pass.

### Why I think failure #9 is a real vLLM-side block

The predecessor's published artifact has the SAME per-shard scales
(`wkv.weight_scale` + `wgate.weight_scale` separate, no fused scale
in safetensors). Predecessor serves successfully with `jasl/vllm@ds4-sm120-experimental`.

Our box has `jasl/vllm@3424fba51` + paul/dsv4 local patches setting
`packed_modules_mapping` on both `DeepseekV4ForCausalLM` (model.py) and
`DeepSeekV4MTP` (mtp.py). These patches handle weight fusion but the
weight_SCALE fusion path in compressed-tensors doesn't appear to be
firing. vLLM is looking for `fused_wkv_wgate.weight_scale` directly
instead of concatenating shards.

Possible angles you may want to check:

1. The predecessor uses `ds4-sm120-experimental` branch, not `3424fba51`.
   The experimental branch may have additional compressed-tensors patches
   for scale fusion that landed after 3424fba51 was cut.
2. There may be additional Phase 0 vLLM patches we need beyond the two
   `packed_modules_mapping` blocks.
3. The compressed-tensors loader at
   `~/src/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:743`
   passes `fused_mapping` but the weight_loader path may need separate
   fusion logic for scales.

## What I changed (committed + pushed)

All on `main` of `canada-quant/dsv4-flash-w4a16-fp8-mtp`:

- `61ee448` — targets= anchor fix (the iter 7→8 critical fix)
- `e35bd6d` — C10 + Bug N3 docs
- `96c5a64` — `verify_option_y.py`
- `319b200` — verify_option_y false-alarm fixes (Y6 threshold, head.weight)
- `8041014` — postprocess: restore source-only config keys + 15min /health
- `45e580e` — postprocess: rewrite targets= for vLLM internal naming
- `cb35043` — FINDINGS Bug N4
- `e39b703` — postprocess: keep scale_fmt=ue8m0
- `7a6938b` — postprocess: flatten hc_head submodule
- `9204e2f` — `scripts/rename_to_upstream.py` (the comprehensive rename)

Plus on the H200 box (uncommitted) `/tmp/pass2_rename.py` for the second
rename pass — should be moved into `scripts/` once you decide whether to
incorporate.

## Artifact state on H200

- `/scratch/weights/w4a16-fp8-mtp-smoke/` — 155 GB, 4 shards, BOTH rename
  passes applied. Layer 10 keys 99.96% match predecessor. Config has
  compress_ratios + scale_fmt + correct targets + restored FP8 act dict.
- `/scratch/weights/bf16-mtp/` — source, untouched.
- `/scratch/offload/`, `/scratch/weights/checkpoints-smoke/` — calibration scratch.

## Decision points for you

**A. Serve smoke debugging:** my best guess is we need additional vLLM
patches from `jasl/vllm@ds4-sm120-experimental`. I can rebase or you can.
Need ~1h to investigate the scale-fusion path.

**B. Phase 2 launch readiness:** the calibration itself is proven (iter 8
end-to-end). If you're confident Phase 2's output will face the same
rename/postprocess path, we can launch in parallel while debugging serve.
Risk: $1300+ + 14h if some unforeseen calibration-side issue emerges.
Phase 2 command staged as task #24.

**C. Defer Phase 2:** wait until serve is green on the smoke artifact,
then launch. Safest. Costs ~1 day of clock time.

My recommendation: **C** (defer until serve is green). The artifact
verification is solid but the serve path has more friction than expected
and I don't want to commit 14h of compute on faith.

## Compute clock

EC2 cost: instance has been running since the H200 pivot ~2026-05-20 ~21:00.
At $98/h × ~16h = ~$1568 burned. Phase 2 (14h) would bring total to ~$3000.
Per-second billing means stopping the instance now and resuming later is
fine — just remember `/scratch` is ephemeral (re-download from HF needed).

## Monitors armed

- (none currently — all stopped before this handoff)

## Files touched but not committed

- `/tmp/pass2_rename.py` on H200 (the second rename pass)

## Quick resume commands

```bash
# SSH
ssh -i ~/.ssh/h200-us-east-2.pem ubuntu@3.147.85.24

# verify the artifact again
python ~/dsv4-flash-w4a16-fp8-mtp/scripts/verify_option_y.py /scratch/weights/w4a16-fp8-mtp-smoke

# re-launch serve smoke
bash /tmp/relaunch_b.sh  # uses scripts/option_b_serve_smoke.sh

# read last serve log
tail -200 /tmp/vllm_optionb.log

# launch Phase 2 (if you decide to)
bash ~/dsv4-flash-w4a16-fp8-mtp/scripts/run_phase2_full.sh  # NOT created yet, see task #24
```

Welcome back.
