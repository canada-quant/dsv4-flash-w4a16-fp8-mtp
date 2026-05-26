# Session summary — 2026-05-26 (RTX PRO 6000 leg, day 3)

## Goal

Continue cleanup + Card D + Card A revisit per the previous-night plan:
1. Card D safetensors `.weight_scale` rename + verification of Marlin patches
2. Card A revisit with same testing rigor as B / D
3. One-shot `install_rtx6000pro.sh` for all 3 cards
4. Documentation cleanup HF + GH

## What actually happened

### ✅ Card B finalized — all updates landed

- HF README: AIME concurrency sweep section (already there from yesterday) + new `vllm-patches/` section
- GH README: `vllm-patches/` section + Repo-layout entry
- **New** `scripts/install_rtx6000pro.sh` — canonical one-shot installer for Card B with verified flags + spinloop-crash tolerance + manual `_moe_C.abi3.so` copy. AIME-2024 thinking-mode bench at c=1/2/4 (from 2026-05-25 session) referenced inline.

### ✅ Patch inventory `vllm-patches/` referenced from all 3 cards

The directory was committed yesterday but no card README mentioned it. Now Card A, Card B, Card D all have a `vllm-patches/` section + their READMEs cross-link to the same 3 patches.

### ✅ Card D safetensors rewrite tooling shipped

- `scripts/rename_weight_scale_to_inv.py` — idempotent header-only rewrite. Rewrites JSON metadata only (no data movement), so a 159 GB shard rewrites in ~3-5 min on local NVMe.
- `scripts/fix_config_for_dequant.py` — config.json fix for the dequant'd compressor/indexer state.

Both scripts work correctly (verified end-to-end). But **the rename direction was wrong** — see below.

### ❌ Card D verification — blocked by a deeper kernel-dispatch bug

Tried both unblock paths from yesterday's plan:

1. **Attempt 1: Safetensors `.weight_scale` → `.weight_scale_inv` rename**. 215 keys rewritten in 21 min. Result: load failed earlier with `KeyError: 'layers.0.attn.fused_wqa_wkv.weight_scale_inv'`. The model's fused `MergedColumnParallelLinear` registers `weight_scale` (no `_inv`) in `params_dict`, so the renamed safetensors keys couldn't be matched. **Rename direction was wrong** — the artifact's `.weight_scale` naming matches the model's params_dict.

2. **Attempt 2: Defensive `getattr` patch on `scaled_mm/marlin.py:71-77`**. Model loaded past `process_weights_after_loading` for the first time. But forward-pass kernel crashed:

```
RuntimeError: b_scales dim 1 = 32 is not size_n = 1536
```

The kernel expects per-channel scales of shape `(1536,)` but receives block-shape `(4, 32)`. **There's a third issue beyond config.json + safetensors naming**: some forward-path kernel-dispatch site distinguishes block-quant vs per-channel by attribute name, and routes the FP8 block layer to the wrong kernel.

Documented in [`cardd_deeper_kernel_dispatch_blocker_2026_05_26.md`](cardd_deeper_kernel_dispatch_blocker_2026_05_26.md). Extended upstream issue [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564) with the new finding, tagged `@kylesayrs`.

### ❌ Card A revisit — blocked by the same deeper bug

Subagent audit confirmed Card A has **identical** `.weight_scale` naming to Card D (33,405 keys vs Card D's 33,239) plus an additional architecture-drift bug (`ffn.gate.e_score_correction_bias` should be renamed to `ffn.gate.bias`). Both fixes prepared in `scripts/rename_weight_scale_to_inv.py --also-e-score-bias` but **not pushed** because the deeper kernel-dispatch issue would still block runtime even after the rename. Card A's H200 / DGX Spark / RTX PRO 6000 benchmarks remain valid for `jasl/vllm@abad5dc71` — they don't reproduce on bleeding-edge vLLM without upstream fixes.

### ✅ `install_rtx6000pro.sh` shipped to all 3 repos

- Card B (canonical): full implementation with verified setup, build-crash tolerance, manual .so copy, runtime deps, download artifact, summary banner.
- Card D + Card A: thin wrappers that delegate to Card B's canonical script with the right `ARTIFACT=` and a prominent "current shipping state" warning.

## Status of the Marlin c_tmp + workspace 4× patches

**Built and ready** in `_moe_C.abi3.so` (181,697,240 bytes, +1,784 vs baseline). Patches compile cleanly against `jasl/vllm@a02a3778f + #40923`. **Cannot be functionally verified** on this artifact until the upstream kernel-dispatch bug (`vllm-project/vllm#43564`) is resolved. Verification still pending for next session — see the next-session plan below.

## Upstream contributions filed today

- [`vllm-project/vllm#43564 comment`](https://github.com/vllm-project/vllm/issues/43564#issuecomment-4545101641) — extended the weight_scale-naming issue with today's deeper kernel-dispatch finding. Tagged @kylesayrs (compressed-tensors maintainer) with the kernel-routing question.

## Net delta vs previous session

| Card | HF README | GH README | install_rtx6000pro.sh | Runs on current vLLM |
|---|---|---|---|---|
| A | + 2026-05-26 update note, + vllm-patches/ section | + 2026-05-26 update, + vllm-patches/, + install_rtx6000pro.sh | ✅ wrapper | ❌ (blocked on #43564) |
| B | + vllm-patches/ section | + vllm-patches/ section, + repo-layout entry, + install_rtx6000pro.sh | ✅ canonical | ✅ verified |
| D | + vllm-patches/ section, + findings doc links | + vllm-patches/, + new findings doc, + install_rtx6000pro.sh, + rename_weight_scale_to_inv.py + fix_config_for_dequant.py | ✅ wrapper | ❌ (blocked on #43564) |

## Next-session plan (when upstream #43564 is resolved)

1. Pull latest vLLM with the kernel-dispatch fix.
2. Run AIME-30 c=1/c=2/c=4 thinking-mode sweep on Card D — this validates the Marlin c_tmp + workspace 4× patches.
3. If Card A needs the same `.weight_scale` rename + `e_score_correction_bias` rename: apply `scripts/rename_weight_scale_to_inv.py --also-e-score-bias` and re-upload Card A's safetensors. Smoke test + AIME sweep on Card A.
4. Update HF READMEs with the new bench numbers.
5. Stop the Brev box.

## Brev box state

`familiar-teal-worm` (~$20/hr) ready to stop after today's writes finish. ~$80 spent today across the rename attempts + 3-session investigation. Everything achievable without an upstream vLLM fix has now been done.
