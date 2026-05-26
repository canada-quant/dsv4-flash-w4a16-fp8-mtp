# 2026-05-26 — Card B + Card A also blocked by the same upstream bug

Late-2026-05-26 finding: when re-downloaded fresh from HF and served with current upstream vLLM (jasl/vllm@a02a3778f), **Card B exhibits the same `weight_scale_inv` AttributeError → kernel-dispatch chain that Card D has been blocked on all week**. The 2026-05-25 successful Card B AIME-c=1/c=2/c=4 sweep used a now-deleted `/opt/dlami/nvme/cardb-local` artifact + an earlier venv-serve snapshot whose exact state we no longer have. Yesterday's bench numbers stand for that exact build, but they don't reproduce on a clean re-install today.

## Confirmed bug class scope

The class of bugs at [`vllm-project/vllm#43564`](https://github.com/vllm-project/vllm/issues/43564) (filed Tue) + [`vllm-project/vllm#43512`](https://github.com/vllm-project/vllm/issues/43512) (fixed by PR [`#43655`](https://github.com/vllm-project/vllm/pull/43655)) affects **all three Flash quants** on RTX PRO 6000 with the artifacts as published today:

- Card A (W4A16-FP8, no MTP) — affected: same `.weight_scale` naming + `e_score_correction_bias` drift
- Card B (NVFP4-FP8-MTP) — affected: confirmed today on fresh re-download
- Card D (W4A16-FP8-MTP) — affected: confirmed yesterday

Patches applied today (PR #43655 plumbing + defensive `getattr` on `scaled_mm/marlin.py:73` + 4 DSv4 patches via `scripts/patch_*.py`) get Card B past the load-time AttributeError but it then hits the **forward-pass kernel dispatch** that routes the FP8 block layer to a per-channel kernel and crashes with `RuntimeError: b_scales dim 1 = 32 is not size_n = 1536`.

This forward-path kernel-dispatch issue is the actual residual scope of #43564. PR #43655 closes the load-time half cleanly but **does not** fix the kernel-routing site, which appears to be a separate place that distinguishes block-quant vs per-channel by attribute name.

## What this means for the published H200 / B300 numbers

All three cards' headline benchmarks remain valid for the older `jasl/vllm` builds they were calibrated against:

- Card A → `jasl/vllm@abad5dc71` (2026-05-05)
- Card B → builds present in May-23 to May-25 jasl/vllm with the older `cardb-local` artifact format
- Card D → ditto

The mathematical content of all three artifacts is correct. The compatibility break is purely on the load + dispatch path of bleeding-edge upstream vLLM.

## What the user asked for vs what we could deliver

The user asked specifically for:

1. **Long deep-thinking bench** (max_tokens=24K / max-model-len=32K) on Card B — **could not run**. Serve doesn't start on current vLLM + freshly downloaded artifact.
2. **TP=2 vs TP=4 AIME sweep on Card B** — **could not run**. Same blocker.
3. **End-to-end test of `install_rtx6000pro.sh` on Card A** — **could not run end-to-end serve**. The script's build phase would succeed (same vLLM build as Card B); the serve phase hits the same bug.

The 2026-05-25 Card B AIME sweep numbers (c=1=24/30, c=2=23/30, c=4=21/30, MTP ~90.7%) **remain the verified baseline**, but they were measured on a venv state that we have since modified beyond easy restoration. A clean rebuild + careful state hygiene next session is needed to reproduce them.

## Recommended path forward

1. **Wait for [`#43655`](https://github.com/vllm-project/vllm/pull/43655) to merge** (closes load-time bug). Maintainer needs to add `ready` label.
2. **Wait for kernel-dispatch follow-up** — `#43564` extension flagged the residual scope; needs `@kylesayrs` or another compressed-tensors maintainer to act.
3. After both land in vllm-project/vllm → pulled into jasl/vllm: clean rebuild + re-run the three deferred tests cleanly.
4. Until then: **artifact consumers should pin to the historical jasl SHAs each card was calibrated against** rather than tracking bleeding-edge.

## What we DID accomplish today (delta vs morning)

- Filed downstream-consumer confirmation comment on [`#43655`](https://github.com/vllm-project/vllm/pull/43655#issuecomment-4545196426)
- Ran PR #43655's own test suite locally against `jasl/vllm@a02a3778f` venv — **all 5 tests pass** ([validation comment](https://github.com/vllm-project/vllm/pull/43655#issuecomment-4545305233))
- Cross-linked PR #43655 from #43564 so reviewers see them together ([cross-ref](https://github.com/vllm-project/vllm/issues/43564#issuecomment-4545200432))
- Documented the kernel-dispatch finding ([this doc + `cardd_deeper_kernel_dispatch_blocker_2026_05_26.md`])

Box: stopping. No further progress is possible client-side until upstream resolves.
