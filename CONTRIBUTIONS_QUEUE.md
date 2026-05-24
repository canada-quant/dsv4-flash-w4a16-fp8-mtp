# Contributions queue — canada-quant

> **Design choice driving the artifact (2026-05-21):** the artifact's MTP
> draft block is preserved at **BF16** while the main 43 layers' routed
> experts go to W4A16 (FP8_BLOCK on attention). This is a deliberate
> mixed-precision recipe — speculative-decoding acceptance rate matters
> more than the ~7% size delta from preserving MTP at higher precision.
> See `model-card-draft.md` "Why MTP stays BF16" for the full rationale,
> and `FINDINGS_FOR_SIBLING.md` "Option Y — MTP stays BF16 by design"
> for the cross-pollination message to the NVFP4 sibling.



Running index of upstream issues / PRs / bug reports waiting to be filed from
diagnostic work in this repo. Add to the bottom as new findings surface.

Filing priority: **issue first**, then PR. The issue gives a public artifact
of the diagnostic work (good for canada-quant brand) and lets upstream
maintainers comment on direction before the PR lands.

## Status legend

- ⏳ identified — captured in repo, not yet filed
- 🔴 filed-issue — issue opened, waiting on maintainer feedback or our PR
- 🟡 filed-pr — PR open, in review
- ✅ merged — landed upstream

---

## Active candidates

### C10 🔴 — `compressed-tensors`: `ignore=` honored at calibration but NOT at save

- **Upstream:** `vllm-project/compressed-tensors`
- **Issue:** https://github.com/vllm-project/compressed-tensors/issues/712 (filed 2026-05-21)
- **Discovered while** verifying smoke iter 7's saved artifact. GPTQ calibration honored `ignore=re:.*mtp\..*` (subgraph 43 was empty), but the save_pretrained_wrapper's RTN-style compression re-packed MTP experts to W4A16 anyway because it only checks `targets=`, not `ignore=`.
- **Workaround in our recipe:** anchor `targets=` at `^model\.layers\.\d+\.` so MTP paths don't match in the first place. Belt-and-suspenders `ignore=` entry kept for visibility.
- **Brand-building angle:** this is the kind of mixed-precision-recipe failure mode that affects any speculative-decoding artifact (Eagle, Medusa, DSv4 MTP). Filing it makes it cheaper for the next person.

### C9 🔴 — `huggingface/transformers`: conversion_mapping doesn't cover mtp.* paths

- **Upstream:** `huggingface/transformers`
- **Issue:** https://github.com/huggingface/transformers/issues/46129
  (filed 2026-05-20)
- **Discovered while** running first end-to-end smoke after #46127's class
  shim landed. Even with the silent-drop regex removed and the MTP class
  instantiated, MTP keys arrive in upstream form
  (`mtp.0.attn.wq_a.weight`) and don't match the HF-named submodules
  (`mtp.0.self_attn.q_a_proj.weight`). Silent random-init of MTP block.
- **Companion to:** transformers#46127 (the class shim). Together they
  close the MTP-preservation gap. Either alone is insufficient.
- **Files referenced (inline-only):** patch sketch with all 33 mappings,
  runtime workaround, 50-line tensor-value verification assertion.

### C7 🟡 — `huggingface/transformers`: add `DeepseekV4NextNPredictor` class (MTP support)

- **Upstream:** `huggingface/transformers`
- **PR:** https://github.com/huggingface/transformers/pull/46127 (filed 2026-05-20)
- **Files in our repo:** `patches/transformers_dsv4_mtp.py.diff` (the PR diff),
  `scripts/transformers_mtp_shim.py` (runtime equivalent for our internal use),
  `patches/UPSTREAM_PR_DRAFTS.md` (PR plan)
- **Severity:** without this class, `from_pretrained` either drops MTP keys
  (current behavior) or loads them with no submodule to attach to (after our
  hunk 1 of modeling_deepseek_v4.py.diff). Either way, downstream calibration
  can't see the MTP block.
- **Empirical verification:** runtime shim instantiates correctly with real
  DSv4-Flash config — 6.63B params per MTP block, all expected children
  (`e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`, `self_attn`, `mlp`, etc.)
  present in `named_modules()`, 3 `hc_head_*` params registered.
- **Open work for PR:** implement `.forward()` for inference-time draft
  (shim omits — calibration only needs the weights to land), tests, MoE
  forward wiring. See `patches/UPSTREAM_PR_DRAFTS.md` § P1.
- **Filing order:** file first; P2 (llm-compressor mapping extension) depends on this.

### C8 🟡 — `vllm-project/llm-compressor`: extend ARCH_TO_2D_MAPPINGS for MTP

- **Upstream:** `vllm-project/llm-compressor`
- **PR:** https://github.com/vllm-project/llm-compressor/pull/2739
  (filed 2026-05-20; based on `kylesayrs/transformers-v5`)
- **Files in our repo:** `patches/llmc_dsv4_mtp_conversion_mappings.diff`,
  `patches/UPSTREAM_PR_DRAFTS.md`
- **Severity:** depends on C7 landing first. Once `DeepseekV4NextNPredictor`
  exists in transformers, the existing regex anchor `^layers\.` still
  excludes `mtp.*` paths from linearize_moe's walk. 3 additional
  WeightRenaming entries cover `mtp.\d+.mlp.experts.{0..255}.{w1,w2,w3}`.
- **Open work for PR:** acceptance test — load + save round-trip via
  `load_quantizable_moe()` on DSv4-Flash MTP-preserved checkpoint, assert
  `mtp.0.mlp.experts.*` keys present in output safetensors index.
- **Related:** #2735 (the issue body for these patches).

### C6 🔴 — `GPTQModifier` compress_module_list line 304 synchronous device→host stall

- **Upstream:** `vllm-project/llm-compressor`
- **Issue:** https://github.com/vllm-project/llm-compressor/issues/2736
  (filed 2026-05-20)
- **Discovered while** working on C1's patches — after C1's `_reduce_hessian_to_target_rank` skip-sharded patch landed,
  the smoke hung at `int(num_samples)` (line 304 of `gptq/base.py`) — `Tensor.item<>` on a CUDA scalar triggers
  cudaStreamSynchronize that never drains.
- **Workaround:** add `torch.cuda.synchronize() + dist.barrier()` at the end of `_reduce_hessian_to_target_rank` (in our `scripts/multirank_patches.py`).
- **Upstream fix candidate:** coerce `num_samples` to host once: `int(num_samples.detach().cpu().item()) if num_samples.is_cuda else int(num_samples)`. Saves >30000 sync points per DSv4 calibration run.
- **Tag:** `@kylesayrs`
- **Related:** #2734 (parent disjoint-set hang), #2735 (MTP drop)
- **Empirical status:** workaround landed in our smoke, retest in progress at time of filing this entry.

### C1 🔴 — `GPTQModifier` hangs on multi-rank with sharded MoE experts

- **Upstream:** `vllm-project/llm-compressor`
- **Issue:** https://github.com/vllm-project/llm-compressor/issues/2734
  (filed 2026-05-20)
- **Files in our repo:** `scripts/multirank_patches.py` (the monkey-patches),
  `scripts/quantize_v4_w4a16_mtp.py` (integration)
- **Severity:** blocks any GPTQ run on a model with module-set divergence
  across ranks (decoupled MoE expert sharding being the canonical case).
- **Empirical confirmation:** mini-GPTQ smoke on 8× H200 reached subgraph
  6/45 cleanly with patches active; `[patch B] skipped reduce for 48
  sharded modules; reducing 4 replicated` confirms the filter fires.
- **Tag:** `@kylesayrs` (mentioned in issue body)
- **Proposed PR:** introduce a per-module "replication group" attribute on
  the quantization config so `_reduce_hessian_to_target_rank` and
  `_broadcast_quantized_params` can gate the collectives on it. Our
  monkey-patches use module-name regex (`.ffn.experts.<id>.`) as a stopgap.

### C2 ⏳ — `transformers` 5.8.1 silently drops `mtp.*` keys on DSv4

- **Upstream:** `huggingface/transformers`
- **Files in our repo:** `patches/modeling_deepseek_v4.py.diff` hunk 1,
  `patches/VERSIONS.md`, `RECOVERY.md` section 2
- **Severity:** silently lossy data drop on `from_pretrained` for DSv4
  models with an MTP layer. Anyone trying to calibrate DSv4 with MTP
  preserved hits this. Predecessor's
  `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` shipped without MTP because of
  this exact bug.
- **Status:** widely known internally; not yet filed upstream.
- **Title (draft):** `[bug] DeepseekV4PreTrainedModel silently drops
  mtp.* keys via _keys_to_ignore_on_load_unexpected — should be a warning
  or removed entirely`
- **Proposed PR:** either (a) remove the regex and add a `DeepSeekV4MTP`
  module class so the MTP weights have somewhere to attach, or (b) at
  minimum emit a `logger.warning` when any matched key is dropped so users
  notice. The current behavior is the worst of both — silent data loss
  with no way for downstream code to discover what happened.

### C3 ⏳ — AWS DLAMI `ami-0bae40837d7422a24` driver/fabricmanager mismatch

- **Upstream:** AWS doesn't host a public DLAMI issue tracker —
  `awsdocs/aws-deep-learning-amis` is archived (2025-10). Practical
  venue: AWS re:Post forum or an AWS Support case for the user's account.
- **Files in our repo:** `RECOVERY.md` section 1 (full repro + fix +
  rollback procedure)
- **Severity:** CUDA Error 802 out of the box on HGX H200 instances (p5en).
  GPUs visible to `nvidia-smi` but `torch.cuda.is_available()` returns
  `False`. Multi-rank NCCL impossible until fixed.
- **Title (draft):** `DLAMI ami-0bae40837d7422a24 ships driver 595.64 +
  fabricmanager-595 (595.71.05) — CUDA Error 802 on p5en out of the box`
- **Workaround documented:** yes, `RECOVERY.md` walks the apt install +
  reboot path. ~10 min total.
- **Brand-building angle:** since there's no upstream tracker, the
  RECOVERY.md doc in this repo IS the public artifact. Anyone hitting
  the same Error 802 on the same AMI will find this via search.

### C4 🔴 — DSv4 canonical example drops MTP layer (`load_quantizable_moe`)

- **Upstream:** `vllm-project/llm-compressor`
- **Issue:** https://github.com/vllm-project/llm-compressor/issues/2735
  (filed 2026-05-20)
- **Files referenced:** `patches/modeling_deepseek_v4.py.diff` (our MTP
  retention patch), `src/llmcompressor/modeling/moe/conversion_mappings.py`
  (the `^layers\.` regex that excludes `mtp.*` paths)
- **Severity:** anyone calibrating DSv4 via `load_quantizable_moe()`
  (current canonical recipe in `kylesayrs/transformers-v5` HEAD,
  commit `8c533c21f` from 2026-05-20) ships an artifact without MTP.
  Same root cause as predecessor `canada-quant/DeepSeek-V4-Flash-W4A16-FP8`.
- **Tag:** `@kylesayrs` (mentioned in issue body; this is the active
  iteration branch for DSv4)
- **Proposed PR:** add `DeepseekV4MTP` module class to `transformers`,
  extend `ARCH_TO_2D_MAPPINGS["deepseek_v4"]` to cover `mtp.\d+.mlp.experts.*`,
  update example to verify MTP keys post-save_pretrained.
- **Brand-building angle:** filed during the week kylesayrs is iterating
  on the DSv4 fallback pathway — canada-quant contributing the MTP angle
  during active upstream development.

### C5 ⏳ — `huggingface_hub` deprecation warning: `HF_HUB_ENABLE_HF_TRANSFER`

- **Upstream:** `huggingface/huggingface_hub`
- **Observed:** during `hf download` on H200 box 2026-05-20.
- **Severity:** low — deprecation warning, replaced by
  `HF_XET_HIGH_PERFORMANCE`. Just an opportunity to update docs +
  `scripts/bootstrap_p5en_h200.sh`.
- **Action:** update our bootstrap script to use `HF_XET_HIGH_PERFORMANCE=1`
  on the next bootstrap revision; no upstream issue needed.

### C16 ⏳ — `vllm`: `nvidia/ops/attention.py:wo_a` access pattern is dynamo-unsafe

- **Upstream:** `vllm-project/vllm`
- **File:** `vllm/models/deepseek_v4/nvidia/ops/attention.py:370`
- **Discovered while** porting the W4A16+FP8+MTP artifact to RTX PRO 6000
  Blackwell (SM 12.0) — 2026-05-24, this session.
- **Problem:** the forward path reads `self.wo_a.weight_scale_inv`
  directly. PR #43290 added a `getattr(..., "weight_scale_inv", None) or
  self.wo_a.weight_scale` fallback for artifacts using the W8A8 naming.
  But that fallback is **not dynamo-safe** — dynamo's `_getattr_static`
  intercepts the attribute lookup with a class-level resolver that fails
  on dynamically-registered parameters like `weight_scale`. Triggers
  `ObservedAttributeError` at trace time and crashes cudagraph capture
  when the MTP block (BF16 wo_a, no scale attr at all) is exercised
  during `profile_run`.
- **Workaround in our repo:** `scripts/patch_wo_a_bf16_path.sh` —
  switch the conditional to use `self.wo_a.weight.dtype ==
  torch.bfloat16` (constant-foldable at trace time). Routes BF16 wo_a
  through the existing `rocm_inv_rope_einsum` path instead of the FP8
  einsum that needs scale.
- **PR candidate:** the same patch upstream would let any Option-Y MTP
  artifact (W4A16+FP8+MTP, NVFP4+FP8+MTP, future schemes) run with
  cudagraph on SM 12.0 hardware. ~5 line change.
- **Without this:** users have to set `--enforce-eager` → 10× decode
  slowdown.

### C17 ⏳ — `vllm`: SM 12.0 compressor / indexer.weights_proj hardcoded `quant_config=None`

- **Upstream:** `vllm-project/vllm`
- **Files:**
  - `vllm/models/deepseek_v4/compressor.py` (`fused_wkv_wgate`,
    `quant_config=None`)
  - `vllm/models/deepseek_v4/nvidia/ops/attention.py:weights_proj`
    (`quant_config=None`)
- **Discovered while** loading the W4A16+FP8+MTP artifact on SM 12.0
  via `jasl/vllm@ds4-sm120-preview-dev`.
- **Problem:** the SM12 branch (and apparently upstream main) constructs
  these modules with `quant_config=None` — i.e. as unquantized BF16
  modules. But artifacts calibrated with FP8_BLOCK on the attention
  path (per the predecessor's published recipe) DO quantize compressor /
  indexer.weights_proj / indexer.wq_b. Loading then hits
  `KeyError: 'layers.10.attn.mla_attn.compressor.fused_wkv_wgate.weight_scale'`
  because the module has no slot for the scale.
- **Workaround in our repo:** `scripts/dequant_compressor.py` —
  one-time preprocess to dequantize 166 weights (FP8 + BF16 scale →
  BF16 weight) and drop the orphan scales from the safetensors index.
- **PR candidate (option A):** pass `quant_config=vllm_config.quant_config`
  through to these MergedColumnParallelLinear / ReplicatedLinear
  constructors, letting compressed-tensors allocate the scale slots.
- **PR candidate (option B, deeper):** have the model class consult the
  artifact's `quantization_config.config_groups` and instantiate
  per-module accordingly.
- **Related to:** vllm-project/vllm#31085 (the SM 12.0 NVFP4 MoE
  selector also has assumptions that break for our scheme layout).

### C18 ⏳ — `jasl/vllm`: `spinloop` extension `USE_SABI 3.11` incompatible with Python 3.10

- **Upstream:** `jasl/vllm` (and possibly upstream when this code lands)
- **File:** `CMakeLists.txt` (declares spinloop extension with `USE_SABI 3.11`)
- **Discovered while** building `jasl/vllm@ds4-sm120-preview-dev` from
  source against the Brev g7e.24xlarge box's Python 3.10 (2026-05-24).
- **Problem:** `Py_LIMITED_API=0x030b0000` (Python 3.11) is set on the
  spinloop compile flags, but `Py_buffer` and `PyBuffer_Release` were
  only promoted to the stable ABI in Python 3.11. Python 3.10 headers
  don't expose them, so the build fails with `error: 'Py_buffer' was
  not declared in this scope`.
- **Workaround in our repo:** drop the `USE_SABI 3.11` line:
  `sed -i '/USE_SABI 3\.11/d' ~/src/vllm/CMakeLists.txt` (mentioned in
  `RECIPE_RTX6000PRO.md` §1).
- **PR candidate:** either bump the required Python version to 3.11+
  documented in the build instructions, OR use `USE_SABI 3.10` for the
  buffer-protocol path.

---

## Procedure when adding to this queue

1. Capture the repro/fix into a focused markdown file (or a section of an
   existing one — `RECOVERY.md` for instance-level incidents,
   `findings/` directory for protocol/algorithm bugs).
2. Commit + push immediately (per the continuous-commit standing rule).
3. Add an entry here with status `⏳ identified`.
4. When filed upstream, update status to `🔴 filed-issue` and link the
   issue URL.
5. When the PR is in review, status `🟡 filed-pr`.
6. When merged, status `✅ merged`. Keep the entry as historical record.
