# Contributions queue — canada-quant

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
