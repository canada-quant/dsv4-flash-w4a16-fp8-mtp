# SUPERVISION_RULES.md — standing discipline for Claude Code agents on this repo

These are rules that emerged from failures during the W4A16-FP8-MTP work
where the agent (main, subagent, or both) lost discipline in ways that
cost the user hours of unnecessary work. They are timeless, not
session-specific. Read this on every resume, alongside `CLAUDE.md` and
`STATUS_HANDOFF.md`.

---

## 1. Subagent briefing must carry the standing rules

**Rule:** when spawning a subagent (Agent tool), the prompt MUST include
the predecessor-repo-read rule from `STATUS_HANDOFF.md` if the subagent
is going to debug an artifact, build, or any error descended from the
predecessor recipe. Otherwise the subagent will diagnose against
incomplete priors and you'll inherit their incomplete answer.

**Concretely:** every subagent prompt about an error involving the
artifact, the calibration recipe, or vLLM build SHOULD start with:

> Before debugging, read predecessor's README + patches/VERSIONS.md at
> `github.com/canada-quant/dsv4-flash-w4a16-fp8` and the sibling
> `github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp` for the documented
> working build pin and vendored patches. Compare to the current state
> and call out the delta before proposing a fix.

**Why this rule exists:** on 2026-05-21, two subagents (A, B) were
dispatched to debug 0% MTP acceptance. Neither was briefed with the
predecessor-repo-read rule. Both diagnosed productively, but A's
postprocess didn't address the alias-key gap that B found — because A
wasn't told to compare against the working sibling. The findings had to
be merged by the main agent and one was nearly missed (the FP32 head
upcast in sibling).

**How to apply:** put the predecessor-repo-read directive in the FIRST
paragraph of any subagent prompt about a debug task.

---

## 2. FINDINGS_FOR_SIBLING.md is journaled, not held in conversation

**Rule:** any vLLM upstream bug, transformers upstream bug, or
calibration-recipe gotcha that surfaces during this work — whether
discovered by the main agent or a subagent — MUST be appended to
`FINDINGS_FOR_SIBLING.md` in the same response that surfaces it.
Treating subagent findings as conversation context is a leak path: they
disappear at the next compaction.

**Concretely:**
- Subagent reports a bug? Add to FINDINGS_FOR_SIBLING.md before
  declaring "done."
- Main agent stumbles on a bug while debugging? Same rule.
- Each finding gets a code (C13, C14, ...), a one-paragraph summary, a
  reproduction or evidence link, and the workaround status (filed
  upstream? carried as patch? not yet filed?).

**Why this rule exists:** on 2026-05-21, two real upstream bugs were
identified during debugging and never made it into
FINDINGS_FOR_SIBLING.md:
- **C13:** `transformers.save_pretrained` silently downcasts FP32 tensors
  to BF16 during shard save (the `hc_*`, `attn_sink`, `ffn.gate.bias`,
  and `*.compressor.ape` keys all suffer this). The DeepSeek release
  spec keeps these FP32 for numerical precision.
- **C14:** `vllm.model_executor.models.deepseek_v4_mtp.DeepSeekV4MTP.
  load_weights` silently skips top-level `head.weight` and `embed.weight`
  for the MTP slot (its `name.replace("mtp.0.", ...)` no-ops on
  non-mtp-prefixed keys, then `get_spec_layer_idx` returns None →
  `continue`). Result: MTP's `shared_head.head` (ParallelLMHead) and
  `embed_tokens` (VocabParallelEmbedding) stay uninitialized → garbage
  logits → 100% rejection.

Both are filable upstream. Both should have been in FINDINGS the moment
they were identified. Adding them now (2026-05-22) is recovery, not
the discipline this rule prescribes.

**How to apply:** make FINDINGS_FOR_SIBLING.md the second file you touch
after any debugging session that uncovers an upstream defect (first
file: whatever you're actually fixing).

---

## 3. Read full subagent output before acting on its summary

**Rule:** when consuming a subagent's report, read the full output the
subagent produced, not just the summary you wrote up for the user.
Synthesizing a multi-page subagent finding into a single conversational
paragraph loses information; that paragraph then becomes the basis for
the next decision, with no way to recover the lost detail later.

**Concretely:**
- If a subagent's report exceeds ~200 lines, scroll it.
- If a claim in the summary contradicts something you observed yourself,
  trust the observation over the claim — and re-read the subagent's
  output to find where the claim came from.
- Don't propagate phrases like "artifact corrupted" without grepping the
  subagent's actual evidence for it.

**Why this rule exists:** on 2026-05-21, the pre-compaction summary
claimed "iter 9 artifact corrupted (4 shards have incomplete metadata
headers from interrupted rename_e_score.py)." On resume, the very
first verification check showed all 4 shards open clean with 101,449
keys. The "corrupted" claim was a telephone-game carry from upstream of
the subagents themselves, repeated through compaction without challenge.
A 3h re-smoke plan ($300) was drafted partly on the basis of this
false claim. Saved only because the resume agent verified-before-acting.

**How to apply:** the first thing you do on resume from a compacted
session is verify the headline state claims (is the artifact actually
broken? is the test actually failing? did the file actually get
written?). Run the cheap check before drafting the expensive plan.

---

## 4. Verify pre-compaction summary claims before acting on them

**Rule:** pre-compaction summary claims are not facts. They are notes
written by an agent that no longer exists, possibly summarized through
multiple layers. Treat them as a HYPOTHESIS that needs ONE verification
step before being load-bearing in your next decision.

**Concretely:**
- "X is corrupted" → `safe_open()` it; check key counts.
- "X failed" → grep the log; check the exit code.
- "X was applied" → spot-check the diff; verify the artifact state.
- "X works" → run the smoke test; don't trust the prior claim of
  greenness.

These checks cost minutes. The wrong-decision cost they prevent is
typically hours.

**Why this rule exists:** see rule 3 above. The cost of the false
corruption claim was a near-miss on a 3h re-smoke. The cost of
verifying took 30 seconds.

**How to apply:** for each load-bearing claim in the resume summary,
ask: "what's the one-command check that would falsify this?" Run it
first. Adjust your plan based on the observation.

---

## 5. Atomic safetensors writes — no exceptions

**Rule:** any script that writes a `.safetensors` file MUST:
- Read all tensors into memory in one pass
- Apply all transforms in memory
- `save_file(..., tmp_path)` to a `.tmp` sibling
- `os.replace(tmp_path, final_path)` for atomic rename

**Never:**
- Run safetensors writes via `timeout N python ...` foreground wrapping
- Run safetensors writes inside a shell that you might Ctrl-C
- Chain follow-up scripts that each rewrite the same shard

**Why this rule exists:** during the 2026-05-21 W4A16 debug cycle, two
separate "fix one thing" follow-up scripts (one for dtype, one for the
e_score_correction_bias rename) were run via `timeout 30s`. Both were
killed mid-write, leaving partial files. The artifact was then
*believed* to be corrupted (it turned out to be intact on resume, but
the belief drove planning anyway — see rule 4). The right pattern is in
`/tmp/fixup_artifact.py` on H200: ONE script, ALL the transforms,
atomic per-shard.

**How to apply:** if you find yourself writing a "small fixup script"
that touches a `.safetensors` file, stop. Roll the fix into the
existing atomic-per-shard script. Or write a new atomic-per-shard
script that does ALL the work in one pass. Never write a third
follow-up.

---

## Index of failures these rules prevent

| Date | Failure | Rule that would have prevented it |
|---|---|---|
| 2026-05-21 morning | Option Y violation (MTP attn BF16→FP8 without checking) | rule 4 (verify before patching) + rule 1 (predecessor-repo-read) |
| 2026-05-21 morning | 4h debug against wrong vLLM build (preview-dev vs experimental) | rule 1 (predecessor-repo-read) |
| 2026-05-21 midday | Artifact "corrupted" by interrupted timeout scripts (twice) | rule 5 (atomic writes) |
| 2026-05-22 resume | 3h re-smoke planned on false corruption claim | rule 3 + rule 4 (verify summary claims) |
| 2026-05-22 fixup | Subagents A/B not briefed on predecessor-repo rule | rule 1 (subagent briefing) |
| 2026-05-22 fixup | C13/C14 nearly lost when subagents returned | rule 2 (journal findings) |

This file grows as new rules emerge. Don't delete rules; supersede with
a note if they need to evolve.
