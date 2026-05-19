# CLAUDE.md — session notes for Claude Code agents

If you're a Claude Code agent resuming work in this repo, **read this first** along with [`PLAN.md`](PLAN.md). The user's persistent memory at `~/.claude/projects/-home-paul/memory/MEMORY.md` carries cross-project context; the project-specific entry there is `project_dsv4_mtp_requant.md`.

## Quick context

This repo re-quantizes DeepSeek-V4-Flash to W4A16-FP8 with the MTP layer included. The predecessor public quant shipped without MTP because `transformers` 5.8.1 silently drops `mtp.*` keys. See README.md "Status" for the per-phase status table.

## Hardware + AWS

- **EC2:** `i-0714f36a266c8c59b`, `p6-b300.48xlarge`, `us-west-2`. **Profile `rozo`** (not default!). 3-day prepaid spot reservation, so don't bother stopping for cost.
- **SSH:** `ssh -i ~/.ssh/qwenv4-quant.pem ubuntu@35.161.108.205`
- **DLAMI:** Ubuntu 24.04 + `/opt/pytorch` Python 3.13 venv with torch 2.11.0+cu130. Bundled CUDA is at `/opt/pytorch/cuda` but is **runtime-only** — for source builds (vLLM, FlashAttention, etc.) install full CUDA: `sudo apt install cuda-toolkit-13-0`, then use `/usr/local/cuda` as `CUDA_HOME`.

## Disk layout

| Path | Persist on stop? | What |
|---|---|---|
| `/data` (300 GB EBS) | yes | `venv-calib`, `venv-serve`, scripts, patches, vendor |
| `/scratch` → `/opt/dlami/nvme` (27.6 TB) | **no** (ephemeral) | weights (upstream, bf16-mtp, w4a16-fp8-mtp), build dirs |
| `~/.ssh`, `/etc/fstab` | yes | persisted via root EBS |

If you need to keep weights across stops: move to `/data` or snapshot to S3.

## Critical gotchas (all in memory bank too)

1. **`mtp.*` silent drop** — transformers 5.8.1 has `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` on `DeepseekV4PreTrainedModel`. patches/modeling_deepseek_v4.py.diff hunk 1 fixes.
2. **Internal naming** — upstream HF checkpoint uses `layers.X.attn.wq_a`, NOT `model.layers.X.self_attn.q_a_proj`. Scale suffix is `.scale` in upstream, becomes `.weight_scale` in compressed-tensors output.
3. **No `--system-site-packages`** — `/opt/pytorch`'s Python 3.13 venv inheriting `/usr/lib/python3/dist-packages` (3.12-compiled wheels) crashes with `pyo3_runtime.PanicException` on import of `cryptography` via `accelerate→boto3→urllib3→pyOpenSSL`.
4. **compressed-tensors pin** — the predecessor's "phantom alpha" `0.15.1a2` shipped 2026-05-15 as `0.15.1a20260515`. llm-compressor `f2aa32e2` imports `compressed_tensors.distributed` which only exists in that alpha line.
5. **Dotless tensors crash `match_quantizable_tensors`** — `hc_head_fn`, `hc_head_base`, `hc_head_scale` have no `.` and the function's `name.rsplit('.', 1)` raises. `scripts/quantize_v4_model_free.py` carries a monkey-patch.
6. **vLLM build CUDA** — pip-wrapped cmake suppresses stderr; always run cmake manually outside pip to see the real error. The DLAMI's `/opt/pytorch/cuda` lacks `lib64/` symlink and unversioned `.so` files — use `/usr/local/cuda` after `apt install cuda-toolkit-13-0`.

## Working norms

- The user prefers terse, factual responses. Don't write trailing summaries unless asked.
- For risky/expensive actions (HF upload, force-push, instance termination), confirm before executing.
- `aws --profile rozo --region us-west-2 ...` is the right form for AWS calls against this box.
- Commit messages: include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` at the bottom (multiline form via heredoc, not `git commit -m`-with-flags).
- The repo is **PRIVATE** at `github.com/pasta-paul/dsv4-flash-w4a16-fp8-mtp`. Do not publish until the user explicitly authorizes Phase 8.

## What's where in the repo

```
PLAN.md                       — phase-by-phase plan, recipe topology, risk register
PHASE2_DESIGN.md              — design for GPTQ refinement path (vendor + adapt upstream)
patches/                      — diffs applied to pip-installed packages
scripts/bootstrap_p6_b300.sh  — idempotent Phase 0
scripts/quantize_v4_model_free.py — Phase 2 (RTN, currently ships)
scripts/quantize_v4_w4a16_mtp.py  — Phase 2 (GPTQ, scaffold — see PHASE2_DESIGN.md)
scripts/upstream/             — adapter for vendor/dsv4-upstream/model.py
vendor/dsv4-upstream/         — verbatim copies of upstream inference/{model,kernel,config}.py
```

## Resuming a session

1. SSH the box; verify `/data/venv-calib` is intact (`ls /data/venv-calib/bin/python`)
2. Check `/scratch/weights/w4a16-fp8-mtp/` — if the instance was stopped, scratch is wiped; re-run from Phase 1 (paths still on /data) or download a fresh upstream checkout
3. Read `PLAN.md` for the current phase; the README's Status table is the quickest way to see what's done

## How to talk to this codebase

When a user says "the quant" they mean this repo's output, not the predecessor's. "MTP" means the speculative-decoding head at upstream key prefix `mtp.0.*` (counter is 0-indexed within MTP, not 43 like the layer number in the architecture). "GPTQ" specifically means Hessian-based refined calibration; the current shipping path is RTN via `model_free_ptq`.
