# CLAUDE.md — session notes for Claude Code agents

If you're a Claude Code agent resuming work in this repo, **read this first** along with [`PLAN.md`](PLAN.md). The user's persistent memory at `~/.claude/projects/-home-paul/memory/MEMORY.md` carries cross-project context; the project-specific entry there is `project_dsv4_mtp_requant.md`.

## Quick context

This repo re-quantizes DeepSeek-V4-Flash to W4A16-FP8 with the MTP layer included. The predecessor public quant shipped without MTP because `transformers` 5.8.1 silently drops `mtp.*` keys. See README.md "Status" for the per-phase status table.

**Active strategy (2026-05-20):** pivoted off B300 to H200 after multi-rank NCCL friction on Blackwell. The H200 box is the *same hardware family* the predecessor `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` was successfully calibrated on, so we run the predecessor's proven GPTQ recipe verbatim and layer in only the MTP-preservation deltas we developed on the B300 attempt (transformers regex patch, llm-compressor helpers patch, MTP shim, 7 dryrun friction fixes, decoupled MoE expert sharding). Blackwell-specific env (`TORCH_CUDA_ARCH_LIST="10.0a"`, no `expandable_segments`) is dropped — use Hopper defaults.

## Hardware + AWS

- **EC2:** `i-06a6c91366be7c18a`, `p5en.48xlarge`, `us-east-2`. 8× H200 (SM 9.0, 143,771 MiB per GPU = ~1.14 TB total HBM3e). Capacity-block lifecycle.
- **Key pair:** `h200-us-east-2`. Private key at `~/.ssh/h200-us-east-2.pem` (0600).
- **SSH:** `ssh -i ~/.ssh/h200-us-east-2.pem ubuntu@3.147.85.24`
- **Public DNS:** `ec2-3-147-85-24.us-east-2.compute.amazonaws.com`
- **DLAMI:** `ami-0bae40837d7422a24` — "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04) 20260517". Driver 595.64. `/opt/pytorch` Python 3.13 venv with torch 2.11.0+cu130. Bundled CUDA at `/opt/pytorch/cuda` is **runtime-only** — for source builds (vLLM, FlashAttention, etc.) install full CUDA: `sudo apt install cuda-toolkit-13-0`, then use `/usr/local/cuda` as `CUDA_HOME`.
- **Retired B300 box** (`i-0714f36a266c8c59b`, `p6-b300.48xlarge`, `us-west-2`, profile `rozo`): keep in mind for context only; Phase 0–1 there are not portable across regions and the B300 NCCL stack was where the multi-rank GPTQ broke.

## Disk layout (H200 box)

| Path | Persist on stop? | What |
|---|---|---|
| `/` (4.9 TB root EBS on `nvme1n1`) | yes | OS, `/opt/pytorch`, plenty of room for `venv-calib`, `venv-serve`, scripts, patches, vendor, *and* a copy of weights if needed |
| `/opt/dlami/nvme` (27.6 TB LVM RAID0 over 8× 3.5 TB NVMe) | **no** (ephemeral) | weights (upstream, bf16-mtp, w4a16-fp8-mtp), build dirs. Symlink `/scratch → /opt/dlami/nvme` for parity with B300 layout. |
| `/dev/shm` (1.0 TB tmpfs) | n/a | NCCL / dataloader shared memory (smaller than B300's 2 TB — watch for OOM if many workers) |
| `~/.ssh` | yes | persisted via root EBS |

No separate `/data` EBS this time — there's no second persistent disk by default. Put `venv-calib`/`venv-serve`/build artifacts on root (`/opt/...` or `~/`) and keep scratch weights on `/opt/dlami/nvme`.

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
- `aws --profile rozo --region us-east-2 ...` for AWS calls against the H200 box. (Profile `rozo`; region changed from `us-west-2` on the retired B300 box.)
- Commit messages: include `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` at the bottom (multiline form via heredoc, not `git commit -m`-with-flags).
- The repo is **PRIVATE** at `github.com/pasta-paul/dsv4-flash-w4a16-fp8-mtp`. Do not publish until the user explicitly authorizes Phase 8.

## What's where in the repo

```
PLAN.md                       — phase-by-phase plan, recipe topology, risk register
PHASE2_DESIGN.md              — design for GPTQ refinement path (vendor + adapt upstream)
patches/                      — diffs applied to pip-installed packages
scripts/bootstrap_p6_b300.sh  — Phase 0 for the retired B300 box (kept for reference)
scripts/bootstrap_p5en_h200.sh — Phase 0 for the active H200 box
scripts/quantize_v4_model_free.py — Phase 2 (RTN, kept as fallback)
scripts/quantize_v4_w4a16_mtp.py  — Phase 2 (GPTQ — shipping path; carries the MTP shim + 7 dryrun fixes + decoupled expert shard)
scripts/upstream/             — adapter for vendor/dsv4-upstream/model.py
vendor/dsv4-upstream/         — verbatim copies of upstream inference/{model,kernel,config}.py
```

## Resuming a session (H200 box)

1. SSH: `ssh -i ~/.ssh/h200-us-east-2.pem ubuntu@3.147.85.24`
2. Check whether bootstrap finished: `ls ~/venv-calib/bin/python` (or wherever the H200 bootstrap put it). If absent, re-run `bash ~/dsv4-flash-w4a16-fp8-mtp/scripts/bootstrap_p5en_h200.sh`.
3. Check `/scratch/weights/` — if the instance was stopped, scratch is wiped (it's the LVM-over-NVMe instance store); re-download upstream and re-run Phase 1.
4. Read `PLAN.md` for the current phase; the README's Status table is the quickest way to see what's done.

## How to talk to this codebase

When a user says "the quant" they mean this repo's output, not the predecessor's. "MTP" means the speculative-decoding head at upstream key prefix `mtp.0.*` (counter is 0-indexed within MTP, not 43 like the layer number in the architecture). "GPTQ" specifically means Hessian-based refined calibration; the current shipping path is RTN via `model_free_ptq`.
