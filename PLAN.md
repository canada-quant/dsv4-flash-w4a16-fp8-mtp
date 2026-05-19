# PLAN — DeepSeek-V4-Flash W4A16-FP8 + MTP re-quant

> **Status snapshot (2026-05-19, updated):** Phase 0 ✓ done. Phase 1 ✓ done (BF16 dequant at `/scratch/weights/bf16-mtp`). Phase 4 ✓ done (vLLM built + patched). **Phases 2 and 3 SUPERSEDED.** The previous session's `model_free_ptq` (RTN) pivot is reversed — the resulting artifact is moved to `/scratch/weights/w4a16-fp8-mtp-rtn-fallback` and is **not the deliverable** because it does not match the predecessor `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` (which is GPTQ-calibrated). Phase 2 will be re-done via GPTQ oneshot per the original plan, using the vendored upstream adapter at `scripts/upstream/`. The concrete delta — what's already scaffolded vs what still needs to be built — is in `PHASE2_GPTQ_DELTA.md`. Phase 2 will not run until the user approves that delta.

**Goal:** Republish `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` as `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` with the MTP layer (layer 43) correctly calibrated in the same GPTQ pass — no SFT, no GRPO, no scope creep. Beat the "Acti" reference of 85.52 tok/s @ 524K on Blackwell DC.

**Hardware:** AWS `p6-b300.48xlarge` — 8× B300 (Blackwell DC, SM 10.0, 275 GB HBM3e per GPU verified, 2,200 GB total).

**AMI:** `ami-02e9fc7da15a197f9` (Ubuntu 24.04, Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11, x86_64).

**Date written:** 2026-05-19.

## Live instance (as of 2026-05-19)

- **Instance ID:** `i-0714f36a266c8c59b`
- **Region:** `us-west-2`
- **Public IPv4:** `35.161.108.205`
- **Public DNS:** `ec2-35-161-108-205.us-west-2.compute.amazonaws.com`
- **Private IPv4:** `172.31.32.120`
- **SSH:** `ssh -i ~/.ssh/qwenv4-quant.pem ubuntu@35.161.108.205` (key on basementdocker)
- **Subnet:** `subnet-2a9bf161`, VPC `vpc-ad9f63d5`
- **Verified on first boot:**
  - 8× `NVIDIA B300 SXM6 AC`, driver `595.58.03`, 275 GB HBM3e each
  - `/dev/shm` 2.0 TB tmpfs (no remount needed)
  - Instance store **27.6 TB** RAID0/LVM at `/opt/dlami/nvme` (already mounted by DLAMI — use as `/scratch`)
  - EBS: 500 GB root + 300 GB unmounted at `/dev/nvme1n1` (mount as `/data`)
  - `nvcc` not in PATH — add `/usr/local/cuda/bin` (CUDA 13.x installed)
- **IAM role:** none yet (add post-launch for S3 access to ship final artifact)

---

## What changed since the original quant (2026-05-06 → 2026-05-19)

Thirteen days of upstream churn. Three things reshape the scope:

1. **CUDA MTP inference works in vLLM main as of 2026-05-18** (PR #42930, merge SHA `67f58ce23f`). Validated on GB300 with `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`. The local `patch_mtp_mapping.py` from the reasoning-agent repo is **partially obsolete** — only the `packed_modules_mapping` gap remains.

2. **PR #43004–#43077 (2026-05-19) refactored DSv4 paths.** `vllm/model_executor/models/deepseek_v4.py` → `vllm/models/deepseek_v4/nvidia/model.py`. Old patches against the legacy path will not apply.

3. **transformers v5.8.1 has DSv4 but no MTP class.** `from_pretrained` still silently drops `mtp.*` keys via an explicit
   `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` regex on `DeepseekV4PreTrainedModel` (~line 1196). This is the smoking-gun root cause for the predecessor quant shipping without MTP — confirmed by direct inspection on 2026-05-19. The `modeling_deepseek_v4.py.diff` patch now carries two hunks: hunk 1 sets that list to `[]` (new this repo), hunk 2 is the ported calibration cache fix.

## Software stack pins

```bash
# Transformers — DSv4 landed; no MTP class. Patch still required.
transformers==5.8.1

# llm-compressor — kylesayrs branch, pin SHA from 2026-05-18
git+https://github.com/vllm-project/llm-compressor.git@f2aa32e2bde1941182d8f8a348837574969335e6

# compressed-tensors — the predecessor's "phantom alpha" 0.15.1a2 actually
# shipped on 2026-05-15 as 0.15.1a20260515. llm-compressor f2aa32e2
# unconditionally imports compressed_tensors.distributed (added in the 0.15.1
# alpha line), so 0.15.0.1 fails with ModuleNotFoundError at import time.
compressed-tensors==0.15.1a20260515

# vLLM — pin to PR #41834 head (jasl/vllm codex/ds4-sm120-min-enable)
# Includes upstream main through 2026-05-17 rebase + #42930 CUDA MTP fix + workspace prereserve
# SM12x fallbacks are inert on B300 (SM 10.0)
git+https://github.com/jasl/vllm.git@3424fba51301504262c3d8355e2560469f18c9c4

# Alternative for vLLM: upstream main at-or-after #43077 (2026-05-19 08:12 UTC)
# B300 doesn't need jasl's SM12x fallbacks; choose whichever is more stable on first boot.
```

## Local patches required (4)

| # | Patch | Source | Target | Why |
|---|---|---|---|---|
| 1 | `modeling_deepseek_v4.py.diff` | reuse from `dsv4-flash-w4a16-fp8/patches/` | transformers 5.8.1 site-packages | Upstream still drops `mtp.*` keys |
| 2 | `helpers.py.diff` | reuse from `dsv4-flash-w4a16-fp8/patches/` | llm-compressor `f2aa32e2` | MTP retention in calibration. Re-rebase against new SHA before applying. |
| 3 | `patch_v4_forcausal_packed_mapping.py` | reuse, but retarget path | `vllm/models/deepseek_v4/nvidia/model.py` | kylesayrs PR #41276 still WIP — `DeepseekV4ForCausalLM.packed_modules_mapping` undefined |
| 4 | `patch_mtp_packed_mapping.py` | new, port from reasoning-agent | `vllm/models/deepseek_v4/nvidia/...` (or wherever MTP class moved in refactor) | `DeepSeekV4MTP` class missing `packed_modules_mapping`; reconcile `.weight_scale_inv` ↔ `.weight_scale` |

Patches **no longer required** (landed upstream):
- Workspace pre-reservation — in jasl PR #41834 head and upstream main
- `kylesayrs-deepseek-ct.patch` — rebased into PR #41834

## Critical bugs to plan around

**Marlin MoE TP scale-sharding (#41511, still OPEN as of 2026-05-19):** `weight_scale` not K-sharded under TP > 2. Blocks W4A16 MoE under TP=4 / TP=8. Architecture-independent — B300 is not exempt. **Plan: 4× TP=2 instances pinned to GPU pairs {0,1}, {2,3}, {4,5}, {6,7}.** Fix locally only if upstream PR appears before serve.

**DeepGEMM E8M0 Blackwell accuracy regression (sglang #12878, vllm #37804):** historical bug, fixes claimed upstream — pin DeepGEMM to a known-good commit, validate with logprob oracle before declaring green.

## Phase-by-phase execution

### Phase 0 — Instance bring-up (1–2 h)

```bash
# After launching p6-b300.48xlarge with ami-02e9fc7da15a197f9:
nvidia-smi                              # Driver ≥580 for SM 10.0a
nvcc --version                          # CUDA 13.x
ibv_devices && fi_info -p efa           # EFA present
nvidia-smi nvlink -s                    # NVSwitch firmware ≥35.2015.4718
df -h /dev/shm                          # need ≥1.8 TiB; remount if needed:
                                        # sudo mount -o remount,size=1900G /dev/shm

# Two clean venvs — NEVER share between calibration and serving stacks
python -m venv ~/venv-calib && source ~/venv-calib/bin/activate
pip install --upgrade pip

# torch must come first so llm-compressor doesn't drag in a mismatched build
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.8.1 compressed-tensors==0.15.0.1
pip install "git+https://github.com/vllm-project/llm-compressor.git@f2aa32e2bde1941182d8f8a348837574969335e6"
pip install accelerate datasets safetensors

deactivate
python -m venv ~/venv-serve && source ~/venv-serve/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/jasl/vllm.git@3424fba51301504262c3d8355e2560469f18c9c4"
```

### Phase 1 — Dequant FP4/FP8 → BF16 (preserve MTP) (2–3 h on 8× B300)

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir ./weights/upstream

# Apply transformers patch
TRANSFORMERS_DIR=$(python -c 'import transformers; print(transformers.__path__[0])')
patch -p1 -d "$TRANSFORMERS_DIR" < patches/modeling_deepseek_v4.py.diff

# Dequant with explicit MTP-key assertion. See scripts/dequant_mtp.py.
python scripts/dequant_mtp.py \
    --input  ./weights/upstream \
    --output ./weights/bf16-mtp
```

**Verification gate:** `scripts/verify_mtp_keys.py ./weights/bf16-mtp` must report ≥1 MTP tensor. Abort if zero.

### Calibration corpus (pinned 2026-05-19 — bit-for-bit matches predecessor)

| Field | Value |
|---|---|
| Dataset | `HuggingFaceH4/ultrachat_200k` |
| Split | `train_sft` |
| Samples | **768** (rank-partitioned via `compressed_tensors.datasets.get_rank_partition`) |
| Seed | 42 |
| Max seq length | **512** tokens |
| Batch size | **4** per rank |
| Chat encoding | DSv4 manual (no Jinja template). `BOS` prefix, then `<｜User｜>...` / `<｜Assistant｜></think>...{EOS}` per message |
| Source | predecessor `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` calibration script verbatim — `/tmp/dsv4-sources/dsv4-flash-w4a16-fp8/scripts/quantize_v4_w4a16.py` lines 67-135 |
| HF dataset revision | record the commit hash returned by `load_dataset(..., revision=None)` at calibration kickoff into `findings/calibration-dataset-commit.txt` for reproducibility |

The point of pinning is so this run is comparable to the predecessor's quality bar; deviating on calibration corpus means evals diverge for non-recipe reasons.

### Launch convention (pinned)

`torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16_mtp.py ...` — the calibration script calls `compressed_tensors.distributed.init_dist()` which reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE` from torchrun's env, sets `cuda:LOCAL_RANK` as the default device, and calls `dist.init_process_group(backend="nccl", ...)`. The vendored `Transformer`'s MoE then shards `n_routed_experts=256` across `world_size=8` ranks = 32 experts per rank.

### Phase 2 — GPTQ calibration W4A16-FP8 + MTP layer 43 (8–12 h on 8× B300)

> **2026-05-19 update — architecture blocker:** transformers 5.8.1's
> `deepseek_v4/` package has **no MTP module class**, only
> `num_nextn_predict_layers: int = 1` in the config. With the load-time regex
> patched, `from_pretrained` will deserialize the 1,575 mtp.* tensors but
> they have no `nn.Module` to attach to → `model.parameters()` will not
> include them, and GPTQ won't see them. Three resolutions tracked in
> `scripts/quantize_v4_w4a16_mtp.py` docstring; the chosen approach is to
> define a `DeepSeekV4MTPLayer` shim *inside* the calibration script
> (driven by upstream `inference/model.py` as the source of truth on the
> e_proj / h_proj / hc_* / shared_head wiring) and attach it as
> `model.mtp = shim` before oneshot. **The names in the recipe regexes
> below must also be updated from the predecessor's HF-style
> (`self_attn.q_a_proj`) to DeepSeek's internal naming
> (`attn.wq_a`) — the upstream checkpoint uses internal names directly
> and `scripts/dequant_mtp.py` preserves them.

```bash
# Patch llm-compressor for MTP retention (rebase first if SHA shifted)
LLMC_DIR=$(python -c 'import llmcompressor; print(llmcompressor.__path__[0])')
patch -p1 -d "$LLMC_DIR" < patches/helpers.py.diff

# /dev/shm sized in Phase 0; B300-specific env (do NOT use H200 env block)
export TORCH_CUDA_ARCH_LIST="10.0a"            # NOT 9.0a
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=0
# DO NOT SET: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (breaks Blackwell allreduce)

torchrun --nproc-per-node 8 scripts/quantize_v4_w4a16_mtp.py \
    --samples 768 --batch-size 4 \
    --input  ./weights/bf16-mtp \
    --output ./weights/w4a16-fp8-mtp
```

**Recipe topology** (extends the original by one layer; names corrected to
DeepSeek internal convention after 2026-05-19 shard inspection — see
`memory:dsv4_naming_convention`):

- Routed experts, all 44 layers including MTP layer 43: W4A16 INT4 group=128 sym, GPTQ
  - regex: `re:.*\.ffn\.experts\.\d+\.(w1|w2|w3)$`
- Attention projections, all layers + MTP attention: FP8_BLOCK 128×128, data-free
  - regex: `re:.*\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$`
- MTP-specific FP8_BLOCK additions (verified to ship as .scale-quantized in upstream): `re:mtp\.\d+\.(e_proj|h_proj)$`
- `ignore` list (BF16 passthrough): `lm_head`, embeddings, `re:.*norm.*` (covers attn_norm, ffn_norm, enorm, hnorm, kv_norm, q_norm, shared_head.norm), `re:.*\.ffn\.gate$` (routing gate), `re:.*\.ffn\.shared_experts\..*`, `re:.*\.hc_.*` (hyper-connection params), `re:.*\.attn\.attn_sink`, `re:.*\.attn\.(compressor|indexer)\..*` (auxiliary submodules — separate calibration story)

**Verification gate:** `scripts/verify_mtp_quantized.py ./weights/w4a16-fp8-mtp` must report:
- MTP routed experts: 256 tensors with `weight_scale` (one per expert)
- MTP attention: 4 tensors with FP8_BLOCK `weight_scale` (wq_a, wkv, wq_b, wo)
- MTP `e_proj`, `h_proj`, `shared_head`, `enorm`, `hnorm`, `hc_*`, `attn_sink`: present, BF16, no scales

### Phase 3 — Fix `quantization_config.ignore` in `config.json` (1 min)

```bash
python scripts/patch_ignore_list.py --model ./weights/w4a16-fp8-mtp
```

**Bug to fix when cherry-picking from reasoning-agent:** original script line 53 has `json.dumps(config, f, indent=2)` — must be `json.dump`. Without this fix, `config.json` is never written and vLLM will fail to load MTP submodules.

### Phase 4 — Patch vLLM for compressed-tensors loading (10 min)

```bash
source ~/venv-serve/bin/activate
VLLM_DIR=$(python -c 'import vllm; print(vllm.__path__[0])')

# Patch DeepseekV4ForCausalLM (gap in kylesayrs PR #41276 — still WIP upstream)
python scripts/patch_v4_forcausal_packed_mapping.py "$VLLM_DIR"

# Patch DeepSeekV4MTP class — packed_modules_mapping + weight_scale naming
python scripts/patch_mtp_packed_mapping.py "$VLLM_DIR"

# Verify patches landed
python -c "
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
from vllm.models.deepseek_v4.nvidia.model import DeepSeekV4MTP  # path may differ post-refactor
assert hasattr(DeepseekV4ForCausalLM, 'packed_modules_mapping')
assert hasattr(DeepSeekV4MTP, 'packed_modules_mapping')
print('Patches OK')
"
```

**Path uncertainty:** the refactor stack #43004–#43077 (2026-05-19) moved files; verify final import path on the pinned vLLM commit before running.

### Phase 5 — Smoke test single instance (TP=2, 1 GPU pair) (30 min)

```bash
# B300 serve config — uses native FlashMLA / DeepGEMM, no Triton fallbacks
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_b300_tp2.sh ./weights/w4a16-fp8-mtp 8000
```

`scripts/serve_b300_tp2.sh` core flags:

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name DSV4-W4A16-FP8-MTP deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 524288 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --trust-remote-code --host 0.0.0.0 --port "$PORT"
```

**Gates:**
- `/health` returns 200 within 5 min
- `chat-smoke quick` 4/4 PASS
- MTP acceptance rate ≥75% pos-0 (Acti reference was 78–81% on GB200)

### Phase 6 — Full benchmark (4 h)

Run on harness `jasl/vllm-ds4-sm120-harness` HEAD:

| Benchmark | Setting | Pass criterion |
|---|---|---|
| chat-smoke quick / quality / coding | std | 10/10 |
| toolcall15 | 15 cases × 2 pts | ≥26/30 (match prior H200) |
| GSM8K | 8-shot strict + flexible | ≥94.5% (match prior Blackwell) |
| HumanEval pass@1 | instruct 0-shot | ≥77% (match prior RTX PRO 6000) |
| NIAH 75K → 524K | single stream | 5/5 PASS |
| Decode tok/s @ 524K | MTP=2, c=1 | **>85.52** (beat Acti) |
| MTP acceptance pos-0 | length-2 spec | ≥78% (match PR #41834 RTX PRO 6000) |

### Phase 7 — 4-instance full-node deploy (2 h)

```bash
# Pin one TP=2 instance per GPU pair; 4 instances = full p6-b300.48xlarge utilization
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_b300_tp2.sh ./weights/w4a16-fp8-mtp 8000 &
CUDA_VISIBLE_DEVICES=2,3 bash scripts/serve_b300_tp2.sh ./weights/w4a16-fp8-mtp 8001 &
CUDA_VISIBLE_DEVICES=4,5 bash scripts/serve_b300_tp2.sh ./weights/w4a16-fp8-mtp 8002 &
CUDA_VISIBLE_DEVICES=6,7 bash scripts/serve_b300_tp2.sh ./weights/w4a16-fp8-mtp 8003 &
wait
```

Front with a basic LB (Caddy round-robin or nginx least-conn) for throughput measurement.

### Phase 8 — HF release (1 h)

- Upload to `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` (new repo, leave original published one alone)
- Model card credits Acti as comparison point; explicit recipe section showing MTP layer 43 inclusion
- Provide `vllm serve` one-liner with the `--speculative-config` flag

## Risk register / first-boot validation

| Risk | Mitigation | Validate on first boot |
|---|---|---|
| Marlin MoE TP>2 bug #41511 unfixed | Use 4× TP=2 instances | Confirm TP=2 W4A16 MoE loads (smoke `vllm serve`) |
| DeepGEMM E8M0 accuracy regression | Pin DeepGEMM, run logprob oracle vs upstream DeepSeek API | Compare 50 prompt logprobs against `deepseek-ai/DeepSeek-V4-Flash` reference |
| Refactor #43004–#43077 moved paths | Verify import paths before patching | `python -c "from vllm.models.deepseek_v4.nvidia.model import ..."` |
| `num_speculative_tokens=2` unverified on W4A16 + MTP at TP=2 | Fall back to `=1` if acceptance drops | Compare acceptance rate spec_tokens=1 vs =2 |
| FlashInfer autotune (PR #42857) + W4A16 TP=2 unverified | Disable if smoke fails | Set `VLLM_FLASHINFER_AUTOTUNE=0` as fallback |
| kv_offload / OffloadingConnector bugs #42992 #43093 | Don't use kv-transfer-config flags | n/a |
| Calibration full-residency on 8× B300 = first time at this hardware | 288 GB/GPU × 8 = 2.3 TB; 284B BF16 = 568 GB so 4× headroom | nvidia-smi monitor during phase 2 |

## Out of scope (deliberate)

- SFT / distillation / GRPO RL — that's the reasoning-agent repo's scope
- NVFP4 alternative — would unlock ~3-4× MoE throughput on B300 via tcgen05 and bypass Marlin bug entirely, but breaks recipe continuity with `pastapaul/DeepSeek-V4-Flash-W4A16-FP8`. Revisit if a separate Blackwell-only SKU is wanted later.
- Multi-node / NVL72 — single 8× B300 box only
- ROCm — only AMD-relevant CUDA PRs touched (#41812, #41946, #42810); not deploying to AMD

## Repo structure (proposed)

```
dsv4-flash-w4a16-fp8-mtp/
├── PLAN.md                              # this file
├── README.md                            # quick-start + model card pointer
├── patches/
│   ├── modeling_deepseek_v4.py.diff     # transformers MTP retention (port from original repo)
│   ├── helpers.py.diff                  # llm-compressor MTP retention (rebase to f2aa32e2)
│   └── VERSIONS.md                      # which upstream SHAs each patch targets
├── scripts/
│   ├── bootstrap_p6_b300.sh             # zero-to-serving on AWS p6-b300.48xlarge
│   ├── dequant_mtp.py                   # FP4/FP8 → BF16 preserving MTP
│   ├── verify_mtp_keys.py               # gate before calibration
│   ├── quantize_v4_w4a16_mtp.py         # GPTQ recipe, MTP layer 43 included
│   ├── verify_mtp_quantized.py          # gate after calibration
│   ├── patch_ignore_list.py             # fix config.json (with json.dump bugfix)
│   ├── patch_v4_forcausal_packed_mapping.py  # retarget to refactored vllm path
│   ├── patch_mtp_packed_mapping.py      # new — DeepSeekV4MTP class
│   └── serve_b300_tp2.sh                # canonical serve, MTP spec=2
├── eval/
│   └── run_harness.sh                   # jasl harness invocation
└── findings/
    └── (mission report fills in here as phases complete)
```

## Open questions for user before kickoff

1. **vLLM base:** jasl PR #41834 head (includes SM12x fallbacks, inert on B300) vs upstream main at #43077? Both valid. Jasl is more battle-tested with your prior recipe; upstream is leaner. Default to jasl unless you want fewer moving parts.
2. **num_speculative_tokens:** start at 2 (B300 native) or 1 (your H200 baseline)? Recommend 2 with fall-back validation.
3. **NVFP4 sidecar:** want a v2 entry in the plan for an NVFP4-FP8 + MTP variant on B300 as a follow-on, or strictly W4A16 only?
4. **HF repo name:** `pastapaul/DeepSeek-V4-Flash-W4A16-FP8-MTP` or different suffix?
5. **AWS region / spot vs on-demand for p6-b300.48xlarge** — informs the bootstrap script's region default.
