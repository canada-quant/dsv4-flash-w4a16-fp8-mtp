# RESUME — pick up where the last session left off

**Last action:** 2026-05-20. Pivoted hardware from `p6-b300.48xlarge` (us-west-2) to `p5en.48xlarge` (us-east-2) after multi-rank NCCL friction on Blackwell. The H200 box matches the predecessor's calibration hardware, so the plan is to run the predecessor's GPTQ recipe + the MTP-preservation deltas we already developed.

## 30-second resume

```bash
# 1) SSH to the active H200 box
ssh -i ~/.ssh/h200-us-east-2.pem ubuntu@3.147.85.24

# 2) If first time on this box: clone repo + bootstrap (idempotent, ~30 min including vLLM build)
git clone git@github.com:pasta-paul/dsv4-flash-w4a16-fp8-mtp.git ~/dsv4-flash-w4a16-fp8-mtp
bash ~/dsv4-flash-w4a16-fp8-mtp/scripts/bootstrap_p5en_h200.sh

# 3) Phase 1 — download upstream and dequant (~10 min IO-bound)
source ~/venv-calib/bin/activate
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir /scratch/weights/upstream
python ~/dsv4-flash-w4a16-fp8-mtp/scripts/dequant_mtp.py \
    --input /scratch/weights/upstream \
    --output /scratch/weights/bf16-mtp
python ~/dsv4-flash-w4a16-fp8-mtp/scripts/verify_mtp_keys.py /scratch/weights/bf16-mtp

# 4) Pre-launch sanity — N-rank load test (catches the OOM the dryrun couldn't predict)
torchrun --nproc-per-node 8 ~/dsv4-flash-w4a16-fp8-mtp/scripts/loadtest_sharded.py \
    --input /scratch/weights/bf16-mtp

# 5) Phase 2 — GPTQ calibration (8× H200, ~10–14 h)
export TORCH_CUDA_ARCH_LIST="9.0a"
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node 8 ~/dsv4-flash-w4a16-fp8-mtp/scripts/quantize_v4_w4a16_mtp.py \
    --samples 768 --batch-size 4 \
    --input /scratch/weights/bf16-mtp \
    --output /scratch/weights/w4a16-fp8-mtp

# 6) Phase 3 — post-process config + verify
python ~/dsv4-flash-w4a16-fp8-mtp/scripts/postprocess_for_vllm.py /scratch/weights/w4a16-fp8-mtp
python ~/dsv4-flash-w4a16-fp8-mtp/scripts/verify_mtp_quantized.py /scratch/weights/w4a16-fp8-mtp

# 7) Phase 5 — smoke serve on GPU pair 0,1
CUDA_VISIBLE_DEVICES=0,1 bash ~/dsv4-flash-w4a16-fp8-mtp/scripts/serve_h200_tp2.sh \
    /scratch/weights/w4a16-fp8-mtp 8000 > /tmp/vllm-serve.log 2>&1 &
while ! curl -fsS http://localhost:8000/health; do sleep 10; done
bash ~/dsv4-flash-w4a16-fp8-mtp/scripts/chat_smoke.sh http://localhost:8000
```

## Active vs retired hardware

| | Active (use this) | Retired (reference only) |
|---|---|---|
| Instance ID | `i-06a6c91366be7c18a` | `i-0714f36a266c8c59b` |
| Type | `p5en.48xlarge` (8× H200, SM 9.0) | `p6-b300.48xlarge` (8× B300, SM 10.0) |
| Region | `us-east-2` | `us-west-2` |
| Public IPv4 | `3.147.85.24` | `35.161.108.205` |
| SSH key | `~/.ssh/h200-us-east-2.pem` | `~/.ssh/qwenv4-quant.pem` |
| HBM per GPU | 144 GB | 275 GB |
| AMI | `ami-0bae40837d7422a24` | `ami-02e9fc7da15a197f9` |
| `TORCH_CUDA_ARCH_LIST` | `9.0a` | `10.0a` |
| `expandable_segments` | OK | **breaks Blackwell allreduce** |

The B300 box's `/scratch/weights/*` is gone (instance store, also wiped by region change). Don't try to copy state — re-run from Phase 1 on H200.

## What carries over from the B300 work

These are the deltas we developed on B300 and still apply on H200:

- `patches/modeling_deepseek_v4.py.diff` — neutralizes the `mtp.*` ignore regex in transformers 5.8.1 (hunk 1) + calibration cache fix (hunk 2)
- `patches/helpers.py.diff` — llm-compressor MTP retention against `f2aa32e2`
- `scripts/quantize_v4_w4a16_mtp.py` — carries the MTP shim, the 7 dryrun friction fixes (see `memory:gptq_dryrun_friction`), and the decoupled MoE expert shard (`_expert_world_size=N`)
- `scripts/loadtest_sharded.py` — N-rank pre-flight to avoid the per-rank-RAM blindspot (`memory:dryrun_projection_blindspot`)
- `scripts/patch_v4_forcausal_packed_mapping.py` + `scripts/patch_mtp_packed_mapping.py` — vLLM `packed_modules_mapping` patches

## What does NOT carry over

- `scripts/bootstrap_p6_b300.sh` — replaced by `scripts/bootstrap_p5en_h200.sh` (no separate `/data` EBS, different SM arch, drops B300 env quirks)
- `scripts/serve_b300_tp2.sh` / `scripts/serve_b300_full_node.sh` — H200 equivalents `serve_h200_tp2.sh` / `serve_h200_full_node.sh` (TORCH_CUDA_ARCH_LIST=9.0a, no Blackwell-only Triton flags)
- `model_free_ptq` RTN fallback path — kept in tree as `scripts/quantize_v4_model_free.py`, but the deliverable is the GPTQ artifact since that's what matches the predecessor's quality bar

## If something is broken

- **Instance unreachable / weights gone?** `/opt/dlami/nvme` (aliased as `/scratch`) is the instance store — wiped on stop. Root EBS (4.9 TB) survives stops. Re-download from HF.
- **vLLM build keeps failing?** Read `memory:vllm_torch_abi_pin_mismatch` — answer is "install the torch version vLLM's pyproject pins BEFORE building" + use setuptools 78-81. Bootstrap script handles this; if it fails check `/tmp/vllm_build.log`.
- **Patches don't apply?** Check the post-refactor paths in `vllm/models/deepseek_v4/nvidia/{model,mtp}.py`. If upstream re-organized again, follow `memory:dsv4_silent_mtp_drop` to find the new homes.
- **Multi-rank OOMs at load?** Decoupled expert shard not active — confirm `_expert_world_size=N` is set in `scripts/quantize_v4_w4a16_mtp.py` and that `loadtest_sharded.py` was actually run before launch.

## Don'ts

- Don't push to `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` (the predecessor public repo — frozen).
- Don't run `upload_hf.sh` without the user's explicit go-ahead on Phase 8.
- Don't stop the instance lightly — `/scratch/weights/*` is ephemeral. If you stop, snapshot to root EBS or S3 first.
- AWS profile is `rozo` (NOT `default`) — see `memory:aws_profiles`.
- Don't import B300-specific env blocks on H200 (`TORCH_CUDA_ARCH_LIST="10.0a"`, the `expandable_segments` prohibition, the `VLLM_TRITON_MLA_SPARSE*` SM12x fallbacks — none apply).
