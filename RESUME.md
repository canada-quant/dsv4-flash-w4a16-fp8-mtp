# RESUME — pick up where the last session left off

**Last session ended:** 2026-05-19. Phases 0-3 ✓ shipped. Phase 4 vLLM build started; the previous session ran out of time before the build finished and Phase 5 smoke serve hasn't been validated yet.

## 30-second resume

```bash
ssh -i ~/.ssh/qwenv4-quant.pem ubuntu@35.161.108.205

# 1) Check whether the vLLM rebuild finished while you were gone
grep "Successfully installed vllm" /tmp/vllm_rebuild2.log | tail -1
# if non-empty, skip to step 3

# 2) If still running, wait; if it died, restart:
bash /tmp/build_vllm_v4.sh           # ~25 min on 32 cores

# 3) Apply the two patches to the freshly-built vllm
source /data/venv-serve/bin/activate
python -c "import vllm; print(vllm.__version__)"
VLLM=$(python -c "import vllm; print(vllm.__path__[0])")
python /data/scripts/patch_v4_forcausal_packed_mapping.py "$VLLM"
python /data/scripts/patch_mtp_packed_mapping.py "$VLLM"

# 4) Phase 5 — smoke serve on GPU pair 0,1
CUDA_VISIBLE_DEVICES=0,1 bash /data/scripts/serve_b300_tp2.sh \
    /scratch/weights/w4a16-fp8-mtp 8000 \
    > /tmp/vllm-serve.log 2>&1 &

# wait for /health 200 (5-10 min model load)
while ! curl -fsS http://localhost:8000/health; do sleep 10; done

# 5) Chat-smoke sanity
bash /data/scripts/chat_smoke.sh http://localhost:8000

# 6) Decode throughput + MTP acceptance probe
python /data/scripts/bench_decode.py http://localhost:8000 \
    --prompt-tokens 65536 --decode-tokens 2048 --requests 4
```

## What's already done

- **Phase 0** ✓ instance bootstrapped (`/data` mounted, `/scratch` symlinked, venv-calib built and patched)
- **Phase 1** ✓ 543 GB BF16 dequant preserving MTP at `/scratch/weights/bf16-mtp`
- **Phase 2** ✓ 146 GB W4A16-FP8+MTP via `model_free_ptq` (RTN) at `/scratch/weights/w4a16-fp8-mtp`
- **Phase 3** ✓ `clean_ignore_list.py` removed the pass-1↔pass-2 config overlap
- **Phase 4** ⏳ CUDA toolkit installed; vLLM build running async (or finished — check the log)

## What remains

- **Phase 4 final step**: apply the two patches once vllm imports cleanly
- **Phase 5**: serve + chat-smoke + bench_decode (all scripts ready)
- **Phase 6**: full eval suite via the external `jasl/vllm-ds4-sm120-harness` harness (not in this repo; clone separately)
- **Phase 7**: 4× TP=2 full-node deploy via `scripts/serve_b300_full_node.sh` + a basic LB (Caddy/nginx) — caller's choice
- **Phase 8**: HF release via `scripts/upload_hf.sh` — **PERMISSION-GATED**; refuses without `HF_UPLOAD_OK=1`. User must explicitly approve before publishing.

## If something is broken

- **Instance unreachable / weights gone?** `/scratch` is the instance store — wiped on stop. `/data` survives stops. Re-run Phase 1 dequant from `/data/weights/upstream` (if you still have it) or re-download from HF.
- **vLLM build keeps failing?** Read `memory:vllm_torch_abi_pin_mismatch` — the answer is "install the torch version vLLM's pyproject pins BEFORE building" + use setuptools 78-81.
- **Patches don't apply?** Check the post-refactor paths in `vllm/models/deepseek_v4/nvidia/{model,mtp}.py`. If upstream re-organized again, follow `memory:dsv4_silent_mtp_drop` to find the new homes.

## Don'ts

- Don't push to `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` (the predecessor public repo — frozen).
- Don't run `upload_hf.sh` without the user's explicit go-ahead on Phase 8.
- Don't stop the instance lightly — `/scratch/weights/w4a16-fp8-mtp` is the 7-min Phase 2 output and lives there ephemerally. If you stop, snapshot or move to `/data` first.
- AWS profile is `rozo` (NOT `default`) — see `memory:aws_profiles`.
