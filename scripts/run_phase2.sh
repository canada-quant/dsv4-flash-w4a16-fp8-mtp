#!/usr/bin/env bash
# Phase 2 — DSv4-Flash W4A16+FP8+MTP full GPTQ calibration
# Predecessor-standard params (768 samples, batch=4, seq=512)
# Recipe matches smoke iter 8/9 (which produced 67.9% MTP acceptance after fixup)
set -euxo pipefail
source ~/venv-calib/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="9.0a"
# Predecessor pinned env (phase3b-recovery.md)
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_BLOCKING_WAIT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/dsv4-flash-w4a16-fp8-mtp

START_TS="$(date -u +%FT%TZ)"
echo "[phase2] start=$START_TS"
echo "[phase2] output=/scratch/weights/w4a16-fp8-mtp-gptq"
echo "[phase2] checkpoint-dir=/scratch/weights/checkpoints-phase2"

time torchrun --nproc-per-node=8 --master-port=29512 scripts/quantize_v4_w4a16_mtp.py \
    --input /scratch/weights/bf16-mtp \
    --output /scratch/weights/w4a16-fp8-mtp-gptq \
    --samples 768 --batch-size 4 --max-seq-len 512 \
    --offload-dir /scratch/offload \
    --checkpoint-dir /scratch/weights/checkpoints-phase2

END_TS="$(date -u +%FT%TZ)"
echo "[phase2] end=$END_TS"
echo "PHASE2_DONE"
