#!/usr/bin/env bash
# Phase 8 — upload the W4A16-FP8+MTP artifact to HuggingFace.
#
# **PERMISSION-GATED.** Does nothing without HF_UPLOAD_OK=1 in the
# environment to prevent an accidental public publication. The repo is
# private until explicit user-approved release.
#
# Usage:
#   HF_UPLOAD_OK=1 HF_TOKEN=hf_xxx bash scripts/upload_hf.sh \
#       /scratch/weights/w4a16-fp8-mtp \
#       canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP

set -euo pipefail

WEIGHTS="${1:?usage: $0 <weights_dir> <hf_repo_id>}"
REPO_ID="${2:?usage: $0 <weights_dir> <hf_repo_id>}"

if [[ "${HF_UPLOAD_OK:-0}" != "1" ]]; then
    cat >&2 <<EOF
$(basename "$0"): blocked by safety guard.

This script will publish to https://huggingface.co/$REPO_ID

Re-run with HF_UPLOAD_OK=1 (and a valid HF_TOKEN) to actually publish:

    HF_UPLOAD_OK=1 HF_TOKEN=hf_xxx bash $0 $WEIGHTS $REPO_ID

A dry-run that lists what *would* be uploaded:

    DRY_RUN=1 bash $0 $WEIGHTS $REPO_ID
EOF
    exit 2
fi

if [[ ! -d "$WEIGHTS" ]]; then
    echo "FATAL: weights dir not found: $WEIGHTS" >&2
    exit 1
fi

# Show what we're about to ship
echo "=== will upload to https://huggingface.co/$REPO_ID ==="
echo "size: $(du -sh "$WEIGHTS" | awk '{print $1}')"
echo "shards: $(ls "$WEIGHTS"/*.safetensors 2>/dev/null | wc -l)"
echo "model card: scripts/model_card.md"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "(dry-run — exiting without uploading)"
    exit 0
fi

# Copy the model card README to the weights dir so HF picks it up automatically
cp "$(dirname "$0")/model_card.md" "$WEIGHTS/README.md"

# Use the `hf` CLI (huggingface_hub >= 0.27)
hf upload "$REPO_ID" "$WEIGHTS" . \
    --repo-type model \
    --commit-message "Initial release: W4A16-FP8 + MTP" \
    --include "*.safetensors" "*.json" "*.txt" "README.md"

echo "UPLOAD_DONE  https://huggingface.co/$REPO_ID"
