#!/usr/bin/env bash
# Push the local README.md to the HuggingFace model card at
# canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP.
#
# This is a permission-gated operation — explicit HF_TOKEN env var
# required (don't hard-code or read from disk).
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/push_hf_modelcard.sh
#
# Optional dry-run:
#   DRY_RUN=1 HF_TOKEN=hf_xxx bash scripts/push_hf_modelcard.sh
#
# What this does:
#   - Uploads README.md only (NOT the artifact weights — those were
#     pushed previously via upload_hf.sh).
#   - Operation is metadata-only; takes <5 seconds.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="$REPO_ROOT/README.md"
HF_REPO="canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP"

if [[ ! -f "$README" ]]; then
    echo "FATAL: $README not found" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    cat >&2 <<'EOF'
push_hf_modelcard.sh: no HF_TOKEN in env.

This script publishes a model-card update at
https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP

Re-run with a write-scoped token:

    HF_TOKEN=hf_xxx bash scripts/push_hf_modelcard.sh

Or dry-run (shows what would be uploaded):

    DRY_RUN=1 HF_TOKEN=hf_xxx bash scripts/push_hf_modelcard.sh
EOF
    exit 2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] would upload:"
    echo "  source:  $README ($(wc -c < "$README") bytes)"
    echo "  target:  $HF_REPO/README.md"
    echo "[dry-run] no upload performed"
    exit 0
fi

python3 - <<EOF
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
result = api.upload_file(
    path_or_fileobj="$README",
    path_in_repo="README.md",
    repo_id="$HF_REPO",
    repo_type="model",
    commit_message="Update model card: RTX PRO 6000 Blackwell hardware demonstration + cudagraph benchmarks",
)
print(f"Pushed: {result}")
EOF
