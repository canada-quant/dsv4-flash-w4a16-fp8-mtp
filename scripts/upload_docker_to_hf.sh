#!/usr/bin/env bash
# upload_docker_to_hf.sh — push the W4A16 Docker image tarball + README to
# canada-quant/dsv4-flash-w4a16-rtxpro6000-image on HF.
#
# Run on the box where the tarball was built:
#   HF_TOKEN=hf_... bash upload_docker_to_hf.sh \
#       /opt/dlami/nvme/dsv4-w4a16-rtxpro6000-v1.tar.gz \
#       /path/to/HF_DATASET_README.md

set -euo pipefail

TARBALL="${1:?usage: $0 <tarball_path> [readme_path]}"
README="${2:-}"
REPO="canada-quant/dsv4-flash-w4a16-rtxpro6000-image"

: "${HF_TOKEN:?HF_TOKEN must be set}"
mkdir -p ~/.cache/huggingface
echo "$HF_TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token

if ! command -v hf >/dev/null 2>&1; then
    pip install --user --quiet "huggingface_hub>=1.16"
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "[upload] repo:    $REPO"
echo "[upload] tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1))"

# Upload tarball (use xet for fast transfer)
HF_HUB_ENABLE_HF_TRANSFER=1 \
hf upload --repo-type dataset \
    "$REPO" \
    "$TARBALL" "$(basename $TARBALL)" \
    --commit-message "Add W4A16 RTX PRO 6000 Docker image v1 ($(date -u +%Y-%m-%d))"

# Upload README if provided
if [ -n "$README" ] && [ -f "$README" ]; then
    hf upload --repo-type dataset \
        "$REPO" \
        "$README" README.md \
        --commit-message "Add dataset README"
fi

echo "[upload] DONE — https://huggingface.co/datasets/$REPO"
