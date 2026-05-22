#!/usr/bin/env bash
# Phase 2 postprocess pipeline — atomic per-step, verified between steps.
set -euxo pipefail
source ~/venv-calib/bin/activate
cd ~/dsv4-flash-w4a16-fp8-mtp

ART=/scratch/weights/w4a16-fp8-mtp-gptq

echo "[postproc] start=$(date -u +%FT%TZ) artifact=$ART"

# === Step 1: rename all keys HF -> upstream form ===
echo "[postproc] step 1: rename_to_upstream"
python scripts/rename_to_upstream.py "$ART"
python -c "
import json
wm = json.load(open('$ART/model.safetensors.index.json'))['weight_map']
mtp = [k for k in wm if k.startswith('mtp.')]
layers43 = [k for k in wm if k.startswith('layers.43.') or k.startswith('model.layers.43.')]
print(f'  after rename: {len(wm)} keys, {len(mtp)} mtp.*, {len(layers43)} layers.43.*')
samp = [k for k in wm if 'model.' in k][:5]
if samp: print(f'  REMAINING model. prefix: {samp}')
"

# === Step 2: config.json + MTP embed rename for vLLM ===
echo "[postproc] step 2: postprocess_for_vllm"
python scripts/postprocess_for_vllm.py --artifact "$ART" --bf16-source /scratch/weights/bf16-mtp

# === Step 3: FP32 restore + MTP head/embed aliases + head FP32 upcast ===
echo "[postproc] step 3: fixup_artifact (FP32 + aliases)"
# Adapt fixup_artifact.py to point at the gptq path. The original was hardcoded
# to w4a16-fp8-mtp-smoke; we use sed to make a copy with the new path.
sed 's|w4a16-fp8-mtp-smoke|w4a16-fp8-mtp-gptq|g' /tmp/fixup_artifact.py > /tmp/fixup_artifact_phase2.py
python /tmp/fixup_artifact_phase2.py

# === Step 4: final verification ===
echo "[postproc] step 4: verify"
python -c "
import json
from pathlib import Path
from safetensors import safe_open
import torch

ART = Path('$ART')
wm = json.load(open(ART / 'model.safetensors.index.json'))['weight_map']
print(f'  total keys: {len(wm)}')
mtp = [k for k in wm if k.startswith('mtp.')]
print(f'  mtp.* keys: {len(mtp)} (expected 799)')
for k in ('head.weight', 'embed.weight', 'mtp.0.head.weight', 'mtp.0.emb.tok_emb.weight'):
    print(f'  {k}: {k in wm}')

# Check head dtype
with safe_open(ART / wm['head.weight'], framework='pt') as f:
    t = f.get_tensor('head.weight')
print(f'  head.weight dtype: {t.dtype} (expected float32)')

# Spot check some FP32 targets
import random
hc_keys = [k for k in wm if 'hc_attn_base' in k][:3]
gate_bias = [k for k in wm if k.endswith('ffn.gate.bias')][:3]
ape = [k for k in wm if 'attn.compressor.ape' in k][:3]
for k in hc_keys + gate_bias + ape:
    with safe_open(ART / wm[k], framework='pt') as f:
        t = f.get_tensor(k)
    print(f'  {k}: {t.dtype} (expected float32)')
"

echo "[postproc] end=$(date -u +%FT%TZ)"
echo "POSTPROC_DONE"
