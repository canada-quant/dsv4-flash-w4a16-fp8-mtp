---
license: mit
language:
- en
tags:
- docker
- vllm
- deepseek
- blackwell
- rtx-pro-6000
- sm120
- w4a16
- fp8
- mtp
- speculative-decoding
pretty_name: DSv4-Flash W4A16+FP8+MTP — RTX PRO 6000 Docker image
---

# canada-quant/dsv4-flash-w4a16-rtxpro6000-image

Pre-built Docker image (`canada-quant/dsv4-w4a16-rtxpro6000:v1`) that serves
[`canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP)
on **RTX PRO 6000 Blackwell Server Edition (SM 12.0)** out of the box.

> Why this image exists: the W4A16 artifact needs a tightly-pinned vLLM build
> (`jasl/vllm@27fd665b` + canada-quant BF16-MTP cherry-pick + ~13 layers of
> dependency/patch fixes) to serve correctly on consumer/server Blackwell.
> Rebuilding all that on a fresh box is ~25 min of friction we already paid;
> this image saves you that time.

## Quickstart

```bash
# 1) Download the tarball (~14 GB compressed)
hf download canada-quant/dsv4-flash-w4a16-rtxpro6000-image \
    --include "*.tar.gz" --local-dir .

# 2) Load into Docker (~5 min)
docker load < dsv4-w4a16-rtxpro6000-v1.tar.gz

# 3) Pre-cache the W4A16 model onto NVMe (~159 GB; 1-2 min via xet on Brev)
HF_HOME=/opt/dlami/nvme/hf-cache hf download \
    canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP

# 4) Pull the serve script (TP-parameterized)
git clone https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp.git
cd dsv4-flash-w4a16-fp8-mtp

# 5) Serve TP=2 (2× RTX PRO 6000) — works on a single-socket box
docker run -d --gpus '"device=0,1"' --name dsv4-w4a16-serve \
    --shm-size=16g --ipc=host -p 8000:8000 \
    -v /opt/dlami/nvme/hf-cache:/root/.cache/huggingface \
    -v $(pwd)/scripts:/workspace/scripts:ro \
    -e TP=2 -e MAX_NUM_SEQS=4 -e MAX_MODEL_LEN=65536 -e GPU_MEM_UTIL=0.95 \
    canada-quant/dsv4-w4a16-rtxpro6000:v1 \
    bash /workspace/scripts/serve_rtx6000pro_w4a16.sh

# 6) Wait for ready (~3-5 min)
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; done

# 7) Smoke test
curl -sX POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"DSV4-W4A16-FP8-MTP",
         "messages":[{"role":"user","content":"What is 17*23?"}],
         "max_tokens":60,"temperature":0}' | jq .choices[0].message.content
# → "391"
```

For TP=4 (single replica, all 4 GPUs): use `--gpus all -e TP=4 -e MAX_NUM_SEQS=16`.

## What's in the image

Base: `nvcr.io/nvidia/pytorch:26.04-py3` (PyTorch 2.12.0a0, CUDA 13.2, Py 3.12).

vLLM: [`jasl/vllm@27fd665bdc3ba58afc5c34cbb9034c9fc1a95029`](https://github.com/jasl/vllm/tree/27fd665bdc3ba58afc5c34cbb9034c9fc1a95029)
(branch `ds4-sm120-preview-dev`), which carries:
- PR #40923 sm_120a Marlin MoE native cubins (eliminates JIT-PTX corruption)
- PR #43730's `c_tmp` clamp removal + Marlin MoE workspace 4× oversize
- `sm12x_deep_gemm_fallbacks.py` shims for DeepGEMM Hopper-only kernels
- canada-quant cherry-pick `5a49d88031 + 5acabf3152` (BF16 MTP block detection
  via safetensors header, fixes `wo_a.weight_scale` AttributeError when MTP
  block is unquantized in W4A16+FP8 mixed artifacts)

Runtime kernel pins (the 13-layer recipe):
| Pin | Why |
|---|---|
| `humming-kernels==0.1.2` | Quant kernel registry expects it; vLLM imports unconditionally |
| `quack-kernels==0.4.1` | DSv4 sparse attention compress path |
| `tokenspeed-mla==0.1.5` | MLA acceleration on Blackwell |
| `fastsafetensors==0.3.2` | Faster shard loading from local NVMe |
| `tilelang==0.1.10` | DSv4 attention HC head fusion kernel |
| `flashinfer-python==0.6.11.post3` | Worker import, sampling kernels |
| `flashinfer-cubin==0.6.11.post3` | Companion cubin payload |
| `nvidia-cutlass-dsl==4.5.0` | **PIN — 4.5.2 removes `cute.arch.fmin`** |
| `setuptools_rust` | Build dependency for tokenizers/safetensors wheels |

Additional patches applied in-image (see `docker/Dockerfile.rtx6000pro`):
- PR #43722: `MarlinFP8.can_implement` refuses block-FP8 → Triton fallback
- PR #43723: DSv4 `attention.py` `wo_a.weight_scale_inv`/`weight_scale` fallback
- `vllm/compilation/backends.py` `has_tuple_return = False` (NGC torch lacks
  `split_module(tuple_return=True)`)
- `sparse_attn_compress_cutedsl.py` `cute.arch.fmin` algebraic-identity shim
- `apt-get remove --purge python3-yaml` (blocks pip yaml installs)

Env defaults baked in:
- `VLLM_TEST_FORCE_FP8_MARLIN=1` (forces attention block-FP8 onto Marlin path)
- `VLLM_USE_LAYERNAME=0` (avoids Inductor MoE FakeScriptObject crash WITHOUT
  needing `--enforce-eager`, so CUDA graphs stay enabled)

## Verified configurations

See [`canada-quant/dsv4-flash-w4a16-fp8-mtp`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp)
README for the full bench matrix (AIME-2024 thinking-mode sweep at chat/high/max
across c=1/4, GSM8K-50 c=8, throughput sweep) on TP=2 and TP=4.

## License

MIT — inherits from upstream `deepseek-ai/DeepSeek-V4-Flash` model license and
vLLM Apache-2.0.

## Acknowledgments

- [jasl](https://github.com/jasl) for the `jasl/vllm` SM 12.0 preview branch
  carrying all the DSv4-on-Blackwell scheduling + kernel fixes.
- [haosdent](https://github.com/haosdent) for the original Marlin MoE `c_tmp`
  fix (vllm-project/vllm#36889).
- NVIDIA for the NGC PyTorch base image.
