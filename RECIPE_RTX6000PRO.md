# RTX PRO 6000 Blackwell (SM 12.0) — full reproduction recipe

End-to-end recipe to serve `canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP`
on **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPUs (96 GiB HBM3
each, SM 12.0). Targets TP=2 (2× GPU pair) and TP=4 (all four GPUs).

This is the **W4A16+FP8+MTP** path. NVFP4 on Blackwell is a separate
artifact (sibling repo `canada-quant/dsv4-flash-nvfp4-fp8-mtp`,
calibrated on B300 SM 10.0) and requires upstream
[vllm-project/vllm#31085](https://github.com/vllm-project/vllm/issues/31085)
to land before the native SM 12.0 NVFP4 MoE kernels become selectable.
Out of scope for this recipe; see the README §"Scope clarification" for
the boundary.

If you came here from the H200 recipe in `README.md`: the H200 path
runs the same artifact through **upstream `vllm-project/vllm` HEAD**
with four cherry-picked PRs. The RTX 6000 Pro path uses **`jasl/vllm`
branch `ds4-sm120-preview-dev`** (the SM12-tuned vLLM rebase) plus
several additional patches that compensate for the SM12 branch's
narrower assumptions about which attention modules are quantized.
See "Why the H200 patches aren't enough" at the bottom for the full
delta.

---

## TL;DR — Quick start (paste into a fresh Brev `g7e.24xlarge`)

```bash
# Prereqs that Brev ships pre-installed: Ubuntu 22.04, NVIDIA driver
# 580.x, CUDA 12.9 toolkit. The script below installs CUDA 13.0
# alongside (needed for torch 2.11+cu130).

sudo apt-get update && sudo apt-get install -y git
sudo ln -sfn /opt/dlami/nvme /scratch && sudo chown -h "$USER:$USER" /scratch
cd /scratch
git clone https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp.git
cd dsv4-flash-w4a16-fp8-mtp

# 1) Build vLLM from jasl/vllm@ds4-sm120-preview-dev (~25 min)
bash scripts/bootstrap_rtx6000pro.sh

# 2) Extra deps (Rust, humming, flashinfer pins)
source ~/venv-serve/bin/activate
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
pip install --quiet setuptools-rust
pip install --quiet git+https://github.com/inclusionAI/humming.git
pip install --quiet "flashinfer-python==0.6.8.post1" "flashinfer-cubin==0.6.8.post1" \
    "numba==0.65.0" "tilelang==0.1.9" "apache-tvm-ffi==0.1.9" "fastsafetensors>=0.2.2"

# 3) Apply 4 patches against the installed vLLM
python scripts/patch_v4_forcausal_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_mtp_packed_mapping.py        "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_nvidia_attn_scale.py         "$(python -c 'import vllm; print(vllm.__path__[0])')"
bash   scripts/patch_wo_a_bf16_path.sh             "$(python -c 'import vllm; print(vllm.__path__[0])')"

# 4) Download artifact (~1.5 min on Brev's 10 Gbps egress)
pip install --user --quiet huggingface_hub hf-transfer
export PATH="$HOME/.local/bin:$PATH"
hf download canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP \
    --local-dir /scratch/weights/w4a16-fp8-mtp-gptq

# 5) Dequantize compressor + indexer modules (~1.5 min, one-time)
python scripts/dequant_compressor.py /scratch/weights/w4a16-fp8-mtp-gptq

# 6) Serve (TP=2 on GPUs 0+1 is the recommended config — 2 replicas
#    fit on the 4-GPU box and give best aggregate $/token)
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/w4a16-fp8-mtp-gptq 8000 2 &
# wait for /health 200, then benchmark:
while ! curl -fsS http://localhost:8000/health; do sleep 5; done
bash scripts/chat_smoke.sh http://localhost:8000
bash scripts/bench_rtx6000pro_suite.sh http://localhost:8000 2 1
```

**Expected first-replica throughput (with cudagraph + MTP-spec k=1):**

| Batch | Output tok/s | TPOT median (ms) | MTP acceptance |
|---|---|---|---|
| bs=1 | ~99 (TP=2) / ~107 (TP=4) | ~8.5 (TP=2) / ~7.8 (TP=4) | 68-71% |
| bs=4 | ~220 | ~12-14 | 68-71% |
| bs=16 | ~483 (TP=2) / ~584 (TP=4) | ~25-30 | 71% |

**Wall-clock for the full flow on a fresh box:** ~35-40 minutes
including the vLLM build, deps, artifact download, dequant, and first
serve startup. After that, every restart is ~3 min model load +
~30 s cudagraph capture.

For the detailed walkthrough — patch rationale, debug notes, future
work — keep reading.

---

## 0. Hardware tested

Brev instance `familiar-teal-worm` (org `NCA-d2e3-84318`):

| Field | Value |
|---|---|
| GPU model | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU count | 4 |
| HBM per GPU | 96 GiB (97887 MiB nominal) |
| Compute capability | SM 12.0 |
| Driver | 580.159.03 |
| Pre-installed CUDA | 12.9 |
| vCPU | 96 |
| RAM | 1 TiB |
| Root disk | 256 GiB |
| Ephemeral NVMe LVM | 7.6 TiB at `/opt/dlami/nvme` |
| GPU topology | GPUs 0–1 on one PCIe switch (PIX), GPUs 2–3 on another; cross-switch is PCIe root (NODE) |
| AWS instance type | `g7e.24xlarge` |
| Region | Columbus OH |
| Hourly rate | $19.92/h |

The 7.6 TiB ephemeral LVM is the right place for everything (artifact,
venv, vLLM source tree). We symlink it as `/scratch` to match the H200
convention.

---

## 1. Bootstrap (~25 min)

`scripts/bootstrap_rtx6000pro.sh` is idempotent and does the full
install. From a freshly provisioned Brev box:

```bash
sudo apt-get update && sudo apt-get install -y git
sudo ln -sfn /opt/dlami/nvme /scratch
sudo chown -h "$USER:$USER" /scratch
cd /scratch
git clone https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp.git
cd dsv4-flash-w4a16-fp8-mtp
bash scripts/bootstrap_rtx6000pro.sh
```

What the script does:

1. Installs `cuda-toolkit-13-0` alongside the box's pre-installed 12.9
   (the vLLM build needs CUDA 13.0 headers to match torch 2.11+cu130).
2. Creates `~/venv-serve` with torch 2.11.0+cu130.
3. Clones `jasl/vllm@ds4-sm120-preview-dev` to `~/src/vllm`.
4. Source-builds vLLM with `TORCH_CUDA_ARCH_LIST=12.0a` and
   `MAX_JOBS=32`. Takes ~25 min.
5. Applies the two `packed_modules_mapping` patches (see §3.1 below).

After bootstrap, two more dependencies that the script does NOT
auto-install but are needed before `vllm serve` works:

```bash
source ~/venv-serve/bin/activate

# 5a. Rust toolchain + setuptools-rust (jasl branch has Rust extensions)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
pip install --quiet setuptools-rust

# 5b. humming-kernels (jasl branch hard-imports it for the FP8 path)
pip install --quiet git+https://github.com/inclusionAI/humming.git

# 5c. flashinfer + numba + tilelang (jasl's pinned versions)
pip install --quiet \
    "fastsafetensors>=0.2.2" \
    "flashinfer-cubin==0.6.8.post1" \
    "flashinfer-python==0.6.8.post1" \
    "numba==0.65.0" \
    "tilelang==0.1.9" \
    "apache-tvm-ffi==0.1.9"
```

**Note on the CMake `spinloop` extension:** jasl's `CMakeLists.txt`
declares the `spinloop` C extension with `USE_SABI 3.11` (Python 3.11
limited API). Our box runs Python 3.10, so the build fails on missing
`Py_buffer` symbols. Drop the `USE_SABI 3.11` line from the `spinloop`
target before building — the extension then builds against Python 3.10's
full API:

```bash
sed -i '/USE_SABI 3\.11/d' ~/src/vllm/CMakeLists.txt
```

This is wired into `bootstrap_rtx6000pro.sh` as a pre-build patch.

---

## 2. Artifact (~1.5 min from HuggingFace)

```bash
export PATH="$HOME/.local/bin:$PATH"
pip install --user --quiet huggingface_hub hf-transfer
mkdir -p /scratch/weights
hf download canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP \
    --local-dir /scratch/weights/w4a16-fp8-mtp-gptq
```

159 GiB total. With `hf-transfer` enabled it completes in ~1.5 min on
Brev's bandwidth.

---

## 3. Patches

The RTX 6000 Pro path needs **six patches**: the two `packed_modules_mapping`
patches from the H200 path (auto-applied by `bootstrap_rtx6000pro.sh`),
**plus four SM12-specific patches** that are unique to this build.

### 3.1 H200-compatible patches (auto-applied by bootstrap)

```bash
python scripts/patch_v4_forcausal_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_mtp_packed_mapping.py        "$(python -c 'import vllm; print(vllm.__path__[0])')"
```

These add `packed_modules_mapping = {"fused_wqa_wkv": [...], "fused_wkv_wgate": [...], "gate_up_proj": [...]}`
to `DeepseekV4ForCausalLM` and `DeepSeekV4MTP` so the compressed-tensors
scheme resolver can find fused attention modules.

### 3.2 SM12-specific patch: weight_scale_inv → weight_scale fallback

`vllm/models/deepseek_v4/nvidia/ops/attention.py:370` accesses
`self.wo_a.weight_scale_inv` directly. The artifact uses `.weight_scale`
(no `_inv` suffix — the artifact was calibrated with the W8A8Fp8 naming
convention, not W8A16Fp8). Falls back to `weight_scale` if no `_inv`:

```python
# Apply via scripts/patch_nvidia_attn_scale.py
wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
if wo_a_scale is None:
    wo_a_scale = self.wo_a.weight_scale
```

This mirrors what upstream PR #43290 did to the (different file path)
`vllm/models/deepseek_v4/attention.py` on the H200 side.

### 3.3 SM12-specific patch: BF16 wo_a path for MTP block (dynamo-safe)

The MTP block (layer 43) is preserved at BF16 in the artifact (Option Y).
Its `wo_a` therefore has NO weight scale at all — neither `weight_scale`
nor `weight_scale_inv`. The vLLM `nvidia/ops/attention.py:forward()`
unconditionally takes the FP8 fast path which needs `wo_a_scale`. This
crashes during `profile_run` when the spec-decode drafter exercises
the MTP block.

Fix: take the same BF16 reference path that's already used on ROCm
when the MTP wo_a is BF16. **Use the weight dtype, not a `getattr(...)`
fallback** — dynamo's `_getattr_static` can't resolve dynamically
registered attributes like `weight_scale`, but it CAN constant-fold a
`tensor.dtype == torch.bfloat16` comparison at trace time:

```python
# Apply via scripts/patch_wo_a_bf16_path.sh
if current_platform.is_rocm() or self.wo_a.weight.dtype == torch.bfloat16:
    z = rocm_inv_rope_einsum(
        self.rotary_emb, o, positions, self.rope_head_dim,
        self.n_local_groups, self.o_lora_rank, self.wo_a,
    )
    return self.wo_b(z.flatten(1))
# else fall through to FP8 einsum (unchanged)
```

**Critical:** this dtype-based check is what unlocks `torch.compile` +
cudagraph on RTX 6000 Pro. An earlier iteration of this patch used
`getattr(self.wo_a, "weight_scale_inv", None) is not None` and forced
`--enforce-eager`, which crippled throughput by ~10× (bs=1 ran at
11.6 tok/s eager vs **98.8 tok/s with cudagraph**). The dtype rewrite is
the single most impactful patch in this recipe.

### 3.4 Compressor / indexer.weights_proj are FP8 in artifact but unquantized in vLLM

`vllm/models/deepseek_v4/compressor.py` and
`vllm/models/deepseek_v4/nvidia/ops/attention.py` construct
`compressor.fused_wkv_wgate`, `indexer.weights_proj`,
`indexer.compressor.fused_wkv_wgate`, and `indexer.wq_b` with
`quant_config=None` — i.e. as unquantized BF16 modules. **Our artifact
explicitly quantizes these to FP8_BLOCK** per the calibration recipe.

vLLM main has the same hardcode. The H200 path runs into the same
mismatch but the H200 vLLM build (with cherry-picks #43248/#43288/#43290/#43319)
handles it via a different code path. On the SM12 build, the simplest
fix is to **dequantize the artifact's FP8 compressor/indexer weights to
BF16 at load preprocessing time**:

```bash
python scripts/dequant_compressor.py /scratch/weights/w4a16-fp8-mtp-gptq
```

The script (`scripts/dequant_compressor.py`):
1. Walks all 4 shards
2. For each `layers.X.attn.compressor.{wkv,wgate}.weight` (FP8) plus its
   matching `.weight_scale` (BF16 block scale 128×128):
   - dequantize → BF16 weight
   - replace the FP8 .weight with the BF16 dequantized version
   - drop the .weight_scale key
3. Same for `indexer.weights_proj`, `indexer.wq_b`, and
   `indexer.compressor.{wkv,wgate}`.
4. Updates `model.safetensors.index.json`.

Total: 166 weights dequantized in ~1.5 min wall. The artifact size
shrinks slightly (per-shard scale tensors are dropped). The compressor
modules then load as plain BF16 — quality loss is bounded by the FP8
calibration noise (small for these utility modules; main expert/attn
weights stay W4A16 / FP8).

**Edge-case bug in the dequant script** — fixed in commit history:
`.replace(".weight", ".weight_scale")` over-replaces on
`indexer.weights_proj.weight` because the substring `.weight` appears
twice. Use `rsplit(".weight", 1)[0] + ".weight_scale"` to only
substitute the suffix. The committed script has the fix.

---

## 4. Serve

After patches + dequant, launch the serve script. It runs with full
`torch.compile` + cudagraph by default (the dynamo-safe wo_a fix in
§3.3 makes that possible) — **do NOT add `--enforce-eager`**; the
eager-mode path is ~10× slower (see "CUDA graphs work — and matter"
below).

```bash
# TP=2 (one PCIe-switch-bound GPU pair, GPUs 0,1)
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/w4a16-fp8-mtp-gptq 8000 2

# TP=4 (all four GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/w4a16-fp8-mtp-gptq 8000 4
```

Important flags baked into `scripts/serve_rtx6000pro.sh`:

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | 2 or 4 | passed via $3 |
| `--kv-cache-dtype` | fp8 | match H200 |
| `--block-size` | 256 | match H200 |
| `--max-model-len` | 4096 | smoke + bench config; raise after stability proves out |
| `--max-num-seqs` | 16 | bench-friendly batch lane |
| `--gpu-memory-utilization` | **0.95** | tight without much headroom after model weights |
| `--no-enable-prefix-caching` | (set) | match H200 |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":1}` | k=1 is the upstream-stable ceiling on SM12 |
| `--disable-custom-all-reduce` | **(set)** | RTX 6000 Pro lacks NVLink — the custom AR kernel fails with CUDA invalid-argument |

### CUDA graphs work — and matter

With the dynamo-safe wo_a dtype check (§3.3), full `torch.compile` +
cudagraph capture works. Throughput vs eager mode:

| Metric | eager (--enforce-eager) | cudagraph (default) | Speedup |
|---|---|---|---|
| TP=2 bs=1 output tok/s | 11.57 | 98.83 | **8.5×** |
| TP=2 bs=1 TPOT (ms) | 82.70 | 8.55 | **9.7×** |
| TP=2 bs=16 output tok/s | 147.00 | 482.61 | **3.3×** |

**Always run with cudagraphs on.** The eager-mode build is documented
in the `benchmarks/rtx6000pro/2026-05-23-throughput-summary.md` file as
a comparison reference only.

---

## 5. Benchmark

```bash
# TP=2 (skip the hour-long MMLU/AIME items)
bash scripts/bench_rtx6000pro_suite.sh http://localhost:8000 2 1

# TP=4
bash scripts/bench_rtx6000pro_suite.sh http://localhost:8000 4 1
```

Outputs land in `benchmarks/rtx6000pro/tp{N}_{TIMESTAMP}/`. The suite runs:

| Bench | Status on RTX 6000 Pro |
|---|---|
| chat_smoke quick (4 prompts) | ✅ 4/4 PASS |
| MTP acceptance @ 200 (random prompts) | ✅ reported by vLLM `/metrics` |
| Throughput TPOT @ bs=1/4/16 via `vllm bench serve` | ✅ |
| GSM8K 8-shot, MMLU 5-shot, HumanEval, AIME 24 | ⚠️ Skipped in this first run — eager mode makes the full sets ~8× slower than H200; would take 8–12 h |

Benchmark numbers land in `BENCHMARKS.md` (this repo's headline doc).

---

## 6. Cost + wall-clock

| Phase | Wall (eager mode) | Notes |
|---|---|---|
| Bootstrap (vLLM source build) | ~25 min | one-time |
| Extra deps (rust, humming, flashinfer) | ~5 min | one-time |
| Artifact download | ~1.5 min | from HF + hf-transfer |
| Compressor dequant preprocess | ~1.5 min | one-time |
| Patch application | ~30 s | one-time |
| Serve TP=2 startup | ~3.5 min | model load + warmup |
| Serve TP=4 startup | ~5 min | model load + warmup |
| Throughput suite (bs=1/4/16) | ~10 min | 8N requests at concurrency N |

**At $19.92/h, a single TP=2 build+bench cycle is ~$15. Both
TP=2 + TP=4 + docs is ~$20-30.**

---

## 7. Future work

1. **Upstream the BF16 wo_a dtype fallback.** This is the dynamo-safe
   shape of upstream PR #43319 (auto-detect BF16 MTP); the runtime
   forward still needs the dtype branch we added. PR candidate.
2. **Native NVFP4 MoE on SM120.** vLLM-project/vllm#31085 reports the
   NVFP4 SM120 kernels exist (`nvfp4_scaled_mm_sm120_kernels.cu`,
   `nvfp4_blockwise_moe_kernel.cu`) but the backend selector only
   recognizes SM100. With that fixed and the sibling NVFP4-FP8-MTP
   artifact, we'd land on a tighter expert kernel than the current
   Marlin W4A16 path. The MoE bf16 acts × NVFP4 weights pattern is
   close to ideal for RTX 6000 Pro's Blackwell-consumer FP4 unit.
3. **k=2 spec-decode** — same DeepGEMM ceiling as H200, unrelated to
   hardware. Tracked in C15.
4. **Skip the dequant step** if jasl lands a per-attribute
   `quant_config` override for compressor / indexer.weights_proj /
   indexer.wq_b. The right end-state is the model class consuming the
   artifact's `quantization_config.config_groups` natively for these
   modules.
5. **Accuracy benchmarks** — the bench suite's lm-eval invocation
   needs `model_args=tokenizer=/path/to/artifact` to avoid HF model-id
   lookup. Apply that and re-run GSM8K / MMLU / HumanEval / AIME 24
   on TP=2 cudagraph to validate quality parity with H200.

---

## 8. Why the H200 patches aren't enough (summary)

| Issue | H200 build | RTX 6000 Pro (SM 12.0) build |
|---|---|---|
| vLLM base | upstream `main HEAD 50d9dd902` | `jasl/vllm@ds4-sm120-preview-dev` |
| `TORCH_CUDA_ARCH_LIST` | `9.0a` | `12.0a` |
| File layout | post-refactor `vllm/models/deepseek_v4/nvidia/*` | same (preview-dev was rebased) |
| `packed_modules_mapping` patches | required | required (same patches) |
| `weight_scale_inv → weight_scale` fallback | done via PR #43290 cherry-pick | needs §3.2 patch (separate file path under `nvidia/ops/`) |
| BF16 wo_a (MTP block) | works via cudagraph w/ static class type | needs §3.3 runtime branch with the **dynamo-safe `weight.dtype == bfloat16` check** so cudagraph still captures (no `--enforce-eager` needed) |
| FP8 compressor / indexer.weights_proj / wq_b loading | works (different code path) | needs §3.4 dequant preprocess |
| Rust extension build | not present | requires Rust toolchain + setuptools-rust |
| `humming-kernels` import | not used | hard-imported at quant_config load |
| spinloop USE_SABI 3.11 | n/a | requires `sed` removal for Python 3.10 |

The summary: SM12's branch is younger, has lighter "happy-path"
assumptions about which attention modules are quantized, and the
toolchain has more moving parts (Rust, humming) than the H200 main
build. The patches above + the dequant preprocess close the gap.
