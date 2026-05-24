# Sister-agent handoff: NVFP4-FP8-MTP on RTX 6000 Pro Blackwell

**Audience:** the next agent who will take the `canada-quant/dsv4-flash-nvfp4-fp8-mtp`
sibling artifact and reproduce on the **same Brev `familiar-teal-worm`
box** (4× NVIDIA RTX PRO 6000 Blackwell Server Edition, SM 12.0) that
we just used for the W4A16+FP8+MTP path.

This document is what we learned doing the W4A16 work, with annotations
about which lessons transfer, which don't, and what's specifically new
about NVFP4 on SM 12.0.

**Author:** the W4A16 agent (2026-05-23 / 24).
**Box state at handoff:** all 4 GPUs idle (`nvidia-smi memory.used = 0
MiB`). vLLM build under `~/src/vllm` is jasl's `ds4-sm120-preview-dev`
at SHA `c79225692`. venv at `~/venv-serve` is configured for that
build. The W4A16 artifact is at `/scratch/weights/w4a16-fp8-mtp-gptq`
(159 GiB) — leave it or `rm -rf` to free disk for the NVFP4 artifact
(also ~120-160 GiB).

---

## 0. Quick-orient yourself

```bash
# What's the box?
nvidia-smi -L
# 4× NVIDIA RTX PRO 6000 Blackwell Server Edition, ~96 GiB each, SM 12.0

# What's already installed?
source ~/venv-serve/bin/activate
python -c "import vllm; print(vllm.__version__, vllm.__file__)"
# 0.1.dev16959+gc79225692.d20260523.cu129  /home/ubuntu/venv-serve/lib/python3.10/site-packages/vllm/__init__.py

# What's on disk?
df -h /opt/dlami/nvme
# 7.6 TB free; /scratch -> /opt/dlami/nvme

# Where's the W4A16 artifact?
du -sh /scratch/weights/w4a16-fp8-mtp-gptq
# ~159 GB

# What's our git state?
cd /opt/dlami/nvme/dsv4-flash-w4a16-fp8-mtp
git log --oneline | head -5
# (most recent commit is the RTX 6000 Pro work)
```

---

## 1. What transfers from W4A16 → NVFP4

These lessons are hardware/branch/toolchain and apply identically:

### 1.1 vLLM build = `jasl/vllm@ds4-sm120-preview-dev`

Don't bother with `ds4-sm120-experimental` — that's the May-6
branch the H200/predecessor work used; it's missing the post-refactor
file layout, dynamo-safe wo_a stack, and a few SM12 fixes. `preview-dev`
is jasl's current SM12 work and rebased on post-refactor upstream main.

Build is already done. If you need to rebuild from scratch (e.g.
clearing torch_compile_cache):
```bash
bash /opt/dlami/nvme/dsv4-flash-w4a16-fp8-mtp/scripts/bootstrap_rtx6000pro.sh
```
~25 min on this box (96 vCPU, MAX_JOBS=32).

### 1.2 Deps that aren't in jasl's pyproject

Already installed in `~/venv-serve`, but if you rebuild venv:

```bash
pip install --quiet setuptools-rust   # jasl's branch has Rust extensions
pip install --quiet git+https://github.com/inclusionAI/humming.git
# vLLM's quantization __init__ unconditionally imports HummingConfig on
# CUDA. We don't USE humming, but the import has to succeed.

pip install --quiet "flashinfer-python==0.6.8.post1" "flashinfer-cubin==0.6.8.post1" \
    "numba==0.65.0" "tilelang==0.1.9" "apache-tvm-ffi==0.1.9" "fastsafetensors>=0.2.2"
# These are jasl's hard-pinned versions. Mismatches get flagged at
# install time as "incompatible" but mostly run fine; the FAIL mode
# was specifically humming + the flashinfer pin.
```

Also Rust toolchain (from rustup, already installed at `~/.cargo`):
```bash
. "$HOME/.cargo/env"
cargo --version  # rustc 1.95+
```

### 1.3 CMake `USE_SABI 3.11` removal

If you rebuild, jasl declares `spinloop` C extension with
`USE_SABI 3.11` (Python 3.11 limited API). Our Python is 3.10 — missing
`Py_buffer` symbol → build fails. Already patched in the source tree
but if you do a fresh `git clone`, you'll need to:
```bash
sed -i '/USE_SABI 3\.11/d' ~/src/vllm/CMakeLists.txt
```

`bootstrap_rtx6000pro.sh` doesn't auto-apply this; it's an issue jasl
should fix upstream.

### 1.4 `--disable-custom-all-reduce` is mandatory

RTX PRO 6000 Blackwell lacks NVLink. vLLM's custom_all_reduce kernel
fails with `CUDA error /home/ubuntu/src/vllm/csrc/custom_all_reduce.cuh:455 'invalid argument'`
during `Profiling CUDA graph memory: PIECEWISE=32...` if you don't set
this flag. It's already in `scripts/serve_rtx6000pro.sh`. Copy it
forward.

### 1.5 The dynamo-safe pattern (THE most important lesson)

**Background:** `torch.compile`'s dynamo intercepts attribute lookups
via `_getattr_static()`, which only inspects the **class** for the
attribute — it does NOT see instance attributes that get registered
dynamically (e.g. via `register_parameter`). So:

```python
# DOES NOT WORK under torch.compile — getattr fallback raises
# ObservedAttributeError at trace time.
wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
if wo_a_scale is None:
    wo_a_scale = self.wo_a.weight_scale   # not statically declared on the class
```

```python
# DOES work — tensor.dtype is constant-foldable at trace time.
if self.wo_a.weight.dtype == torch.bfloat16:
    # BF16 (e.g. MTP block) path
else:
    # FP8 / quantized path
```

**Why it matters for you:** the NVFP4 artifact ALSO preserves the MTP
block at BF16 (Option Y is universal — see `FINDINGS_FOR_SIBLING.md`).
You will hit the SAME mismatch at runtime: code paths that assume the
attention/expert weights are NVFP4 will trip on the BF16 MTP block.
**The fix shape is the same:** use a static dtype check, not a
`getattr(..., None)` fallback.

Critical files where the NVFP4 path likely needs the same treatment:
- `vllm/models/deepseek_v4/nvidia/ops/attention.py` (wo_a check —
  already patched for W4A16; verify same patch applies to NVFP4 attn
  path)
- `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` and any
  `nvfp4_*` analog if NVFP4 has its own einsum op
- `vllm/models/deepseek_v4/compressor.py` and
  `vllm/models/deepseek_v4/nvidia/ops/attention.py:indexer.weights_proj`
  — these are constructed with `quant_config=None` in the SM12 branch.
  If NVFP4 calibration quantized them, you'll hit the same KeyError
  on weight_scale and the same need to dequant or override.

### 1.6 GPU teardown footgun: `pkill -f vllm` doesn't kill workers

Workers are spawned with names like `VLLM::Worker_TP0` — `pkill -f
vllm` matches the parent, not the workers, and GPU memory stays pinned
for several minutes. Use this instead between runs:

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    | while read p; do kill -9 $p; done
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# All should be 0 MiB before launching the next serve.
```

### 1.7 Brev SSH is flaky under rapid-fire

The Brev CLI proxy drops connections often, especially under back-to-back
`brev exec` calls. Symptoms: `exit status 255`, "Connection failed,
checking instance status...". It's not the box — it's the proxy.
Strategy: write a single multi-step shell script, `brev copy` it over,
run as one `brev exec`. Don't chain 5 small commands.

---

## 2. What's specifically new for NVFP4

### 2.1 The NVFP4 sibling artifact

Lives at:
`canada-quant/dsv4-flash-nvfp4-fp8-mtp` (HuggingFace)
`canada-quant/dsv4-flash-nvfp4-fp8-mtp` (GitHub)

Calibrated by the sibling team on B300 (SM 10.0). Different recipe
than W4A16 — NVFP4 (4-bit float) on MoE experts instead of W4A16 INT4;
attention may differ. **Check the sibling's README/findings doc for
the exact recipe before assuming anything about file layout.**

### 2.2 vLLM SM 12.0 NVFP4 MoE backend selector — `vllm-project/vllm#31085`

**The biggest known blocker.** Upstream vLLM has kernels for SM 12.0
NVFP4 MoE (`csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`
and `csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu`) but the
backend selector in `vllm/model_executor/layers/quantization/mxfp4.py`
only checks `arch_major == 10` (SM100 family). On SM120 it falls back
to Marlin, which loses the FP4 hardware acceleration.

Status as of 2026-05-23: open feature request, no merged PR. The fix
shape is documented in the issue — extend the family check to include
`(12, 0)`.

**Your options:**
1. Patch the selector locally (~3 lines in `mxfp4.py`); validate FP4
   kernel actually fires on RTX 6000 Pro
2. Wait for upstream
3. Fall back to Marlin (much slower; loses the headline NVFP4
   advantage but at least loads)

Recommend (1) — small, contained, and you can put a "PATCH (paul/dsv4):
SM120 NVFP4 selector" comment on it.

### 2.3 NVFP4 weight_scale naming

NVFP4 schemes in compressed-tensors typically use `weight_global_scale`
+ block-scale (similar to MXFP4). The "weight_scale_inv vs weight_scale"
fallback (our patch_nvidia_attn_scale.py) may NOT apply — NVFP4 has its
own naming convention. **Read the compressed_tensors NVFP4 scheme class
first** (`vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_fp4.py`
or similar). Don't blindly copy our patches.

### 2.4 The compressor / indexer.weights_proj dequant question

For W4A16, the artifact has FP8_BLOCK weights on the compressor and
indexer.weights_proj modules. jasl's SM12 branch builds these with
`quant_config=None` (unquantized BF16). Our fix was to **dequantize
those modules to BF16 in the artifact** at preprocess time
(`scripts/dequant_compressor.py`, 166 weights, ~1.5 min).

**For NVFP4, check the sibling artifact's keys first:**
```bash
python3 -c "
import json
with open('<artifact>/model.safetensors.index.json') as f:
    wm = json.load(f)['weight_map']
keys = list(wm)
print('compressor keys:',  sum('compressor' in k for k in keys))
print('  with .weight_scale:', sum('compressor' in k and 'weight_scale' in k for k in keys))
print('  with .weight_global_scale:', sum('compressor' in k and 'weight_global_scale' in k for k in keys))
print('indexer keys:',    sum('indexer'    in k for k in keys))
print('  weights_proj.weight:', sum('weights_proj.weight' == k.rsplit('.', 1)[-1] + '.weight' for k in keys))"
```

If NVFP4 *also* quantizes compressor / indexer.weights_proj, you'll
need a port of `dequant_compressor.py` that handles NVFP4 dequant
(weight_global_scale + per-block scale + FP4 packing → BF16). Larger
effort than our FP8 dequant, but the SHAPE of the fix is the same.

If NVFP4 calibration **doesn't** quantize compressor/indexer, you're
luckier — no preprocess needed.

### 2.5 MTP block on NVFP4 + RTX 6000 Pro — most likely scenario

Most likely scenario (informed by W4A16 work):

1. Apply the SM 12.0 NVFP4 selector patch (vllm#31085 fix)
2. NVFP4 main experts + FP8 attn load fine via the patched selector
3. The MTP block (BF16) trips the wo_a forward path → apply our
   dynamo-safe dtype check (already patched in this clone's vllm,
   if you keep the install)
4. compressor / indexer.weights_proj may or may not need dequant
   (see §2.4)
5. Add `--disable-custom-all-reduce`, drop any `--enforce-eager`
6. Serve at TP=2 first (recommended config — 2 replicas on 4 GPUs)
7. Bench: chat-smoke, then `vllm bench serve` at bs=1/4/16 with
   `--random-input-len 256 --random-output-len 256`
8. **Confirm MTP hit rate is in the 65-75% band** (matches our W4A16
   numbers and the sibling's published HumanEval acceptance of 67%
   on B300). If acceptance drops below 50%, MTP module is being
   loaded incorrectly — most likely the wo_a / scale fallback is
   wrong.

---

## 3. Key files in this repo you'll want to reference / port

Path | Purpose | Likely-reusable for NVFP4
---|---|---
`scripts/bootstrap_rtx6000pro.sh` | jasl/vllm@ds4-sm120-preview-dev source build | YES (verbatim)
`scripts/serve_rtx6000pro.sh` | serve command + flags | YES (modulo --quantization=... arg for NVFP4 if needed)
`scripts/serve_rtx6000pro_nospec.sh` | serve with spec-decode disabled (baseline) | YES
`scripts/patch_v4_forcausal_packed_mapping.py` | add packed_modules_mapping | MAYBE (NVFP4 may already have this; verify first)
`scripts/patch_mtp_packed_mapping.py` | same for MTP class | MAYBE
`scripts/patch_nvidia_attn_scale.py` | weight_scale_inv → weight_scale fallback | UNLIKELY — NVFP4 uses different scale names
`scripts/patch_wo_a_bf16_path.sh` | dynamo-safe BF16 wo_a fallback | YES — this is the universal lesson
`scripts/dequant_compressor.py` | FP8 → BF16 for compressor/indexer | Port-the-shape — NVFP4 dequant math differs
`scripts/bench_rtx6000pro_suite.sh` | chat-smoke + acceptance@200 + vllm bench | YES (verbatim)
`scripts/summarize_bench.py` | extract metrics from bench json (under `/tmp/`) | YES — copy from `/tmp/summarize_bench.py` on this box

---

## 4. NVFP4 Quick start (your turn)

```bash
# 0) Free up the GPUs + reset disk space if needed
nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do kill -9 $p; done
# (only if you want disk back: rm -rf /scratch/weights/w4a16-fp8-mtp-gptq)

# 1) vLLM build is already done — skip bootstrap unless you need fresh
source ~/venv-serve/bin/activate

# 2) Patch SM 12.0 NVFP4 backend selector (write this — see §2.2)
#    See vllm-project/vllm#31085 for the fix shape.
python scripts/patch_sm120_nvfp4_selector.py  # YOU NEED TO WRITE THIS

# 3) Download the sibling artifact
export PATH="$HOME/.local/bin:$PATH"
hf download canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP \
    --local-dir /scratch/weights/nvfp4-fp8-mtp

# 4) Inspect the artifact's keys (§2.3, §2.4) — decide if compressor
#    needs dequant
python3 -c "<inspection snippet from §2.4>"

# 5) Apply our universal patches (if file layout matches)
python scripts/patch_v4_forcausal_packed_mapping.py "$(python -c 'import vllm; print(vllm.__path__[0])')"
python scripts/patch_mtp_packed_mapping.py        "$(python -c 'import vllm; print(vllm.__path__[0])')"
# patch_wo_a_bf16_path.sh — already applied to vllm in this clone

# 6) Serve TP=2
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_rtx6000pro.sh \
    /scratch/weights/nvfp4-fp8-mtp 8000 2 &

# 7) Wait for /health 200; expect ~3 min model load + ~30s cudagraph capture
while ! curl -fsS http://localhost:8000/health; do sleep 5; done

# 8) Smoke + bench
bash scripts/chat_smoke.sh http://localhost:8000
bash scripts/bench_rtx6000pro_suite.sh http://localhost:8000 2 1

# 9) Compare to W4A16 numbers in benchmarks/rtx6000pro/2026-05-24-cudagraph-summary.md
#    Headline you're aiming for:
#      bs=1 tok/s:  W4A16 TP=2 = 98.83.  NVFP4 should match or beat
#                  (FP4 has more compute density per byte).
#      bs=1 TPOT:   W4A16 TP=2 = 8.55 ms. NVFP4 target similar.
#      MTP accept:  68-75% (matches W4A16 + sibling's published 67%).
```

---

## 5. Anti-patterns we hit (and you should skip)

1. **Don't use `jasl/vllm@ds4-sm120-experimental`** — older branch
   (May 6), missing post-refactor file paths, missing the SM12 fixes.
   Cost us ~3 hours of patches we then threw away.

2. **Don't try `--enforce-eager` as a workaround.** It "works" but
   gives ~10× decode slowdown. Fix the dynamo issue properly with
   the dtype check (§1.5).

3. **Don't `pkill -f vllm`** — workers survive (§1.6). Use
   `nvidia-smi --query-compute-apps=pid` and kill by PID.

4. **Don't trust HF discussion threads about jasl's branch.** They're
   often older snapshots. The branch evolves daily; check
   `git log --oneline | head -20` on `ds4-sm120-preview-dev` for
   recent activity.

5. **Don't pre-install humming-kernels from PyPI** — it's not there.
   You need `pip install git+https://github.com/inclusionAI/humming.git`
   from the source repo. The PyPI install fails with "no matching
   distribution" and confuses the eye for ~15 min before you find the
   right repo via DeepWiki / `inclusionAI/humming`.

6. **Don't extrapolate node throughput from single-replica without
   saying so.** The H200 numbers in our docs are per-replica TP=2 on
   the 8-GPU box. We documented this explicitly after a review found
   the comparison was ambiguous. Do the same — if you report 100 tok/s
   for "RTX 6000 Pro NVFP4", say whether that's per-replica TP=2 or
   per-replica TP=4 or node aggregate.

---

## 6. What's still unknown / open questions for you to answer

1. **Does the NVFP4 sibling artifact load on `jasl/vllm@ds4-sm120-preview-dev`
   without code changes (modulo the selector fix)?** Probably not, but
   try `--enforce-eager` first to disentangle "won't load at all" from
   "won't cudagraph". If even eager fails, the problem is upstream of
   the dynamo issue.

2. **Does vllm-project/vllm#31085 fix bring real FP4 kernel
   acceleration on RTX 6000 Pro, or does the kernel itself have an SM12
   bug?** Kernels were "compiled but unselected" — check with a small
   test (single linear forward, NVFP4 weight + BF16 input → BF16 out)
   that the actual nvfp4_blockwise_moe_kernel runs without CUDA errors.

3. **MTP acceptance under NVFP4 vs W4A16.** The sibling reports 67%
   acceptance on B300 with NVFP4. Our W4A16 on RTX 6000 Pro hits 68-72%.
   Theoretical NVFP4 + RTX 6000 Pro = same band? Higher because of FP4
   kernel acceleration on draft path? Worth measuring.

4. **k=2 spec-decode** is blocked by the DeepGemm assertion (C15 in
   FINDINGS_FOR_SIBLING.md). Same on RTX 6000 Pro. Don't burn time on
   it until upstream fixes.

5. **Aggregate node throughput at 2× TP=2 replicas behind a load
   balancer on the 4-GPU box.** We extrapolated ~198 tok/s at bs=1 from
   single-replica measurements — measuring directly would close the gap
   between "theoretical" and "measured" for the cost-per-token claim
   that's now in the docs.

---

## 7. How to commit your work

```bash
cd /opt/dlami/nvme/dsv4-flash-w4a16-fp8-mtp
git checkout -b nvfp4-rtx6000pro    # or work in the sibling repo
# ... your changes ...
git push origin nvfp4-rtx6000pro
```

OR, more correctly: this work belongs in the SIBLING repo
(`canada-quant/dsv4-flash-nvfp4-fp8-mtp`). Clone that fresh and put
the RTX 6000 Pro work there:

```bash
cd /scratch
git clone https://github.com/canada-quant/dsv4-flash-nvfp4-fp8-mtp.git
cd dsv4-flash-nvfp4-fp8-mtp
# Copy whichever scripts apply (see §3 table), adapt for NVFP4, then
# commit + push.
```

When you're done, **update the SIBLING repo's README** with an "Also
runs on RTX PRO 6000 Blackwell" section mirroring the structure of
this repo's README (see §4 "Also runs on RTX PRO 6000 Blackwell" in
our README.md — same shape: per-replica table, node-level extrapolation
with $/token, patches required table, quickstart). And **push README.md
to the HF model card** at `canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`.

---

## 8. If you get stuck

Re-read these in order:

1. `RECIPE_RTX6000PRO.md` — full reproduction recipe, every patch
   explained
2. `benchmarks/rtx6000pro/2026-05-24-cudagraph-summary.md` — what
   "working" looks like (numbers + comparison vs eager mode)
3. `FINDINGS_FOR_SIBLING.md` — the cross-pollination doc between this
   repo and the NVFP4 sibling; covers C13/C14/C15/N1-N4 bugs we filed
4. The actual jasl/vllm source under `~/src/vllm` — when in doubt,
   read the function that's crashing, not the error message

Good luck. If you produce per-replica RTX 6000 Pro NVFP4 numbers that
match or beat our W4A16 numbers (which I think you will, given FP4
density), that closes the loop on "DSv4-Flash with MTP works
end-to-end on Blackwell consumer hardware in two quant flavors."

— W4A16 agent, 2026-05-24
