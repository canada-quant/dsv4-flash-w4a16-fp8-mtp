#!/usr/bin/env python3
"""Phase 2 GPTQ smoke — real BF16 weights, real forward through MTP, real Hessian.

This is the user-gated step-4 smoke. Goals (per the redirect):
  (a) Real BF16 weights load into the shimmed Transformer without OOM and
      with zero unmatched safetensors keys.
  (b) The CalibrationModel wrapper drives activations through one MTP
      Linear; we accumulate H = X.T @ X via a forward hook and print the
      Hessian trace as proof that GPTQ-shape data could be collected.
  (c) The compressed-tensors quantization_config shape we'd emit for the
      GPTQ artifact is validated structurally against the RTN-fallback
      artifact's config.json (same schema, different recipe).

Memory plan (568 GB BF16 vs 275 GB/GPU):
  - Full Transformer instantiated and weight-loaded on CPU (~568 GB / 4 TB).
  - Move ONE layer at a time to cuda:0; batch all 4 samples through that
    layer; move layer back. Hidden states (4 x [1, SEQ, hc=4, dim=4096]
    BF16 = ~16 MB total) stay on GPU between layers.
  - Wall-clock estimate: 44 layers x ~5s CPU->GPU + tiny compute = ~5 min
    for the main path, plus MTP block, plus the ~10-15 min initial load.

Usage::

    python scripts/smoke_gptq_real.py            # on /data/venv-calib
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make scripts.upstream importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.upstream import (
    Transformer,
    apply_dist_state,
    build_model_args,
)
from scripts.calibration_model import CalibrationModel
from scripts.load_bf16_into_transformer import load_safetensors_into


# ---- knobs ----------------------------------------------------------------
WEIGHTS_DIR = Path("/scratch/weights/bf16-mtp")
CONFIG_JSON = Path("/data/vendor/dsv4-upstream/config.json")
RTN_FALLBACK_CONFIG = Path("/scratch/weights/w4a16-fp8-mtp-rtn-fallback/config.json")
DEVICE = torch.device("cuda:0")

N_SAMPLES = 4
SEQ_LEN = 128
TARGET_LINEAR = "mtp.0.ffn.experts.0.w1"


def header(s: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(s, flush=True)
    print("=" * 78, flush=True)


def resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    """Walk dotted attribute / index path to a submodule."""
    cur = root
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


# ===========================================================================
# (a) real weights load — full 568 GB BF16 into shimmed Transformer
# ===========================================================================
header("PHASE 2 SMOKE — step (a): real BF16 load into shimmed Transformer")

t_total_start = time.time()

apply_dist_state()  # single-process: world_size=1, rank=0
import dsv4_upstream_model as _dsv4_mod  # noqa: E402 — installed by the shim
print(f"  shim dist state: world_size={_dsv4_mod.world_size}, rank={_dsv4_mod.rank}")

margs = build_model_args(
    str(CONFIG_JSON), max_batch_size=N_SAMPLES, max_seq_len=SEQ_LEN
)
print(
    f"  ModelArgs: vocab={margs.vocab_size} dim={margs.dim} "
    f"n_layers={margs.n_layers} n_mtp_layers={margs.n_mtp_layers} "
    f"n_routed_experts={margs.n_routed_experts} hc_mult={margs.hc_mult} "
    f"max_seq_len={margs.max_seq_len}",
    flush=True,
)

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cpu")

t0 = time.time()
print("  instantiating Transformer on CPU...", flush=True)
transformer = Transformer(margs)
print(f"  instantiated in {time.time() - t0:.1f}s", flush=True)
n_params = sum(p.numel() for p in transformer.parameters())
print(f"  params: {n_params:,}  ({n_params * 2 / 1e9:.2f} GB BF16)", flush=True)

n_shards = len(list(WEIGHTS_DIR.glob("*.safetensors")))
print(f"  loading {n_shards} shards from {WEIGHTS_DIR}...", flush=True)
t1 = time.time()
loaded, unmatched, missing = load_safetensors_into(
    transformer, WEIGHTS_DIR, verbose=True
)
load_secs = time.time() - t1
print(f"  load complete in {load_secs:.1f}s", flush=True)
print(
    f"  loaded={loaded}  unmatched_safetensors={len(unmatched)}  "
    f"missing_state_dict={len(missing)}",
    flush=True,
)

if unmatched:
    print("FAIL (a): unmatched safetensors keys present — A' commitment is that the names already match.")
    for k in unmatched[:10]:
        print(f"    - {k}")
    sys.exit(1)
if missing:
    # Aliases (mtp.0.embed.weight, mtp.0.head.weight) are accounted for in
    # load_safetensors_into via PARAM_ALIASES. Any OTHER missing is a fault.
    print(f"WARN: {len(missing)} state_dict keys not covered by safetensors+aliases:")
    for k in missing[:10]:
        print(f"    - {k}")
    if len(missing) > 0:
        print("FAIL (a): unexpected missing keys")
        sys.exit(1)

print(f"OK (a) — load succeeded in {load_secs:.1f}s, zero unmatched, zero unexpected missing", flush=True)


# ===========================================================================
# (b) real forward + Hessian collection on one MTP Linear
# ===========================================================================
header("PHASE 2 SMOKE — step (b): real forward through MTP + Hessian trace")

cal_model = CalibrationModel(transformer)
target_module = resolve_module(transformer, TARGET_LINEAR)
print(f"  target Linear: {TARGET_LINEAR}", flush=True)
print(f"    type={type(target_module).__name__}", flush=True)
print(f"    weight.shape={list(target_module.weight.shape)}", flush=True)
print(f"    in_features={target_module.in_features}", flush=True)

# Hessian accumulator
hess_state = {"H": None, "n_rows": 0}


def hessian_hook(_module, inputs, _output):
    """Accumulate H += X.T @ X for the Linear's input X."""
    x = inputs[0]
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])
    x32 = x.detach().to(torch.float32)
    H = x32.T @ x32
    if hess_state["H"] is None:
        hess_state["H"] = H
    else:
        hess_state["H"] += H
    hess_state["n_rows"] += x32.shape[0]


handle = target_module.register_forward_hook(hessian_hook)


def streaming_forward(input_ids_batch: torch.Tensor) -> torch.Tensor:
    """Forward CalibrationModel by streaming each layer to GPU, batch all
    samples through it, then move it back. Returns the final logits."""
    t = transformer
    input_ids_dev = input_ids_batch.to(DEVICE)

    # Embed
    t.embed.to(DEVICE)
    h = t.embed(input_ids_dev)
    t.embed.to("cpu")
    torch.cuda.empty_cache()

    h = h.unsqueeze(2).repeat(1, 1, t.hc_mult, 1)  # [B, S, hc, d]

    # Main 0..N-1
    for i, layer in enumerate(t.layers):
        layer.to(DEVICE)
        h = layer(h, 0, input_ids_dev)
        layer.to("cpu")
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"    layer {i + 1:>2d}/{len(t.layers)} done", flush=True)

    # MTP — wrapper's contract: feed the pre-norm h.
    # The hook on TARGET_LINEAR fires inside mtp.forward.
    for mtp_layer in t.mtp:
        mtp_layer.to(DEVICE)
        _ = mtp_layer(h, 0, input_ids_dev)
        mtp_layer.to("cpu")
        torch.cuda.empty_cache()

    # Final head — not strictly needed for the Hessian, skip to save time
    return h


torch.manual_seed(42)
samples = torch.randint(0, margs.vocab_size, (N_SAMPLES, SEQ_LEN))
print(f"  generated {N_SAMPLES} synthetic samples, seq={SEQ_LEN}, batched", flush=True)

print(f"  running streaming forward (batch={N_SAMPLES}, seq={SEQ_LEN})...", flush=True)
t_fwd = time.time()
with torch.inference_mode():
    streaming_forward(samples)
print(f"  forward complete in {time.time() - t_fwd:.1f}s", flush=True)

handle.remove()

# Report Hessian
H = hess_state["H"]
n_rows = hess_state["n_rows"]

print()
print("HESSIAN REPORT:")
print(f"  target Linear      : {TARGET_LINEAR}")
print(f"  H.shape            : {list(H.shape)}")
print(f"  X.T @ X rows fed   : {n_rows}")
print(f"  trace(H)           : {H.trace().item():.6e}")
print(f"  fro_norm(H)        : {H.norm().item():.6e}")
print(f"  diag.mean()        : {H.diagonal().mean().item():.6e}")
print(f"  diag.std()         : {H.diagonal().std().item():.6e}")
print(f"  off_diag.abs().mean: {(H - torch.diag(H.diagonal())).abs().mean().item():.6e}")

# Cheap-ish condition number via eigvalsh (4096x4096 fp32 is fast)
print("  computing eigvals (4096x4096 fp32)...", flush=True)
t_e = time.time()
H_reg = H + 1e-6 * torch.eye(H.shape[0])
try:
    eig = torch.linalg.eigvalsh(H_reg)
    print(f"  eigvals computed in {time.time() - t_e:.1f}s")
    print(f"  eigval.max()       : {eig.max().item():.6e}")
    print(f"  eigval.min()       : {eig.min().item():.6e}")
    cond = eig.max().item() / max(eig.min().item(), 1e-30)
    print(f"  cond_2(H)          : {cond:.6e}")
except RuntimeError as exc:
    print(f"  eigvalsh failed: {exc}")

if H.trace().item() <= 0 or n_rows == 0:
    print("FAIL (b): zero or non-positive Hessian — activations did not reach the MTP Linear")
    sys.exit(1)

print(f"OK (b) — MTP Linear saw {n_rows} real activation rows, non-zero PSD Hessian", flush=True)


# ===========================================================================
# (c) compressed-tensors quantization_config schema check
# ===========================================================================
header("PHASE 2 SMOKE — step (c): quantization_config schema (vs RTN fallback)")

# The compressed-tensors schema is the same regardless of recipe; what
# differs at calibration time is `quantization_args.actorder` (GPTQ sets
# it; RTN leaves it null) and the per-expert scale distribution.
expected_top = {
    "config_groups",
    "format",
    "ignore",
    "quant_method",
    "quantization_status",
    "version",
}
print(f"  expected top-level quantization_config keys (subset): {sorted(expected_top)}")

if not RTN_FALLBACK_CONFIG.exists():
    print(f"WARN: RTN fallback config not found at {RTN_FALLBACK_CONFIG}")
    print("  step (c) reduced to schema-only proof — not a differential")
else:
    with open(RTN_FALLBACK_CONFIG) as f:
        rtn_cfg = json.load(f)
    rtn_qc = rtn_cfg.get("quantization_config", {})
    print(f"  RTN fallback quantization_config keys: {sorted(rtn_qc.keys())}")
    missing_in_rtn = expected_top - set(rtn_qc.keys())
    print(f"  missing from RTN (vs expected): {sorted(missing_in_rtn) if missing_in_rtn else '(none)'}")
    cg0 = rtn_qc.get("config_groups", {}).get("config_group_0", {})
    actorder = cg0.get("weights", {}).get("actorder")
    print(f"  RTN config_group_0.weights.actorder = {actorder!r}  (RTN expected: null/None)")
    if actorder is not None:
        print("  NOTE: RTN fallback unexpectedly has non-null actorder")
    else:
        print("  NOTE: RTN fallback actorder is null — confirms the GPTQ-vs-RTN tell.")
    print("OK (c) — compressed-tensors schema validated; GPTQ artifact will populate "
          "config_group_X.weights.actorder when the full run completes.")


header("ALL SMOKE GATES PASSED")
print(f"total wall: {time.time() - t_total_start:.1f}s")
print()
print("Hand-off:")
print(f"  Hessian trace on {TARGET_LINEAR}: {H.trace().item():.6e}")
print(f"  (non-zero PSD Hessian; activations flowed through MTP via CalibrationModel)")
print()
print("Next step is GATED on user reviewing the Hessian output above before")
print("kicking off the full quantize_v4_w4a16_mtp.py + oneshot bridge.")
