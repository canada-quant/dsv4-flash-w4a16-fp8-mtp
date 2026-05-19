"""kernel_shim.py — PyTorch reference replacements for the six upstream kernels.

vendor/dsv4-upstream/kernel.py uses tilelang JIT kernels that depend on a
build-time CUDA toolchain we don't want to drag into the calibration path
(and on dtypes — float4_e2m1fn_x2 — that we don't need during a BF16
calibration pass).

This file provides drop-in PyTorch reference replacements for the six names
that ``vendor.dsv4-upstream.model`` imports::

    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn

For calibration purposes:

  - ``sparse_attn`` -> compose F.scaled_dot_product_attention over the
    topk-gathered KV positions.
  - ``hc_split_sinkhorn`` -> hand-rolled softmax+Sinkhorn (the upstream
    kernel is itself a few-iteration row/column normalization).
  - ``act_quant``, ``fp4_act_quant`` -> no-ops in inplace mode (BF16
    passthrough); raise in non-inplace mode (would only be called from the
    FP8/FP4 deployment paths, which calibration doesn't exercise).
  - ``fp8_gemm``, ``fp4_gemm`` -> direct BF16 matmul. Calibration replaces
    upstream's custom Linear with nn.Linear, which routes through these only
    via the load path (never the forward), so these stubs are defensive.

Status: 2026-05-19 — draft only, not exercised end-to-end. The next session
should validate numerical equivalence of ``sparse_attn`` and
``hc_split_sinkhorn`` against a recorded golden output from upstream before
running the 768-sample calibration.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# -------- sparse_attn --------

def sparse_attn(
    q: torch.Tensor,         # [b, s, h, d]
    kv: torch.Tensor,        # [b, kv_len, d_kv]  (kv_len = window + compressed positions)
    attn_sink: torch.Tensor, # [h]   per-head learnable sink bias
    topk_idxs: torch.Tensor, # [b, s, k]  int indices into kv along its 1st dim
    softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention reference implementation.

    Strategy: gather the topk KV rows per query position, expand to per-head,
    score with q @ k^T, add attn_sink as an extra column (softmax over k+1),
    weighted sum the values.
    """
    b, s, h, d = q.shape
    _, kv_len, d_kv = kv.shape
    k = topk_idxs.shape[-1]
    if d != d_kv:
        # Upstream uses MLA-style projections where d_kv == d; if they differ we
        # would need a separate v_proj. Calibration path here uses upstream's
        # MLA so d == d_kv.
        raise ValueError(f"sparse_attn shim: d={d} != d_kv={d_kv}; verify caller")

    # Gather [b, s, k, d_kv] — broadcast topk_idxs into a position index
    idx = topk_idxs.unsqueeze(-1).expand(b, s, k, d_kv)  # [b, s, k, d_kv]
    kv_expand = kv.unsqueeze(1).expand(b, s, kv_len, d_kv)
    gathered = torch.gather(kv_expand, 2, idx)  # [b, s, k, d_kv]

    # Per-head scores: [b, s, h, k]
    # Reshape gathered for matmul: q [b, s, h, d] x k^T [b, s, d, k]
    g_kT = gathered.unsqueeze(2).transpose(-1, -2)  # [b, s, 1, d, k]
    scores = (q.unsqueeze(-2) @ g_kT).squeeze(-2) * softmax_scale  # [b, s, h, k]

    # Append attn_sink as an extra logit column (softmax over k+1, drop the sink output)
    sink_col = attn_sink.view(1, 1, h, 1).expand(b, s, h, 1)
    scores_aug = torch.cat([scores, sink_col], dim=-1)
    attn = F.softmax(scores_aug, dim=-1)[..., :k]  # drop sink mass

    # Weighted sum: [b, s, h, k] @ [b, s, k, d_kv] -> [b, s, h, d]
    out = attn.unsqueeze(-2) @ gathered.unsqueeze(2)  # [b, s, h, 1, d_kv]
    return out.squeeze(-2)


# -------- hc_split_sinkhorn --------

def hc_split_sinkhorn(
    mixes: torch.Tensor,     # [b, s, mix_hc]  where mix_hc = (2 + hc_mult) * hc_mult
    hc_scale: torch.Tensor,  # [3]
    hc_base: torch.Tensor,   # [mix_hc]
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Reference Sinkhorn split for hyper-connection routing.

    The (2 + hc_mult) * hc_mult mix is partitioned into three blocks of sizes
    hc_mult, hc_mult, and hc_mult^2 — to be interpreted as `pre` weights,
    `post` weights, and the `comb` mixing matrix. hc_scale[0..2] are
    temperature scalars; hc_base is an additive bias before splitting.

    The upstream Sinkhorn-normalizes `comb` to be (approximately) doubly
    stochastic. This reference does the same with explicit row-and-column
    softmax iterations.
    """
    mix_hc = (2 + hc_mult) * hc_mult
    assert mixes.shape[-1] == mix_hc
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (mix_hc,)

    biased = mixes + hc_base   # [b, s, mix_hc]
    pre_logits = biased[..., :hc_mult] * hc_scale[0]
    post_logits = biased[..., hc_mult : 2 * hc_mult] * hc_scale[1]
    comb_logits = biased[..., 2 * hc_mult :].reshape(*biased.shape[:-1], hc_mult, hc_mult) * hc_scale[2]

    pre = F.softmax(pre_logits, dim=-1)
    post = F.softmax(post_logits, dim=-1)

    # Sinkhorn: alternating row/col normalization on exp(comb_logits)
    comb = comb_logits.exp().clamp(min=eps)
    for _ in range(sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


# -------- act_quant / fp4_act_quant (no-ops for BF16 calibration) --------

def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt=None,
    scale_dtype=torch.float32,
    inplace: bool = False,
):
    """BF16 calibration: inplace mode is identity (no precision simulation).

    The real upstream kernel does a fused quant+dequant round-trip when
    inplace=True (FP8 QAT simulation). For W4A16 calibration we accept the
    optimistic BF16 activations — the recipe quantizes WEIGHTS, not
    activations, so the calibration target is unaffected. If you later want
    QAT-faithful activations, replace this with an explicit float32 -> FP8
    -> float32 cycle using torch.Tensor.to(float8_e4m3fn) + bf16 cast.
    """
    if inplace:
        return x
    raise NotImplementedError(
        "kernel_shim.act_quant non-inplace path is not used during W4A16 "
        "calibration. Reach the upstream FP8 deployment by using "
        "vendor/dsv4-upstream/kernel.py directly."
    )


def fp4_act_quant(x: torch.Tensor, block_size: int = 32, inplace: bool = False):
    """Same rationale as ``act_quant``. Calibration doesn't exercise FP4 activations."""
    if inplace:
        return x
    raise NotImplementedError("kernel_shim.fp4_act_quant non-inplace path is not used during calibration")


# -------- fp8_gemm / fp4_gemm (calibration uses nn.Linear; defensive stubs) --------

def fp8_gemm(*args, **kwargs):
    raise NotImplementedError(
        "kernel_shim.fp8_gemm should not be called during W4A16 calibration: "
        "we substitute nn.Linear for upstream's FP-aware Linear before any "
        "matmul, so the FP8 gemm path is never reached. If you hit this, "
        "the upstream Linear replacement was missed somewhere."
    )


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError(
        "kernel_shim.fp4_gemm should not be called during W4A16 calibration "
        "(same reason as fp8_gemm)."
    )
