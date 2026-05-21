# Draft: DeepseekV4NextNPredictor.forward + compute_mtp_logits for PR #46127.
#
# This is the proposed expansion for Matt's "structural shell without forward()"
# feedback. Math ported from DeepSeek's release inference/model.py (the only
# pure-PyTorch reference; vLLM's HCHeadOp.forward_native is NotImplementedError).
#
# Files this would patch:
#   src/transformers/models/deepseek_v4/modular_deepseek_v4.py
#   src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
#
# Reference: DeepSeek-V4-Flash/inference/model.py:MTPBlock, ParallelHead.hc_head
#
# Public API on DeepseekV4NextNPredictor (forward) and DeepseekV4Model
# (compute_mtp_logits) lets vLLM / sglang / etc. consume the MTP head without
# transformers presuming a speculation protocol.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ---- Additions to DeepseekV4NextNPredictor (already partially defined) -------


def forward(
    self,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    embed_tokens: nn.Module,
    *,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Single MTP draft-block forward.

    Args:
        hidden_states: ``[B, S, hc_mult, D]`` — the pre-final-norm residual
            stream from the main model's last decoder layer (the hc_mult-stream
            shape used inside :class:`DeepseekV4DecoderLayer`, BEFORE the
            HC-head collapse).
        input_ids: ``[B, S]`` — the token ids the draft head should condition
            on. Matches the main model's input ids (with the position-0
            embedding masked to zero, per DeepSeek's reference).
        embed_tokens: the main model's :attr:`DeepseekV4Model.embed_tokens`
            module. Passed in rather than carried as a ref so the MTP block
            stays usable when the model is sharded or partially loaded.
        position_ids, attention_mask, **kwargs: forwarded to the inherited
            :meth:`DeepseekV4DecoderLayer.forward`.

    Returns:
        ``[B, S, hc_mult, D]`` — the post-decoder residual stream for the
        draft head. Call :meth:`DeepseekV4Model.compute_mtp_logits` or feed
        this to a second MTP step for ``num_speculative_tokens > 1``.

    Math (DeepSeek release, ``inference/model.py:MTPBlock.forward``)::

        e = enorm(embed(input_ids))           # [B,S,D]
        h = hnorm(hidden_states)              # [B,S,hc,D]
        x = e_proj(e).unsqueeze(-2) + h_proj(h)  # broadcast e across hc dim
        x = super().forward(x, input_ids, ...)   # main decoder body
    """
    # MTP draft conditions on input_ids[1:] (shift-left). Position-0 has no
    # prior token to predict from; masking inputs_embeds at position 0
    # mirrors the reference and avoids leaking the BOS into the draft.
    inputs_embeds = embed_tokens(input_ids)
    if position_ids is not None:
        inputs_embeds = torch.where(
            position_ids.unsqueeze(-1) == 0,
            torch.zeros_like(inputs_embeds),
            inputs_embeds,
        )
    e = self.enorm(inputs_embeds)
    h = self.hnorm(hidden_states)
    # e: [B,S,D] -> e_proj -> [B,S,D] -> unsqueeze(-2) -> [B,S,1,D]
    # h: [B,S,hc,D] -> h_proj -> [B,S,hc,D]
    x = self.e_proj(e).unsqueeze(-2) + self.h_proj(h)
    # super().forward is :meth:`DeepseekV4DecoderLayer.forward`, which expects
    # [B,S,hc,D] and returns [B,S,hc,D] after attn_hc + ffn_hc plumbing.
    x = super().forward(
        x,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        **kwargs,
    )
    return x


def hc_head_reduce(self, x: torch.Tensor) -> torch.Tensor:
    """Collapse the hc_mult residual streams to one via sigmoid-gated mixing.

    Pure-PyTorch port of DeepSeek release :meth:`ParallelHead.hc_head`. Same
    math fp_native variant is :class:`HCHeadOp` in vLLM, but vLLM's
    ``forward_native`` is NotImplementedError, so we go directly to the
    reference here.

    Args:
        x: ``[B, S, hc_mult, D]``

    Returns:
        ``[B, S, D]``
    """
    shape, dtype = x.size(), x.dtype
    hc_eps = getattr(self, "hc_eps", 1e-6)
    norm_eps = self.norm.variance_epsilon  # DeepseekV4RMSNorm exposes eps here

    x_flat = x.flatten(-2).float()  # [B, S, hc*D]
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt  # [B, S, hc_mult]
    pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + hc_eps
    # weighted sum over the hc dim
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape).float(), dim=-2)
    return y.to(dtype)


# ---- Addition to DeepseekV4Model: orchestration ------------------------------


def compute_mtp_logits(
    self,
    input_ids: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    lm_head: Optional[nn.Module] = None,
    mtp_index: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one MTP draft step and return ``(logits, post_residual)``.

    Args:
        input_ids: ``[B, S]`` token ids to condition the draft on.
        previous_hidden_states: ``[B, S, hc_mult, D]`` residual stream from
            the main model's last decoder layer (before HC head collapse).
            For ``num_speculative_tokens > 1`` the caller re-feeds the
            ``post_residual`` from the previous MTP step.
        position_ids, attention_mask: forwarded to the MTP block's main
            decoder body. Pass them through so MTP shares the same masking
            and rotary semantics as the main forward.
        lm_head: an ``nn.Linear`` (or equivalent) projecting
            ``[B, S, D] -> [B, S, vocab_size]``. If None, returns ``None``
            for ``logits`` — callers that only need the residual (e.g. to
            chain a second MTP step) can omit the lm_head pass.
        mtp_index: which MTP block in ``self.mtp`` to invoke
            (default 0; DSv4-Flash has one).

    Returns:
        Tuple of:
            - ``logits``: ``[B, S, vocab_size]`` or None.
            - ``post_residual``: ``[B, S, hc_mult, D]`` — feed to the next
              MTP step or discard.
    """
    if not getattr(self, "mtp", None) or len(self.mtp) == 0:
        raise ValueError(
            "compute_mtp_logits called but model has no MTP blocks. "
            "Check that config.num_nextn_predict_layers > 0."
        )
    if mtp_index >= len(self.mtp):
        raise IndexError(
            f"mtp_index={mtp_index} out of range; only "
            f"{len(self.mtp)} MTP block(s) present."
        )

    mtp_block = self.mtp[mtp_index]
    post_residual = mtp_block(
        hidden_states=previous_hidden_states,
        input_ids=input_ids,
        embed_tokens=self.embed_tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        **kwargs,
    )

    if lm_head is None:
        return None, post_residual

    # hc_head collapse + final norm + lm_head
    collapsed = mtp_block.hc_head_reduce(post_residual)
    collapsed = mtp_block.norm(collapsed)
    logits = lm_head(collapsed)
    return logits, post_residual


# ---- Tests (sketch — to live under tests/models/deepseek_v4/) ----------------

TEST_TEMPLATE = '''
# tests/models/deepseek_v4/test_mtp.py

import pytest
import torch
from transformers import DeepseekV4Config, DeepseekV4Model


def _tiny_config():
    """Tiny synthetic config; no weight download needed for CI."""
    return DeepseekV4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        hc_mult=2,
        # ... other required minima
    )


def test_mtp_forward_shape_and_finite():
    """Floor test: synthetic forward → output shape + no NaN."""
    config = _tiny_config()
    model = DeepseekV4Model(config)
    B, S = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    # Need a hc_mult-stream residual from a fake main forward.
    hidden = torch.randn(B, S, config.hc_mult, config.hidden_size)
    logits, post = model.compute_mtp_logits(
        input_ids,
        hidden,
        lm_head=torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False),
    )
    assert logits.shape == (B, S, config.vocab_size)
    assert post.shape == (B, S, config.hc_mult, config.hidden_size)
    assert torch.isfinite(logits).all()


def test_mtp_weight_round_trip(tmp_path):
    """from_pretrained → save_pretrained preserves mtp.0.* keys bit-identical."""
    config = _tiny_config()
    model = DeepseekV4Model(config)
    # Mark a sentinel MTP weight
    with torch.no_grad():
        model.mtp[0].e_proj.weight.fill_(0.42)
    model.save_pretrained(str(tmp_path))
    reloaded = DeepseekV4Model.from_pretrained(str(tmp_path))
    assert torch.equal(reloaded.mtp[0].e_proj.weight, model.mtp[0].e_proj.weight)


def test_mtp_hc_head_matches_reference():
    """HC math equivalence: our hc_head_reduce matches DeepSeek release math."""
    config = _tiny_config()
    model = DeepseekV4Model(config)
    torch.manual_seed(42)
    x = torch.randn(2, 4, config.hc_mult, config.hidden_size)
    # Reference: paste DeepSeek release ParallelHead.hc_head body
    def reference(x_, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
        shape, dtype = x_.size(), x_.dtype
        x_ = x_.flatten(-2).float()
        rsqrt = torch.rsqrt(x_.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(x_, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
        return torch.sum(pre.unsqueeze(-1) * x_.view(shape).float(), dim=-2).to(dtype)
    block = model.mtp[0]
    ours = block.hc_head_reduce(x)
    ref = reference(
        x,
        block.hc_head_fn,
        block.hc_head_scale,
        block.hc_head_base,
        block.norm.variance_epsilon,
        getattr(block, "hc_eps", 1e-6),
    )
    assert torch.allclose(ours, ref, atol=1e-5), \\
        f"hc_head_reduce drift: max diff = {(ours - ref).abs().max()}"


def test_no_mtp_path_unchanged():
    """num_nextn_predict_layers=0 doesn't change main forward behavior."""
    config = _tiny_config()
    config.num_nextn_predict_layers = 0
    model = DeepseekV4Model(config)
    # MTP block list is empty
    assert len(model.mtp) == 0
    # compute_mtp_logits raises clearly
    with pytest.raises(ValueError, match="no MTP blocks"):
        model.compute_mtp_logits(
            torch.zeros(1, 4, dtype=torch.long),
            torch.zeros(1, 4, config.hc_mult, config.hidden_size),
        )
'''
