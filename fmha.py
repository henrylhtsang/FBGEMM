#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
    cutlass_blackwell_fmha_func,
)

DEBUG = False
SEED = 1234


# ============================================================================
# Inlined from bert_padding.py
# ============================================================================


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (
        (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    )
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


# ============================================================================
# Inlined from test_utils.py
# ============================================================================


def generate_random_padding_mask(
    max_seqlen, batch_size, device, mode="random", zero_lengths=False
):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full(
            (batch_size, 1), max_seqlen, device=device, dtype=torch.int32
        )
    elif mode == "random":
        lengths = torch.randint(
            max(0 if zero_lengths else 1, max_seqlen - 20),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )
    elif mode == "third":
        lengths = torch.randint(
            max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device
        )

    if zero_lengths:
        # Generate zero-lengths every 5 batches and the last batch.
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size)
        < lengths
    )
    return padding_mask


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    kvpacked=False,
    qkvpacked=False,
    add_unused_qkv=False,
    query_unused_mask=None,
    key_unused_mask=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)
    if query_unused_mask is not None or key_unused_mask is not None:
        assert not kvpacked
        assert not qkvpacked

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, seqused_q = unpad_input(
            q,
            query_padding_mask,
            query_unused_mask,
        )
        output_pad_fn = lambda output_unpad: pad_input(  # noqa
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q_unpad.device,
        )
        seqused_q = None
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(  # noqa
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, seqused_k = unpad_input(
            k, key_padding_mask, key_unused_mask
        )
        v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask, key_unused_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=k_unpad.device,
        )
        seqused_k = None
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(  # noqa
                dqkv_unpad, indices_q, batch_size, seqlen_q
            )
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(  # noqa
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(  # noqa
                dkv_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(  # noqa
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(  # noqa
                dk_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(  # noqa
                dk_unpad, "(b s) h d -> b s h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk

    if window_size[0] < 0 and window_size[1] < 0:
        # Both windows infinite - no local masking
        return torch.zeros_like(col_idx, dtype=torch.bool)
    elif window_size[0] < 0:
        # Left infinite, right finite - mask right only
        return col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk - 1)
    elif window_size[1] < 0:
        # Left finite, right infinite - mask left only
        return col_idx < row_idx + sk - sq - window_size[0]
    else:
        # Both finite - mask both sides
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk - 1),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(  # noqa
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
    softmax_scale=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
        softmax_scale: float, scale for softmax. If None, use 1/sqrt(head_dim)
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    # Use provided softmax_scale or default to 1/sqrt(d)
    scale = softmax_scale if softmax_scale is not None else 1 / math.sqrt(d)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * scale)
    if softcap > 0:
        scores /= softcap
        scores = scores.tanh()
        scores *= softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
            key_leftpad=key_leftpad,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    if key_padding_mask is not None:
        output.masked_fill_(
            rearrange(
                torch.logical_not(torch.any(key_padding_mask, 1)), "b -> b 1 1 1"
            ),
            0.0,
        )
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def _generate_qkv(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        batch_size,
        seqlen_q,
        q_heads,
        head_dim,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    k, v = (
        torch.randn(
            batch_size,
            seqlen_k,
            kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        for _ in range(2)
    )
    return q, k, v


def _execute_cutlass_blackwell_attn_varlen(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    window_size: tuple[int, int],
    fwd_only: bool,
    sm_scale: Optional[float],
) -> None:
    device = torch.accelerator.current_accelerator()
    assert device is not None

    torch.manual_seed(SEED)

    q_ref, k_ref, v_ref = _generate_qkv(
        batch_size,
        seqlen_q,
        seqlen_k,
        q_heads,
        kv_heads,
        head_dim,
        device,
        dtype,
    )

    q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, device, mode="random", zero_lengths=False
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch_size, device, mode="random", zero_lengths=False
    )

    # Always have seqlen_k >= seqlen_q
    key_padding_mask[:, :seqlen_q] |= query_padding_mask

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        _,
        _,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        kvpacked=False,
        query_unused_mask=None,
        key_unused_mask=None,
    )

    # Run attention forwards
    out_ref, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        window_size=window_size,
        softmax_scale=sm_scale,
    )

    out_pt, _ = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True,
        softmax_scale=sm_scale,
    )

    out_unpad = cutlass_blackwell_fmha_func(
        q_unpad,
        k_unpad,
        v_unpad,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seq_len_q=max_seqlen_q,
        max_seq_len_k=max_seqlen_k,
        page_table=None,
        window_size=window_size,
        deterministic=False,
        softmax_scale=sm_scale,
    )
    out = output_pad_fn(out_unpad)

    g_unpad = torch.randn_like(out_unpad)
    dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
        out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad
    )
    # dq = dq_pad_fn(dq_unpad)
    # dk = dk_pad_fn(dk_unpad)
    # dv = dk_pad_fn(dv_unpad)

    g = output_pad_fn(g_unpad)

    # this is the bad line
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)


def main():
    seqlen = 8
    kv_heads = 2
    dtype = torch.float16
    head_dim = 128
    q_heads = kv_heads  # q_heads_per_kv_head = 1

    # Run tests with different batch sizes
    for batch_size in [1, 37]:
        _execute_cutlass_blackwell_attn_varlen(
            batch_size,
            seqlen,
            seqlen,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            causal=False,
            window_size=(-1, -1),
            fwd_only=False,
            sm_scale=None,
        )

    print("done")


if __name__ == "__main__":
    main()
