# -*- coding:utf-8 -*-
# Code adapted from https://github.com/pytorch/pytorch/blob/v1.6.0/torch/nn/functional.py

import torch
from torch import nn
from torch.nn import functional
import warnings

from utils import xavier_uniform_fan_in_


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.in_proj.weight)
        xavier_uniform_fan_in_(self.out_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    # key_padding_mask: [N, S]
    # attn_mask: [L, S] or [N, L, S]
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        return multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj.weight, self.in_proj.bias,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask)


def multi_head_attention_forward(query,
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 num_heads,
                                 in_proj_weight,
                                 in_proj_bias,
                                 dropout_p,
                                 out_proj_weight,
                                 out_proj_bias,
                                 training=True,
                                 key_padding_mask=None,
                                 need_weights=True,
                                 attn_mask=None):
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if torch.equal(query, key) and torch.equal(key, value):
        # self-attention
        q, k, v = functional.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

    elif torch.equal(key, value):
        # encoder-decoder attention
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = functional.linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        k, v = functional.linear(key, _w, _b).chunk(2, dim=-1)

    else:
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = functional.linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = functional.linear(key, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = functional.linear(value, _w, _b)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) == [bsz, query.size(0), key.size(0)]:
                attn_mask = torch.repeat_interleave(attn_mask, num_heads, dim=0)
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = functional.softmax(attn_output_weights, dim=-1)
    attn_output_weights = functional.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = functional.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        return attn_output, attn_output_weights
    else:
        return attn_output, None
