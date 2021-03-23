# -*- coding:utf-8 -*-

import os
import numpy as np
import random
import math
import torch
from torch import nn
from torch.nn import functional
from itertools import chain
import time

from tools.jap_kana_tools import count_moras


def timedelta(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def ensure_deterministic(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# no need to scale for dropout, because in pytorch dropout is implemented as inverted mode (vanilla / inverted)
def xavier_uniform_fan_in_(weight, activation="linear", *args):
    assert weight.dim() == 2
    fan_in = weight.size(-1)
    scale = math.sqrt(3 / fan_in) * nn.init.calculate_gain(activation, *args)
    nn.init.uniform_(weight, -scale, scale)


def cal_amount_of_params(module, summary=True):
    assert isinstance(module, nn.Module)
    if summary:
        print("Module Structure:\n{}".format(module))
    amount = sum(p.numel() for p in module.parameters())
    print("Amount of paramaters:", amount)
    return amount


# check if tensors have the same storage
def same_storage(*tensors):
    return len(set(t.data_ptr() for t in tensors)) == 1 and len(set(tuple(t.size()) for t in tensors)) == 1


# check if modules share parameters
def shared_module(*modules):
    module_params_list = [list(m.parameters()) for m in modules]
    if len(set(len(t) for t in module_params_list)) > 1:
        return False
    for pms in zip(*module_params_list):
        if not same_storage(*pms):
            return False
    return True


def make_word2syllable(word2id, special_tokens):
    word2sylnum = {k: count_moras(k) for k in word2id.keys()}
    for k in special_tokens:
        word2sylnum[k] = 0
    word2sylnum["<UNK>"] = max(word2sylnum.values()) + 1
    id2word = sorted(word2id.keys(), key=word2id.get)
    word2sylnum = [word2sylnum[id2word[i]] for i in range(len(id2word))]
    return word2sylnum


# logits : [seq_len, batch_size, vocab_size]
# tgt : [seq_len, batch_size]
# mask : [seq_len, batch_size]
def getPPL(logits, tgt, mask=None, return_all=False):
    if mask is not None:
        assert mask.shape == tgt.shape
        assert mask.dtype == torch.bool
    with torch.no_grad():
        log_probs = functional.log_softmax(logits, dim=-1)
        out = torch.gather(log_probs, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1).neg()
        if mask is not None:
            mask = mask.to(logits.dtype)
            out = (out * mask).sum(0) / (mask.sum(0) + 1e-12)
        else:
            out = out.mean(0)
        ppl = out.exp()
    if not return_all:
        return ppl.mean()
    return ppl


def generate_triangle_attention_mask(seq_len, device="cuda", dtype=torch.bool):
    assert dtype in (torch.bool, torch.float16, torch.float32, torch.float64)
    if dtype == torch.bool:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device), diagonal=1)


# segment_ids [seq_len, batch_size]
def generate_segment_attention_mask(segment_ids, triangle=False, dtype=torch.bool):
    assert dtype in (torch.bool, torch.float16, torch.float32, torch.float64)
    seq_len, batch_size = segment_ids.size()
    device = segment_ids.device
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
    segment_ids = segment_ids.t()
    mask[segment_ids.unsqueeze(1) == segment_ids.unsqueeze(2)] = 1
    if triangle:
        mask = torch.triu(mask).transpose(1, 2)
    if dtype == torch.bool:
        return mask.contiguous().logical_not_()
    attn_mask = torch.full((batch_size, seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    attn_mask[mask] = 0.
    return attn_mask


# segment_ids [seq_len, batch_size]
def generate_context_attention_mask(segment_ids, context_range=(0, 0), triangle=False, dtype=torch.bool):
    assert len(context_range) == 2
    assert dtype in (torch.float16, torch.float32, torch.float64), "dtype must be float type"
    seq_len, batch_size = segment_ids.size()
    device = segment_ids.device
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=torch.bool)
    segment_ids = segment_ids.t()
    delta = segment_ids.unsqueeze(1) - segment_ids.unsqueeze(2)
    mask[(delta >= context_range[0]) & (delta <= context_range[1])] = 1
    if triangle:
        mask = torch.triu(mask).transpose(1, 2)
    if dtype == torch.bool:
        return mask.contiguous().logical_not_()
    attn_mask = torch.full((batch_size, seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    attn_mask[mask] = 0.
    return attn_mask


# partition: the partition of the segment_id set, a list of sets
def generate_attention_mask_by_set_partition(segment_ids, partition, triangle=False, dtype=torch.bool):
    max_id = segment_ids.max().item()
    unique_ids = set(range(max_id + 1))
    partition = [s for s in map(lambda x: x & unique_ids, partition) if len(s) > 0]
    assert sum(len(s) for s in partition) == max_id + 1
    assert set(chain(*partition)) == unique_ids
    partition = [list(s) for s in partition]
    segment_id_to_field_id = torch.zeros(max_id + 1, dtype=torch.long, device=segment_ids.device)
    for i, sub in enumerate(partition):
        segment_id_to_field_id[sub] = i
    field_ids = segment_id_to_field_id[segment_ids]
    return generate_segment_attention_mask(field_ids, triangle=triangle, dtype=dtype)


def bidirectional_padding(batch_seqs, PAD_token, align_pos, left_length=None, right_length=None, batch_first=False, device="cuda"):
    if align_pos is None:
        align_pos = 0
    if isinstance(align_pos, int):
        align_pos = [align_pos] * len(batch_seqs)
    align_pos = [p % len(s) for p, s in zip(align_pos, batch_seqs)]
    if left_length is None:
        left_length = max(align_pos)
    else:
        assert left_length >= max(align_pos)
    min_right_length = max(len(s) - p - 1 for p, s in zip(align_pos, batch_seqs))
    if right_length is None:
        right_length = min_right_length
    else:
        assert right_length >= min_right_length
    seq_len = left_length + right_length + 1
    slices = slice(None, None, 1 if batch_first else -1)
    padded = torch.full((len(batch_seqs), seq_len)[slices], PAD_token, dtype=torch.long, device=device)
    for i, s in enumerate(batch_seqs):
        left_ind = left_length - align_pos[i]
        padded[(i, slice(left_ind, left_ind + len(s)))[slices]] = torch.as_tensor(s, dtype=torch.long, device=device)
    return padded, left_length


def get_segment_ids(input, SEP_token, align_pos=0, relative=True, sep_as_new_segment=False):
    segment_ids = torch.zeros_like(input)
    if sep_as_new_segment:
        segment_ids[0][input[0] == SEP_token] = 1
    for step in range(1, input.size(0)):
        segment_ids[step] = segment_ids[step - 1]
        input_step = input[step] if sep_as_new_segment else input[step - 1]
        segment_ids[step][input_step == SEP_token] += 1
    if relative:
        segment_ids %= 2
        indexes = torch.arange(input.size(1))[segment_ids[align_pos] == 1]
        segment_ids[:, indexes] = 1 - segment_ids[:, indexes]
    return segment_ids


def get_remain_syllables(word2syllable, SEP_token, decoder_input=None, decoder_target=None, check_input=True):
    assert (decoder_input is None) ^ (decoder_target is None)
    input = decoder_input if decoder_target is None else decoder_target
    if check_input:
        if decoder_target is None:
            assert (word2syllable[input[0]] == 0).all()
        else:
            assert (word2syllable[input[-1]] == 0).all()
    if decoder_target is None:
        decoder_target = decoder_input[1:]
    remain_syllables = torch.zeros_like(input)
    prefix_sum_array = torch.cumsum(word2syllable[decoder_target], dim=0)
    minuend = prefix_sum_array[-1].clone()
    tgt_len = decoder_target.size(0)
    for idx in range(2, tgt_len + 1):
        cur_ids = decoder_target[-idx]
        cur_prefix_sum = prefix_sum_array[-idx]
        remain_syllables[tgt_len - idx + 1] = minuend - cur_prefix_sum
        mask = cur_ids == SEP_token
        minuend[mask] = cur_prefix_sum[mask]
    remain_syllables[0] = minuend
    return remain_syllables
