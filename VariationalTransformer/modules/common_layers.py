# -*- coding: utf-8 -*-

import torch
from torch import nn
if hasattr(nn, "GELU"):
    from torch.nn import GELU
else:
    from .gelu import GELU

from .multihead_attention import MultiheadAttention
from .linear import BatchLinear
from utils import xavier_uniform_fan_in_, shared_module


class StraightSum(nn.Module):
    def __init__(self, num_feats, feat_dim):
        super(StraightSum, self).__init__()
        self.num_feats = num_feats
        self.feat_dim = feat_dim

    def forward(self, input):
        input = input.view(*input.shape[:-1], self.num_feats, self.feat_dim)
        return input.sum(dim=-2, keepdim=False)


class MergeUnit(nn.Module):
    def __init__(self, num_feats, feat_dim, weight_activation):
        super(MergeUnit, self).__init__()

        assert weight_activation in ("sigmoid", "softmax")
        self.num_feats = num_feats
        self.feat_dim = feat_dim
        self.act = weight_activation

        if num_feats >= 2:
            self.batch_linear = BatchLinear(num_feats, feat_dim, feat_dim, bias=False)
            if num_feats > 2:
                self.linear = nn.Linear(num_feats * feat_dim, num_feats * feat_dim, bias=False)
            else:
                self.linear = nn.Linear(num_feats * feat_dim, feat_dim, bias=False)
            self.layer_norm = nn.LayerNorm(feat_dim)

            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.linear.weight, activation="tanh")

    # input: [*, num_feats * feat_dim], which can be viewed as [*, num_feats, feat_dim]
    def forward(self, input):
        assert input.shape[-1] == self.num_feats * self.feat_dim
        if self.num_feats == 1:
            return input
        extra_dims = input.shape[:-1]
        input = input.view(*extra_dims, self.num_feats, self.feat_dim)
        output = torch.tanh(self.batch_linear(input))
        output = self.linear(output.view(*extra_dims, -1))
        if self.num_feats > 2:
            output = output.view(*extra_dims, self.num_feats, self.feat_dim)
            if self.act == "softmax":
                weights = torch.softmax(output, dim=-2)
            else:
                weights = torch.sigmoid(output)
            output = (weights * input).sum(-2, keepdim=False)
        else:
            a = torch.sigmoid(output)
            inp1, inp2 = torch.unbind(input, dim=-2)
            output = a * inp1 + (1 - a) * inp2
        return self.layer_norm(output)


class GMU(nn.Module):
    def __init__(self, num_feats, feat_dim, use_softmax=False):
        super(GMU, self).__init__()

        self.num_feats = num_feats
        self.feat_dim = feat_dim
        self.use_softmax = use_softmax

        if num_feats >= 2:
            self.batch_linear = BatchLinear(num_feats, feat_dim, feat_dim, bias=False)
            if num_feats > 2:
                self.linear = nn.Linear(num_feats * feat_dim, num_feats * feat_dim, bias=False)
            else:
                self.linear = nn.Linear(num_feats * feat_dim, feat_dim, bias=False)
            self.layer_norm = nn.LayerNorm(feat_dim)

            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.linear.weight, activation="tanh")

    # input: [*, num_feats * feat_dim], which can be viewed as [*, num_feats, feat_dim]
    def forward(self, input):
        assert input.shape[-1] == self.num_feats * self.feat_dim
        if self.num_feats == 1:
            return input
        extra_dims = input.shape[:-1]
        input = input.view(*extra_dims, self.num_feats, self.feat_dim)
        transformed = torch.tanh(self.batch_linear(input))
        output = self.linear(transformed.view(*extra_dims, -1))
        if self.num_feats > 2:
            output = output.view(*extra_dims, self.num_feats, self.feat_dim)
            if self.use_softmax:
                weights = torch.softmax(output, dim=-2)
            else:
                weights = torch.sigmoid(output)
            output = (weights * transformed).sum(-2, keepdim=False)
        else:
            a = torch.sigmoid(output)
            inp1, inp2 = torch.unbind(transformed, dim=-2)
            output = a * inp1 + (1 - a) * inp2
        return self.layer_norm(output)


# implemented based on paper "GATED MULTIMODAL UNITS FOR INFORMATION FUSION"
class GMUORI(nn.Module):
    def __init__(self, num_feats, feat_dim, use_softmax=False):
        super(GMUORI, self).__init__()

        self.num_feats = num_feats
        self.feat_dim = feat_dim
        self.use_softmax = use_softmax

        if num_feats >= 2:
            self.batch_linear = BatchLinear(num_feats, feat_dim, feat_dim, bias=False)
            if num_feats > 2:
                self.linear = nn.Linear(num_feats * feat_dim, num_feats * feat_dim, bias=False)
            else:
                self.linear = nn.Linear(num_feats * feat_dim, feat_dim, bias=False)
            self.layer_norm = nn.LayerNorm(feat_dim)

            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.linear.weight, activation="tanh")

    # input: [*, num_feats * feat_dim], which can be viewed as [*, num_feats, feat_dim]
    def forward(self, input):
        assert input.shape[-1] == self.num_feats * self.feat_dim
        if self.num_feats == 1:
            return input
        extra_dims = input.shape[:-1]
        input = input.view(*extra_dims, self.num_feats, self.feat_dim)
        transformed = torch.tanh(self.batch_linear(input))
        weights = self.linear(input.view(*extra_dims, -1))
        if self.num_feats > 2:
            weights = weights.view(*extra_dims, self.num_feats, self.feat_dim)
            if self.use_softmax:
                weights = torch.softmax(weights, dim=-2)
            else:
                weights = torch.sigmoid(weights)
            output = (weights * transformed).sum(-2, keepdim=False)
        else:
            a = torch.sigmoid(weights)
            inp1, inp2 = torch.unbind(transformed, dim=-2)
            output = a * inp1 + (1 - a) * inp2
        return self.layer_norm(output)


class TanhFusion(nn.Module):
    def __init__(self, in_features, out_features):
        super(TanhFusion, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.in_proj = nn.Linear(in_features, out_features, bias=False)
        self.out_proj = nn.Linear(out_features, out_features, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.in_proj.weight)
        xavier_uniform_fan_in_(self.out_proj.weight, activation="tanh")

    def forward(self, concatenated=None, *inputs):
        if concatenated is None:
            concatenated = torch.cat(inputs, dim=-1)
        assert concatenated.size(-1) == self.in_features
        return self.out_proj(torch.tanh(self.in_proj(concatenated)))


class FeedForward(nn.Module):
    def __init__(self, input_size, filter_size, output_size, use_gelu, dropout):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.output_size = output_size
        self.use_gelu = use_gelu
        self.dropout = dropout

        self.in_proj = nn.Linear(input_size, filter_size)
        self.act = GELU() if use_gelu else nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(filter_size, output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        xavier_uniform_fan_in_(self.out_proj.weight, activation="relu")
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input):
        return self.out_proj(self.drop(self.act(self.in_proj(input))))


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 num_self_attn_layers,
                 merge_approach,
                 share_self_attn,
                 use_gelu=False,
                 prebuilt_SA=None,
                 prebuilt_LN=None,
                 prebuilt_FF=None,
                 prebuilt_FFLN=None,
                 prebuilt_MG=None):
        super(TransformerLayer, self).__init__()

        self.num_self_attn_layers = num_self_attn_layers
        self.merge_approach = merge_approach
        self.use_gelu = use_gelu

        if prebuilt_SA is not None:
            if share_self_attn:
                assert shared_module(*prebuilt_SA)
            self.self_attn_layers = prebuilt_SA
        else:
            if share_self_attn:
                self.self_attn_layers = nn.ModuleList([MultiheadAttention(d_model, nhead, dropout=dropout)] * num_self_attn_layers)
            else:
                self.self_attn_layers = nn.ModuleList([MultiheadAttention(d_model, nhead, dropout=dropout) for _ in range(num_self_attn_layers)])
        self.layer_norm_layers = prebuilt_LN
        if prebuilt_LN is None:
            self.layer_norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_self_attn_layers)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_self_attn_layers)])

        if num_self_attn_layers == 1:
            self.merge_layer = nn.Identity()
        elif prebuilt_MG is not None:
            self.merge_layer = prebuilt_MG
        else:
            if merge_approach == "sum":
                merge_layer = StraightSum(num_self_attn_layers, d_model)
            elif merge_approach == "tanh":
                merge_layer = TanhFusion(num_self_attn_layers * d_model, d_model)
            elif merge_approach == "merge_unit":
                merge_layer = MergeUnit(num_self_attn_layers, d_model, weight_activation="softmax")
            elif merge_approach == "merge_unit_sigmoid":
                merge_layer = MergeUnit(num_self_attn_layers, d_model, weight_activation="sigmoid")
            elif merge_approach == "gmu":
                merge_layer = GMU(num_self_attn_layers, d_model)
            elif merge_approach == "gmu_softmax":
                merge_layer = GMU(num_self_attn_layers, d_model, use_softmax=True)
            elif merge_approach == "gmu_ori":
                merge_layer = GMUORI(num_self_attn_layers, d_model)
            elif merge_approach == "gmu_ori_softmax":
                merge_layer = GMUORI(num_self_attn_layers, d_model, use_softmax=True)
            else:
                raise ValueError("Unknown merge approach: {}".format(merge_approach))
            self.merge_layer = nn.Sequential(merge_layer, nn.LayerNorm(d_model))

        self.feedforward = prebuilt_FF
        if prebuilt_FF is None:
            self.feedforward = FeedForward(d_model, dim_feedforward, d_model, use_gelu, dropout)
        self.feedforward_ln = prebuilt_FFLN
        if prebuilt_FFLN is None:
            self.feedforward_ln = nn.LayerNorm(d_model)
        self.feedforward_drop = nn.Dropout(dropout)

    def forward(self, input, attn_mask=None, key_padding_mask=None):
        if attn_mask is None or isinstance(attn_mask, torch.Tensor):
            attn_masks = (attn_mask,) * self.num_self_attn_layers
        else:
            attn_masks = list(attn_mask)
            assert len(attn_masks) == self.num_self_attn_layers

        output_list = []
        for i in range(self.num_self_attn_layers):
            attn_output = self.self_attn_layers[i](input, input, input,
                                                   attn_mask=attn_masks[i],
                                                   key_padding_mask=key_padding_mask)[0]
            output = self.layer_norm_layers[i](input + self.dropout_layers[i](attn_output))
            output_list.append(output)
        output = self.merge_layer(torch.cat(output_list, dim=-1))

        feedforward_out = self.feedforward(output)
        output = self.feedforward_ln(output + self.feedforward_drop(feedforward_out))
        return output

    def get_module_by_name(self, name):
        name2realname = {"FF": "feedforward",
                         "SA": "self_attn_layers",
                         "LN": "layer_norm_layers",
                         "FFLN": "feedforward_ln",
                         "MG": "merge_layer"}
        return getattr(self, name2realname[name])


class LatentLayer(nn.Module):
    def __init__(self, input_dim, latent_dim, use_tanh):
        super(LatentLayer, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_tanh = use_tanh
        self.is_normal_distribution = input_dim == 0

        if not self.is_normal_distribution:
            self.hidden2latent = nn.Linear(input_dim, 2 * latent_dim)
            self.latent_activation = nn.Tanh() if use_tanh else nn.Identity()

            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.hidden2latent.weight)
        nn.init.zeros_(self.hidden2latent.bias)

    # cond: [*, batch_size, cond_dim]
    # hidden: [*, batch_size, hidden_size]
    # normal_vector: [*, sample_n, batch_size, latent_dim]
    def forward(self, input=None, normal_vector=None, sample_n=1, reparam=True, **kwargs):
        if input is None:
            head_dims = kwargs["head_dims"]
            batch_size = kwargs["batch_size"]
            dtype = kwargs["dtype"]
            device = kwargs["device"]
        else:
            *head_dims, batch_size, _ = input.size()
            dtype = input.dtype
            device = input.device
        assert sample_n >= 1

        if self.is_normal_distribution:
            mu = log_var = torch.zeros(*head_dims, batch_size, self.latent_dim, dtype=dtype, device=device)
        else:
            mu, log_var = self.latent_activation(self.hidden2latent(input)).chunk(2, dim=-1)
        latent_vector = None
        if reparam:
            if normal_vector is None:
                normal_vector = torch.randn((*head_dims, sample_n, batch_size, self.latent_dim),
                                            dtype=dtype, device=device)
            else:
                normal_vector = normal_vector.view(*head_dims, sample_n, batch_size, self.latent_dim)
            latent_vector = mu.unsqueeze(-3) + torch.exp(0.5 * log_var).unsqueeze(-3) * normal_vector
        return mu, log_var, latent_vector


class LogitsMaskLayer(nn.Module):
    def __init__(self, word2syllables, SEP_token, UNK_token, KEYWORD_token):
        super(LogitsMaskLayer, self).__init__()
        assert word2syllables is not None
        word2syllables = torch.as_tensor(word2syllables, dtype=torch.long)
        self.register_buffer("word2syllables", word2syllables)
        pattern = torch.as_tensor([5, 7, 5, 7, 7, 0], dtype=torch.long)
        self.register_buffer("pattern", pattern)
        self.SEP_token = SEP_token
        self.UNK_token = UNK_token
        self.KEYWORD_token = KEYWORD_token
        self.cache = ()

    def forward(self,
                logits,
                remain_syllables=None,
                use_cache=False,
                decoder_input=None,
                solve_ktoken=False,
                keyword_ids=None,
                sample_n_to_check=1):

        if remain_syllables is None:
            assert decoder_input is not None
            if (decoder_input == self.UNK_token).any():
                print("Found unk in decoder input when calculating remain syllables. This could cause wrong results.")
            if use_cache:
                remain_syllables = self._update_remain_syllables(decoder_input)
            else:
                remain_syllables = self.get_remain_syllables(decoder_input)

        seq_len, batch_size = remain_syllables.size()
        assert tuple(logits.shape[:2]) == (seq_len, sample_n_to_check * batch_size)
        vocab_size = logits.size(-1)

        if solve_ktoken:
            assert keyword_ids is not None
            word2syl = self.word2syllables.unsqueeze(0).repeat(batch_size, 1)
            word2syl[:, self.KEYWORD_token] = self.word2syllables[keyword_ids]
            word2syl = word2syl.unsqueeze(0).expand(seq_len, -1, -1)
        else:
            word2syl = self.word2syllables.view(1, 1, -1).expand(seq_len, batch_size, -1)

        word_mask = word2syl > remain_syllables.unsqueeze(-1).expand(-1, -1, vocab_size)
        logits = logits.view(seq_len, sample_n_to_check, batch_size, vocab_size)
        logits = logits.masked_fill(word_mask.unsqueeze(1), float("-inf"))
        logits = logits.view(seq_len, sample_n_to_check * batch_size, vocab_size)

        return logits

    def get_remain_syllables(self, decoder_input):
        batch_size = decoder_input.size(1)
        remain_syllables = torch.zeros_like(decoder_input)
        cur_segment = torch.zeros(batch_size, dtype=torch.long, device=decoder_input.device)
        arange_seq = torch.arange(batch_size, dtype=torch.long, device=decoder_input.device)
        remain_syllables[0] = self.pattern[0]
        indexes = arange_seq[decoder_input[0] == self.SEP_token]
        cur_segment[indexes] += 1
        remain_syllables[0][indexes] = self.pattern[1]
        for i in range(1, decoder_input.size(0)):
            remain_syllables[i] = remain_syllables[i-1] - self.word2syllables[decoder_input[i]]
            remain_syllables[i][remain_syllables[i] < 0] = 0
            indexes = arange_seq[decoder_input[i] == self.SEP_token]
            cur_segment[indexes] += 1
            cur_segment[cur_segment > 5] = 5
            remain_syllables[i][indexes] = self.pattern[cur_segment[indexes]]
        return remain_syllables

    # input_step: [batch_size] or [seq_len, batch_size], only update remain_syllables from the last step
    def _update_remain_syllables(self, input_step):
        if input_step.dim() == 2:
            input_step = input_step[-1]
        batch_size = input_step.size(0)
        if len(self.cache) == 0:
            remain_syllables = torch.full_like(input_step, self.pattern[0]).unsqueeze(0)
            cur_segment = torch.zeros(batch_size, dtype=torch.long, device=input_step.device)
            arange_seq = torch.arange(batch_size, dtype=torch.long, device=input_step.device)
            indexes = arange_seq[input_step == self.SEP_token]
            cur_segment[indexes] += 1
            remain_syllables[0][indexes] = self.pattern[1]
        else:
            last_remain_syllables, cur_segment, arange_seq = self.cache
            remain_syllables_step = last_remain_syllables[-1] - self.word2syllables[input_step]
            remain_syllables_step[remain_syllables_step < 0] = 0
            indexes = arange_seq[input_step == self.SEP_token]
            cur_segment[indexes] += 1
            cur_segment[cur_segment > 5] = 5
            remain_syllables_step[indexes] = self.pattern[cur_segment[indexes]]
            remain_syllables = torch.cat([last_remain_syllables, remain_syllables_step.unsqueeze(0)], dim=0)
        self.cache = (remain_syllables, cur_segment, arange_seq)
        return remain_syllables

    def clear_cache(self):
        del self.cache
        self.cache = ()
