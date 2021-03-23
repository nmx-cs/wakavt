# -*- coding: utf-8 -*-

import torch
from torch import nn
from utils import xavier_uniform_fan_in_


def get_activation_layer(activation):
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Unknown activation: {}".format(activation))


class MLP3(nn.Module):
    def __init__(self, input_size, filter_size, output_size, activation, dropout):
        super(MLP3, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.output_size = output_size
        self.activation = activation
        self.dropout = dropout

        self.in_proj = nn.Linear(input_size, filter_size)
        self.act = get_activation_layer(activation)
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(filter_size, output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        xavier_uniform_fan_in_(self.out_proj.weight, activation=self.activation)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input):
        return self.out_proj(self.drop(self.act(self.in_proj(input))))


class LatentLayer(nn.Module):
    def __init__(self, input_dim, mlp_size, latent_dim, use_tanh):
        super(LatentLayer, self).__init__()

        self.input_dim = input_dim
        self.mlp_sie = mlp_size
        self.latent_dim = latent_dim
        self.use_tanh = use_tanh
        self.is_normal_distribution = input_dim == 0

        if not self.is_normal_distribution:
            self.hidden2latent = MLP3(input_dim, mlp_size, 2 * latent_dim, activation="leaky_relu", dropout=0.)
            self.latent_activation = nn.Tanh() if use_tanh else nn.Identity()

            self._reset_parameters()

    def _reset_parameters(self):
        return

    # input: [*, batch_size, input_dim]
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
    def __init__(self, word2syllables, SEP_token, UNK_token):
        super(LogitsMaskLayer, self).__init__()
        assert word2syllables is not None
        word2syllables = torch.as_tensor(word2syllables, dtype=torch.long)
        self.register_buffer("word2syllables", word2syllables)
        pattern = torch.as_tensor([5, 7, 5, 7, 7, 0], dtype=torch.long)
        self.register_buffer("pattern", pattern)
        self.SEP_token = SEP_token
        self.UNK_token = UNK_token
        self.cache = ()

    def forward(self,
                logits,
                remain_syllables=None,
                use_cache=False,
                decoder_input=None,
                sample_n_to_check=1,
                only_last_step=False):
        if remain_syllables is None:
            assert decoder_input is not None and decoder_input.dim() == 2
            if (decoder_input == self.UNK_token).any():
                print("Found unk in decoder input when calculating remain syllables. This could cause wrong results.")
            if use_cache:
                remain_syllables = self._update_remain_syllables(decoder_input)
            else:
                remain_syllables = self.get_remain_syllables(decoder_input)

        vocab_size = logits.size(-1)
        seq_len, batch_size = remain_syllables.size()
        if only_last_step:
            assert logits.size(0) == sample_n_to_check * batch_size
            remain_syllables = remain_syllables[-1]
            word2syl = self.word2syllables.view(1, -1).expand(batch_size, -1)
            word_mask = word2syl > remain_syllables.unsqueeze(-1).expand(-1, vocab_size)
            logits = logits.view(sample_n_to_check, batch_size, vocab_size)
            logits = logits.masked_fill(word_mask.unsqueeze(0), float("-inf"))
            logits = logits.view(sample_n_to_check * batch_size, vocab_size)
        else:
            assert tuple(logits.shape[:2]) == (seq_len, sample_n_to_check * batch_size)
            word2syl = self.word2syllables.view(1, 1, -1).expand(seq_len, batch_size, -1)
            word_mask = word2syl > remain_syllables.unsqueeze(-1).expand(-1, -1, vocab_size)
            logits = logits.view(seq_len, sample_n_to_check, batch_size, vocab_size)
            logits = logits.masked_fill(word_mask.unsqueeze(1), float("-inf"))
            logits = logits.view(seq_len, sample_n_to_check * batch_size, vocab_size)

        return logits

    # decoder_input: [seq_len, batch_size]
    def get_remain_syllables(self, decoder_input):
        assert decoder_input.dim() == 2
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
