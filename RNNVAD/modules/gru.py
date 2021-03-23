# -*- coding:utf-8 -*-


import torch
from torch import nn
from utils import xavier_uniform_fan_in_


class StandardGRUCell(nn.GRUCell):
    def __init__(self, *args, **kwargs):
        super(StandardGRUCell, self).__init__(*args, **kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        if self.bias:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)


# Code adapted from https://github.com/ElektrischesSchaf/LayerNorm_GRU/blob/main/GRU_layernorm_cell.py
class LayerNormedGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, layer_norm_trainable=True):
        super(LayerNormedGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        self.ln_i2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=layer_norm_trainable)
        self.ln_h2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=layer_norm_trainable)
        self.ln_cell_1 = nn.LayerNorm(hidden_size, elementwise_affine=layer_norm_trainable)
        self.ln_cell_2 = nn.LayerNorm(hidden_size, elementwise_affine=layer_norm_trainable)

        self.input_transform = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.states_transform = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_fan_in_(self.input_transform.weight)
        nn.init.orthogonal_(self.states_transform.weight)
        if self.use_bias:
            nn.init.zeros_(self.input_transform.bias)
            nn.init.zeros_(self.states_transform.bias)

    # x: [batch_size, input_size]
    # h: [batch_size, hidden_size]
    def forward(self, x, h):
        x_transformed = self.input_transform(x)
        h_transformed = self.states_transform(h)
        i2h = x_transformed[:, :(2 * self.hidden_size)]
        h_hat_first_half = x_transformed[:, (2 * self.hidden_size):]
        h2h = h_transformed[:, :(2 * self.hidden_size)]
        h_hat_last_half = h_transformed[:, (2 * self.hidden_size):]

        # layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)
        preact = i2h + h2h

        # activation
        gates = torch.sigmoid(preact)
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, self.hidden_size:]

        # layer norm
        h_hat_first_half = self.ln_cell_1(h_hat_first_half)
        h_hat_last_half = self.ln_cell_2(h_hat_last_half)

        h_hat = torch.tanh(h_hat_first_half + r_t * h_hat_last_half)
        h_t = (1 - z_t) * h + z_t * h_hat

        return h_t


class StackedGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 use_layer_norm=True, layer_norm_trainable=True, truncate_each_step=False):
        super(StackedGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.truncate_each_step = truncate_each_step

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_layer_norm:
                self.layers.append(LayerNormedGRUCell(input_size, hidden_size,
                                                      layer_norm_trainable=layer_norm_trainable))
            else:
                self.layers.append(StandardGRUCell(input_size, hidden_size))
            input_size = hidden_size

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    # input: [batch_size, input_size]
    # prev_hidden: [num_layers, batch_size, hidden_size]
    def forward(self, input, prev_hidden):
        if prev_hidden is None:
            prev_hidden = [None] * self.num_layers
        cur_hidden = []
        for i, layer in enumerate(self.layers):
            hidden_i = layer.forward(input, prev_hidden[i])
            cur_hidden.append(hidden_i)
            input = hidden_i
            if i < self.num_layers - 1:
                input = self.dropout(input)
        cur_hidden = torch.stack(cur_hidden, dim=0)
        if self.truncate_each_step:
            cur_hidden = cur_hidden.detach()
        return cur_hidden

    def __repr__(self):
        string = "{}(\n".format(self.__class__.__name__)
        for i, layer in enumerate(self.layers):
            layer_string = "({}): {} \n".format(i, repr(layer))
            string = string + layer_string
        string = string + ")\n"
        return string
