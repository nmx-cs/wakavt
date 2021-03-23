# -*- coding:utf-8 -*-


import math
import numpy as np
import torch
from torch import nn


class BatchLinear(nn.Module):
    def __init__(self, batch_size, in_features, out_features, bias=True):
        super(BatchLinear, self).__init__()
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(batch_size, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(batch_size, out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        scale = math.sqrt(3.0 / self.in_features)
        nn.init.uniform_(self.weight, a=-scale, b=scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # input: [*, batch_size, in_features]
    def forward(self, input):
        assert tuple(input.shape[-2:]) == (self.batch_size, self.in_features)
        extra_dims = [1] * (input.dim() - 2)
        weight = self.weight.view(*extra_dims, self.batch_size, self.out_features, self.in_features)
        output = (weight * input.unsqueeze(-2)).sum(dim=-1, keepdim=False)
        if self.bias is not None:
            output = output + self.bias.view(*extra_dims, self.batch_size, self.out_features)
        return output

    def extra_repr(self):
        return 'batch_size={}, in_features={}, out_features={}, bias={}'.format(
            self.batch_size, self.in_features, self.out_features, self.bias is not None)


class SharedLinearNoBias(nn.Module):
    def __init__(self, weight, in_feature_dim=1, parted_block=None):
        super(SharedLinearNoBias, self).__init__()
        assert isinstance(weight, nn.Parameter) and weight.dim() == 2
        assert in_feature_dim in (0, 1)
        self.weight = weight
        self.in_feature_dim = in_feature_dim
        if parted_block is not None:
            dim0_indices, dim1_indices = parted_block
            self.register_buffer("dim0_indexes", self._convert_to_index(dim0_indices, 0))
            self.register_buffer("dim1_indexes", self._convert_to_index(dim1_indices, 1))
            self.block = nn.Parameter(torch.Tensor(self.dim0_indexes.size(0), self.dim1_indexes.size(0)))
            scale = math.sqrt(3 / self.weight.size(in_feature_dim))
            nn.init.uniform_(self.block, a=-scale, b=scale)

    def _convert_to_index(self, indices, dim):
        if isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices)
        if isinstance(indices, torch.BoolTensor):
            assert indices.dim() == 1 and indices.size(0) == self.weight.size(dim)
        if isinstance(indices, torch.BoolTensor) or isinstance(indices, slice):
            indexes = torch.arange(self.weight.size(dim), device=self.weight.device, dtype=torch.long)[indices]
        else:
            indexes = torch.as_tensor(indices, device=self.weight.device, dtype=torch.long)
            if indexes.dim() == 0:
                indexes = indexes.view(1)
        assert indexes.dim() == 1 and ((indexes >= 0) & (indexes < self.weight.size(dim))).all()
        return indexes

    def forward(self, input):
        weight = self.weight
        if hasattr(self, "block"):
            rows = weight[self.dim0_indexes]
            rows[:, self.dim1_indexes] = self.block
            weight = weight.index_copy(0, self.dim0_indexes, rows)
        return nn.functional.linear(input, weight if self.in_feature_dim == 1 else weight.t())
