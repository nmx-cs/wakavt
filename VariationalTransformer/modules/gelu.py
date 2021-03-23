# -*- coding:utf-8 -*-


from torch import nn


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, input):
        return nn.functional.gelu(input)
