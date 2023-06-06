# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from mini_mdls.mca import SA


class VUMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(VUMLayer, self).__init__()

        layers = [
            weight_norm(nn.Linear(in_dim, out_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits