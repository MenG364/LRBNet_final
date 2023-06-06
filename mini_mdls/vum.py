# -*- coding: utf-8 -*-

import torch.nn as nn

from mini_mdls.mca import SA
from mini_mdls.vum_layer import VUMLayer


class VUM(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, layers=1):
        super(VUM, self).__init__()

        # layers = [VUMLayer(in_dim, out_dim, dropout) for _ in range(layers)]
        # self.main = nn.Sequential(*layers)
        self.main = SA(in_dim, dropout)
    def forward(self, x,mask):
        logits = self.main(x,mask)
        return logits
