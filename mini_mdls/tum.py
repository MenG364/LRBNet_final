# -*- coding: utf-8 -*-

import torch.nn as nn

from ReGAT_models.fc import FCNet
from mini_mdls.mca import SA
from mini_mdls.tum_layer import TUMLayer


class TUM(nn.Module):
    def __init__(self, in_dim, out_dim, layers=1, dropout=0.1):
        super(TUM, self).__init__()

        # layers = [TUMLayer(in_dim, out_dim, dropout) for _ in range(layers)]
        # self.main = nn.Sequential(*layers)
        self.main = SA(in_dim, dropout)

    def forward(self, x, mask):
        logits = self.main(x,mask)
        return logits
