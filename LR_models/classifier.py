"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
GNU General Public License v3.0
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        if in_dim==512:
            layers = [
                weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
                nn.ReLU(),
                nn.Dropout(dropout, inplace=True),
                weight_norm(nn.Linear(hid_dim, hid_dim*2), dim=None),
                nn.ReLU(),
                nn.Dropout(dropout, inplace=True),
                weight_norm(nn.Linear(hid_dim*2, out_dim), dim=None)
            ]
        else:
            layers = [
                weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
                nn.ReLU(),
                nn.Dropout(dropout, inplace=True),
                weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
            ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits