import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ReGAT_models.fc import FCNet


class CoAttention(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super(CoAttention, self).__init__()

        self.query_proj = FCNet([in_dim, hid_dim], act='')
        self.key_proj = FCNet([in_dim, hid_dim], act='')
        self.v_proj = FCNet([in_dim, hid_dim], act='')
        self.linear_merge = FCNet([hid_dim, in_dim], act='LeakyReLU')
        # self.dropout = nn.Dropout(dropout)
        self.head = 8

    def forward(self, source, target):
        """
        source: [batch, k, vdim]
        target: [batch, k, qdim]

        """
        batch = source.size(0)
        k = source.size(1)

        query = self.query_proj(target).view(
            batch,
            k,
            self.head, -1).transpose(1, 2)  # [batch, h,k, vdim/k]

        key = self.key_proj(source).view(
            batch,
            k,
            self.head, -1).transpose(1, 2)  # [batch, h,k, vdim/k]

        v = self.v_proj(source).view(
            batch,
            k,
            self.head, -1).transpose(1, 2)  # [batch, h,k, vdim/k]

        atted = self.att(v, key, query)
        atted = atted.transpose(1, 2).contiguous().view(
            batch,
            k,
            -1
        )
        atted = self.linear_merge(atted)
        # ret = target + atted

        return atted

    def att(self, value, key, query):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        att_map = F.softmax(scores, dim=-1)
        # att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
