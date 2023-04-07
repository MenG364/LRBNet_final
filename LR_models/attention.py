import math

import torch
from torch import nn

from ReGAT_models.fc import FCNet
from devkit.ops import SwitchNorm2d


class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super(Attention, self).__init__()

        self.query_proj = FCNet([in_dim, hid_dim], act='LeakyReLU')
        self.key_proj = FCNet([in_dim, hid_dim], act='LeakyReLU')
        self.v_proj = FCNet([in_dim, hid_dim], act='LeakyReLU')
        self.linear_merge = FCNet([hid_dim, in_dim], act='LeakyReLU')
        self.linear_merge1 = FCNet([hid_dim, in_dim], act='LeakyReLU')
        # self.dropout = nn.Dropout(dropout)
        self.hid_dim = hid_dim
        self.head = 8

        # self.norm1 = SwitchNorm2d(1024)
        # self.norm2 = SwitchNorm2d(1024)

    def forward(self, source, target,mask=None):
        batch = source.size(0)
        dim = source.size(1)

        # target = target.permute(0, 2, 3, 1).view(batch, -1, dim)  # [batch,49,1024]
        source = source.unsqueeze(1)  # [batch,1,1024]
        # mask=self.make_mask(target)

        # source = self.norm(source.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        # target = self.norm(target.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

        query = self.query_proj(source)  # [batch, 1, vdim]

        key = self.key_proj(target)  # [batch, k, vdim]

        v = self.v_proj(target)  # [batch, k, vdim]

        atted, att_map = self.att(v, key, query,mask)
        atted = self.linear_merge(atted)  # [batch, k, vdim]
        q = self.linear_merge1(query.squeeze(1))  # [batch, k, vdim]

        # ret = target + atted

        return q,atted, att_map

    def att(self, value, key, query,mask=None):
        d_k = query.size(-1)

        ##1#### att_map maybe 0
        # scores = torch.matmul(
        #             query, key.transpose(-2, -1)
        #         )
        # if mask is not None:
        #     scores = scores.masked_fill(mask, -1e9)
        # att_map = torch.sigmoid(scores)  # [batch,1,k]


        ##2###
        # scores = torch.matmul(
        #             query, key.transpose(-2, -1)
        #         )
        # if mask is not None:
        #     scores = scores.masked_fill(mask, 1e-9)
        # att_map = torch.tanh(scores)  # [batch,1,k]

        ##3###
        # score = torch.cdist(query, key, 2)
        # if mask is not None:
        #     score = score.masked_fill(mask, 1e9)
        # att_map = 1.0 / (1 + score)


        ##4###
        att_map = torch.cosine_similarity(query, key, -1)
        if mask is not None:
            att_map = att_map.unsqueeze(1).masked_fill(mask, 1e-9)


        atted = att_map.permute(0, 2, 1)* value
        return atted, att_map.squeeze(1)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1)
