# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from LR_models.counting import Counter
from ReGAT_models.fc import FCNet
from ReGAT_models.graph_att import GAttNet as GAT
from film_models.models.film_gen import FiLMGen
from mini_mdls.net_utils import LayerNorm
from mini_mdls.tum import TUM
from mini_mdls.vum import VUM


def q_expand_v_cat(q, v, mask=True):
    q = q.view(q.size(0), 1, q.size(1))
    repeat_vals = (-1, v.shape[1], -1)
    q_expand = q.expand(*repeat_vals).clone()  # [batch_size, num_rois, q_dim]
    if mask:
        v_sum = v.sum(-1)  # 对每一个对象的特征求和
        mask_index = torch.nonzero(v_sum == 0)  # 特征和为0的对象的索引地址
        if mask_index.dim() > 1:
            # 若有特征和为0的对象
            q_expand[mask_index[:, 0], mask_index[:, 1]] = 0  # 对应的q_expand=0
    v_cat_q = torch.cat((v, q_expand), dim=-1)
    return v_cat_q


class ImplicitRelationEncoder(nn.Module):
    def __init__(self, v_dim, q_dim, out_dim, dir_num, pos_emb_dim,
                 nongt_dim, num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True, separation=True, use_count=True):
        super(ImplicitRelationEncoder, self).__init__()
        self.v_dim = v_dim  # visual features
        self.q_dim = q_dim  # question features
        self.out_dim = out_dim
        self.residual_connection = residual_connection
        self.num_steps = num_steps
        self.separation = separation
        print("In ImplicitRelationEncoder, num of graph propogate steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        in_dim = out_dim + q_dim

        self.T = FCNet([in_dim, out_dim])
        self.tum = nn.ModuleList([TUM(out_dim, out_dim) for _ in range(num_steps)])

        self.use_count = use_count
        if use_count:
            self.count_L = Counter(10, already_sigmoided=True)
        # self.count_R = Counter(10, already_sigmoided=True)

    def forward(self, q, cap, b, v_mask, cap_mask, position_embedding_L, position_embedding_R):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            Q: [batch_size, Q_dim] t2i_feat
            q: [batch_size, q_dim]
            cap: [batch_size,num_rois,cap_dim]
            num_rois：每个样本的候选框个数
            nongt_dim：每个图片的对象个数
        Returns:
            output: [batch_size, num_rois, out_dim]*2
        """

        # [batch_size, num_rois, num_rois, 1]
        count = None
        if self.use_count:
            count = self.count(q, cap, b, cap_mask, v_mask)

        cap_cat_q = q_expand_v_cat(q, cap, mask=True)  # [cap||q] batch_size, num_rois, out_dim+q_dim
        imp_cap_rel = self.T(cap_cat_q)

        for i in range(self.num_steps):
            imp_cap_rel = self.tum[i](imp_cap_rel, cap_mask.unsqueeze(1).unsqueeze(2))


        return imp_cap_rel, count

    def count(self, q, cap, b, cap_mask=None, v_mask=None):
        d_k = cap.size(2)

        cap_att_map = (torch.matmul(q.unsqueeze(1), cap.permute(0, 2, 1)).squeeze(1))
        cap_att_map = torch.sigmoid(cap_att_map)
        if cap_mask is not None:
            cap_att_map = cap_att_map.masked_fill(cap_mask, 1e-6)

        count_L = self.count_L(b[:, :, :4].permute(0, 2, 1), cap_att_map)  # [batch_size, mini_obj_num]

        return count_L
