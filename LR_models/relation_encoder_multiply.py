# -*- coding: utf-8 -*-
"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from LR_models.attention import Attention
from LR_models.counting import Counter
from ReGAT_models.graph_att import GAttNet as GAT
from ReGAT_models.fc import FCNet
from film_models.models.film_gen import FiLMGen


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

        if self.v_dim != self.out_dim:
            if self.q_dim == 512:
                self.v_transform = FCNet([v_dim, 1024, out_dim], act="LeakyReLU")
            else:
                self.v_transform = FCNet([v_dim, out_dim], act="LeakyReLU")
        else:
            self.v_transform = None

        # self.U = FCNet([out_dim, out_dim])
        # self.V = FCNet([q_dim, out_dim])

        # self.Q_transform = FCNet([2048, out_dim])
        in_dim = out_dim + q_dim
        self.combined_feature_dim = 512
        self.edge_layer_1 = nn.Linear(in_dim, out_dim)
        self.edge_layer_2 = nn.Linear(out_dim, self.combined_feature_dim) if self.q_dim == 1024 else None
        self.edge_layer_3 = nn.Linear(out_dim, out_dim)
        self.edge_layer_4 = nn.Linear(out_dim, self.combined_feature_dim) if self.q_dim == 1024 else None

        self.BN_L = nn.BatchNorm1d(in_dim)
        # self.BN_R = nn.BatchNorm1d(in_dim)
        self.implicit_relation_L = GAT(dir_num, 1, in_dim, out_dim,
                                       nongt_dim=nongt_dim,
                                       label_bias=label_bias,
                                       num_heads=num_heads,
                                       pos_emb_dim=pos_emb_dim)
        self.implicit_relation_R = GAT(dir_num, 1, out_dim, out_dim,
                                       nongt_dim=nongt_dim,
                                       label_bias=label_bias,
                                       num_heads=num_heads,
                                       pos_emb_dim=pos_emb_dim)

        self.film_L = FCNet([out_dim, out_dim], act="LeakyReLU")
        self.film_R = FCNet([out_dim, out_dim], act="LeakyReLU")
        self.use_count = use_count
        if use_count:
            self.count_L = Counter(10, already_sigmoided=True)
        # self.count_R = Counter(10, already_sigmoided=True)

    def forward(self, v, q, cap, b, v_mask, cap_mask, position_embedding_L, position_embedding_R):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            Q: [batch_size, Q_dim] t2i_feat
            q: [batch_size, q_dim]
            cap: [batch_size,num_rois,cap_dim]
            position_embedding: [batch_size, num_rois, nongt_dim, emb_dim]
            num_rois：每个样本的候选框个数
            nongt_dim：每个图片的对象个数
        Returns:
            output: [batch_size, num_rois, out_dim]*2
        """

        # [batch_size, num_rois, num_rois, 1]

        num_rois = v.size(1)
        batch_size = v.size(0)
        k = num_rois
        v = torch.cat((v, b), -1)

        imp_v = self.v_transform(v) if self.v_transform else v  # [batch_size, num_rois, out_dim]

        count = None
        if self.use_count:
            count = self.count(q, cap, b, cap_mask, v_mask)

        for i in range(self.num_steps):
            cap_cat_q = q_expand_v_cat(q, cap, mask=True)  # [cap||q] batch_size, num_rois, out_dim+q_dim
            emb_size = cap_cat_q.size(-1)
            cap_cat_q = cap_cat_q.view(-1, emb_size)
            cap_cat_q_bn = self.BN_L(cap_cat_q)
            cap_cat_q_bn = cap_cat_q_bn.view(-1, num_rois, emb_size)

            imp_adj_mat_L = self.create_adj_mat1(cap_cat_q_bn, k).view(v.size(0), num_rois, num_rois, 1)
            imp_cap_rel = self.implicit_relation_L.forward(cap_cat_q_bn,
                                                           imp_adj_mat_L,
                                                           position_embedding_L)
            v_cat_q_bn = imp_v.clone()
            imp_adj_mat_R = self.create_adj_mat2(v_cat_q_bn, k).view(v.size(0), num_rois, num_rois, 1)
            imp_v_rel = self.implicit_relation_R.forward(v_cat_q_bn,
                                                         imp_adj_mat_R,
                                                         position_embedding_R)

            cap_v=imp_v_rel*imp_cap_rel
            imp_v_rel = self.film_L(cap_v)  # [batch, num_rois, out_dim * 2]
            imp_cap_rel = self.film_R(cap_v)  # [batch, num_rois, out_dim * 2]

            imp_v_rel = F.relu(imp_v_rel)
            imp_cap_rel = F.relu(imp_cap_rel)

            if self.residual_connection:
                imp_v += imp_v_rel
                cap += imp_cap_rel
            else:
                imp_v = imp_v_rel
                cap = imp_cap_rel

        return imp_v, cap, count

    def create_adj_mat1(self, graph_nodes, k, mask=None):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        in_dim = graph_nodes.size(2)
        graph_nodes = graph_nodes.contiguous().view(-1, in_dim)

        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.leaky_relu(h)

        # layer 2
        if self.edge_layer_2 is not None:
            h = self.edge_layer_2(h)
            h = F.leaky_relu(h)

        # outer product
        h = h.view(-1, k, self.combined_feature_dim)
        adjacency_matrix = torch.cosine_similarity(h.unsqueeze(1), h.unsqueeze(2), -1)
        return adjacency_matrix

    def create_adj_mat2(self, graph_nodes, k, mask=None):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        in_dim = graph_nodes.size(2)
        graph_nodes = graph_nodes.contiguous().view(-1, in_dim)

        # layer 1
        h = self.edge_layer_3(graph_nodes)
        h = F.leaky_relu(h)

        # layer 2
        if self.edge_layer_4 is not None:
            h = self.edge_layer_4(h)
            h = F.leaky_relu(h)

        # outer product
        h = h.view(-1, k, self.combined_feature_dim)
        adjacency_matrix = torch.cosine_similarity(h.unsqueeze(1), h.unsqueeze(2), -1)
        return adjacency_matrix

    def count(self, q, cap, b, cap_mask=None, v_mask=None):
        d_k = cap.size(2)

        cap_att_map = (torch.matmul(q.unsqueeze(1), cap.permute(0, 2, 1)).squeeze(1))
        cap_att_map = torch.sigmoid(cap_att_map)
        if cap_mask is not None:
            cap_att_map = cap_att_map.masked_fill(cap_mask, 1e-6)

        count_L = self.count_L(b[:, :, :4].permute(0, 2, 1), cap_att_map)  # [batch_size, mini_obj_num]

        return count_L


class ExplicitRelationEncoder(nn.Module):
    def __init__(self, v_dim, q_dim, out_dim, dir_num, label_num,
                 nongt_dim=20, num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True, separation=True):
        super(ExplicitRelationEncoder, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.out_dim = out_dim
        self.num_steps = num_steps
        self.residual_connection = residual_connection
        self.separation = separation
        print("In ExplicitRelationEncoder, num of graph propogation steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim])
            self.cap_transform = FCNet([v_dim, out_dim])
        else:
            self.v_transform = None
            self.cap_transform = None
        in_dim = out_dim + q_dim

        self.BN_L = nn.BatchNorm1d(in_dim)
        self.BN_R = nn.BatchNorm1d(in_dim)
        self.explicit_relation_L = GAT(dir_num, label_num, in_dim, out_dim,
                                       nongt_dim=nongt_dim,
                                       num_heads=num_heads,
                                       label_bias=label_bias,
                                       pos_emb_dim=-1)
        self.explicit_relation_R = GAT(dir_num, label_num, in_dim, out_dim,
                                       nongt_dim=nongt_dim,
                                       num_heads=num_heads,
                                       label_bias=label_bias,
                                       pos_emb_dim=-1)

        if self.separation:
            self.film_L = FiLMGen(output_batchnorm=False,
                                  bidirectional=False,
                                  encoder_type='FCNet',
                                  act_type='linear',
                                  gamma_option='linear',
                                  gamma_baseline=1,
                                  module_dim=in_dim)

            self.film_R = FiLMGen(output_batchnorm=False,
                                  bidirectional=False,
                                  encoder_type='FCNet',
                                  act_type='linear',
                                  gamma_option='linear',
                                  gamma_baseline=1,
                                  module_dim=in_dim)
        else:
            self.film = FiLMGen(output_batchnorm=False,
                                bidirectional=False,
                                encoder_type='FCNet',
                                act_type='linear',
                                gamma_option='linear',
                                gamma_baseline=1,
                                module_dim=in_dim)

    def forward(self, v, Q, q, cap, exp_adj_matrix_L, exp_adj_matrix_R):
        """
        Args:
            v: [batch_size, num_rois, v_dim]
            Q: [batch_size, Q_dim] t2i_feat
            q: [batch_size, q_dim]
            cap: [batch_size,num_rois,cap_dim]
            exp_adj_matrix: [batch_size, num_rois, num_rois, num_labels]

        Returns:
            output: [batch_size, num_rois, out_dim]
        """
        num_rois = v.size(1)
        exp_v = self.v_transform(v) if self.v_transform else v
        exp_cap = self.cap_transform(cap) if self.cap_transform else cap

        for i in range(self.num_steps):
            cap_cat_q = q_expand_v_cat(q, exp_cap, mask=True)  # [cap||q] batch_size, num_rois, out_dim+q_dim
            emb_size = cap_cat_q.size(-1)
            cap_cat_q = cap_cat_q.view(-1, emb_size)
            cap_cat_q_bn = self.BN_L(cap_cat_q)
            cap_cat_q_bn = cap_cat_q_bn.view(-1, num_rois, emb_size)
            exp_cap_rel = self.explicit_relation_L.forward(cap_cat_q_bn, exp_adj_matrix_L)

            v_cat_q = q_expand_v_cat(Q, exp_v, mask=True)  # [V||Q] batch_size, num_rois, out_dim+q_dim
            v_cat_q = v_cat_q.view(-1, emb_size)
            v_cat_q_bn = self.BN_R(v_cat_q)
            v_cat_q_bn = v_cat_q_bn.view(-1, num_rois, emb_size)
            exp_v_rel = self.explicit_relation_R.forward(v_cat_q_bn, exp_adj_matrix_R)

            if self.Separation:
                L_film = self.film_L(exp_cap_rel)  # [batch, num_rois, out_dim * 2]
                R_film = self.film_R(exp_v_rel)  # [batch, num_rois, out_dim * 2]
            else:
                L_film = self.film(exp_cap_rel)  # [batch, num_rois, out_dim * 2]
                R_film = self.film(exp_v_rel)  # [batch, num_rois, out_dim * 2]

            gs = slice(0, self.out_dim)
            bs = slice(self.out_dim, (2 * self.out_dim))
            exp_v_rel = torch.mul(exp_v_rel, L_film[:, :, gs]) + L_film[:, :, bs]
            exp_cap_rel = torch.mul(exp_cap_rel, R_film[:, :, gs]) + R_film[:, :, bs]

            exp_v_rel = F.relu(exp_v_rel)
            exp_cap_rel = F.relu(exp_cap_rel)
            if self.residual_connection:
                exp_v += exp_v_rel
                exp_cap += exp_cap_rel
            else:
                exp_v = exp_v_rel
                exp_cap = exp_cap_rel
        return exp_v, exp_cap
