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
from LR_models.bilinear_attention import BiAttention
import torch.nn.functional as F
from ReGAT_models.fc import FCNet
from LR_models.bc import BCNet
from LR_models.counting import Counter
from block import fusions
import torch
from torch import nn
from ramen_models.rnn import RNN
from components import nonlinearity

"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is modified from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""


class BAN(nn.Module):
    def __init__(self, v_relation_dim, num_hid, gamma,
                 min_num_objects=10, use_counter=True):
        super(BAN, self).__init__()

        self.v_att = BiAttention(v_relation_dim, num_hid, num_hid, gamma)
        self.glimpse = gamma
        self.use_counter = use_counter
        b_net = []
        q_prj = []
        c_prj = []
        q_att = []
        v_prj = []

        for i in range(gamma):
            b_net.append(BCNet(v_relation_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            if self.use_counter:
                c_prj.append(FCNet([min_num_objects + 1, num_hid], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.q_att = nn.ModuleList(q_att)
        self.v_prj = nn.ModuleList(v_prj)
        if self.use_counter:
            self.c_prj = nn.ModuleList(c_prj)
            self.counter = Counter(min_num_objects)

    def forward(self, v_relation, q_emb, b):
        '''

        :param v_relation: [batch_size, num_rois, out_dim]
        :param q_emb: [batch_size, num_rois, out_dim]
        :param b: [batch_size, num_rois, 6]
        :return:
        '''

        if self.use_counter:
            boxes = b[:, :, :4].transpose(1, 2)  # [batch_size, 4, num_rois]

        b_emb = [0] * self.glimpse
        # b x g x v x q
        att, att_logits = self.v_att.forward_all(v_relation, q_emb)

        for g in range(self.glimpse):
            # b x l x h
            b_emb[g] = self.b_net[g].forward_with_weights(
                v_relation, q_emb, att[:, g, :, :])
            # atten used for counting module
            atten, _ = att_logits[:, g, :, :].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

            if self.use_counter:
                embed = self.counter(boxes, atten)
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)
        joint_emb = q_emb.sum(1)
        return joint_emb, att


"""
This code is modified by Linjie Li from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
GNU General Public License v3.0
"""


class BUTD(nn.Module):
    def __init__(self, v_relation_dim, q_dim, num_hid, dropout=0.2):
        super(BUTD, self).__init__()
        self.v_proj = FCNet([v_relation_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = FCNet([q_dim, 1])
        self.q_net = FCNet([q_dim, num_hid])
        self.v_net = FCNet([v_relation_dim, num_hid])

    def forward(self, v_relation, q_emb):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        b: bounding box features, not used for this fusion method
        """
        logits = self.logits(v_relation, q_emb)
        att = nn.functional.softmax(logits, 1)
        v_emb = (att * v_relation).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_emb = q_repr * v_repr
        return joint_emb, att

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


"""
This code is modified by Linjie Li from  Remi Cadene's repository.
https://github.com/Cadene/vqa.pytorch
"""


class MuTAN_Attention(nn.Module):
    def __init__(self, dim_v, dim_q, dim_out, method="Mutan", mlp_glimpses=0):
        super(MuTAN_Attention, self).__init__()
        self.mlp_glimpses = mlp_glimpses
        self.fusion = getattr(fusions, method)(
            [dim_q, dim_v], dim_out, mm_dim=1200,
            dropout_input=0.1)
        if self.mlp_glimpses > 0:
            self.linear0 = FCNet([dim_out, 512], '', 0)
            self.linear1 = FCNet([512, mlp_glimpses], '', 0)

    def forward(self, q, v):
        alpha = self.process_attention(q, v)

        if self.mlp_glimpses > 0:
            alpha = self.linear0(alpha)
            alpha = F.relu(alpha)
            alpha = self.linear1(alpha)

        alpha = F.softmax(alpha, dim=1)

        if alpha.size(2) > 1:  # nb_glimpses > 1
            alphas = torch.unbind(alpha, dim=2)
            v_outs = []
            for alpha in alphas:
                alpha = alpha.unsqueeze(2).expand_as(v)
                v_out = alpha * v
                v_out = v_out.sum(1)
                v_outs.append(v_out)
            v_out = torch.cat(v_outs, dim=1)
        else:
            alpha = alpha.expand_as(v)
            v_out = alpha * v
            v_out = v_out.sum(1)
        return v_out

    def process_attention(self, q, v):
        batch_size = q.size(0)
        n_regions = v.size(1)
        q = q[:, None, :].expand(q.size(0), n_regions, q.size(1))
        alpha = self.fusion([
            q.contiguous().view(batch_size * n_regions, -1),
            v.contiguous().view(batch_size * n_regions, -1)
        ])
        alpha = alpha.view(batch_size, n_regions, -1)
        return alpha


class MuTAN(nn.Module):
    def __init__(self, v_relation_dim, num_hid, num_ans_candidates, gamma):
        super(MuTAN, self).__init__()
        self.gamma = gamma
        self.attention = MuTAN_Attention(v_relation_dim, num_hid,
                                         dim_out=360, method="Mutan",
                                         mlp_glimpses=gamma)
        self.fusion = getattr(fusions, "Mutan")(
            [num_hid, v_relation_dim * 2], num_ans_candidates,
            mm_dim=1200, dropout_input=0.1)

    def forward(self, v_relation, q_emb):
        # b: bounding box features, not used for this fusion method
        att = self.attention(q_emb, v_relation)
        logits = self.fusion([q_emb, att])
        return logits, att


class MHAtt(nn.Module):
    def __init__(self, relation_dim, num_hid, dropout=0.1):
        super(MHAtt, self).__init__()

        self.query_proj = FCNet([relation_dim, num_hid], act='')
        self.key_proj = FCNet([relation_dim, num_hid], act='')
        self.v_proj = FCNet([relation_dim, num_hid], act='')
        self.linear_merge = FCNet([num_hid, num_hid], act='')
        self.dropout = nn.Dropout(dropout)
        self.head = 8

    def forward(self, v, key, query, mask):
        """
        v: [batch, k, vdim]
        q: [batch, k, qdim]

        """
        batch = v.size(0)
        k = v.size(1)
        query = self.query_proj(query).view(
            batch,
            k,
            self.head, -1).transpose(1, 2)  # [batch, h,k, vdim/k]

        key = self.key_proj(key).view(
            batch,
            k,
            self.head, -1).transpose(1, 2)  # [batch, h,k, vdim/k]

        v = self.v_proj(v).view(
            batch,
            k,
            self.head, -1).transpose(1, 2)  # [batch, h,k, vdim/k]

        atted, att = self.att(v, key, query, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            batch,
            k,
            -1
        )
        atted = self.linear_merge(atted)
        return atted, att

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value), att_map


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FCNet([in_size, mid_size], dropout=dropout_r)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class FFN(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=in_size,
            mid_size=mid_size,
            out_size=out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


class QC_SlefAttention(nn.Module):
    def __init__(self, relation_dim, num_hid, dropout=0.1):
        super(QC_SlefAttention, self).__init__()

        self.mhatt = MHAtt(num_hid, num_hid, dropout)
        self.ffn = FFN(num_hid, num_hid * 4, num_hid, dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(num_hid)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(num_hid)

    def forward(self, embed, mask):
        """
        v: [batch, k, vdim]

        """

        dim = embed.shape[-1]
        batch = embed.shape[0]

        x, att = self.mhatt(embed, embed, embed, mask)
        embed = embed + self.dropout1(x)
        embed = self.norm1(embed.view(-1, dim)).view(batch, -1, dim)

        embed = embed + self.dropout2(self.ffn(embed))
        embed = self.norm2(embed.view(-1, dim)).view(batch, -1, dim)
        return embed, att


class AttFlat(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.out_size = out_size

        self.mlp = MLP(
            in_size=in_size,
            mid_size=mid_size,
            out_size=out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            in_size * out_size,
            1024
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.out_size):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class QC_Fusion(nn.Module):
    def __init__(self, num_hid, fusion_num_layer=1, dropout=0.1):
        super(QC_Fusion, self).__init__()
        self.fusion_num_layer = fusion_num_layer
        self.self_attention_models = nn.ModuleList(
            [QC_SlefAttention(num_hid, num_hid) for _ in range(fusion_num_layer)])
        self.attflat = AttFlat(num_hid, num_hid, 1, dropout)
        self.BN = nn.BatchNorm1d(1024)

    def forward(self, embed):
        mask = self.make_mask(embed)
        for model in self.self_attention_models:
            embed, att = model(embed, mask)
        joint_embedding = self.attflat(embed, mask)
        joint_embedding = self.BN(joint_embedding)
        return joint_embedding, att

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

class RAMENFusion(nn.Module):
    """
    Concatenates visual and linguistic features and passes them through an MLP.
    """

    def __init__(self, out_s, num_ans_candidates,mmc_aggregator_dim=1024, mmc_aggregator_layers=1, disable_late_fusion=False,
                 disable_batch_norm_for_late_fusion=False):
        super(RAMENFusion, self).__init__()
        self.disable_late_fusion = disable_late_fusion
        self.disable_batch_norm_for_late_fusion = disable_batch_norm_for_late_fusion

        # Aggregation
        if not self.disable_late_fusion:
            if not self.disable_batch_norm_for_late_fusion:
                self.batch_norm_before_aggregation = nn.BatchNorm1d(out_s)
        self.aggregator = RNN(out_s, mmc_aggregator_dim, nlayers=mmc_aggregator_layers, bidirect=True)

        clf_in_size = 1024 * 2
        classifier_layers = []
        for ix, size in enumerate([2048]):
            in_s = clf_in_size if ix == 0 else [2048]
            out_s = size
            lin = nn.Linear(in_s, out_s)
            classifier_layers.append(lin)
            classifier_layers.append(getattr(nonlinearity, 'Swish')())
            classifier_layers.append(nn.Dropout(p=0.5))

        pre_classification_dropout=0
        if pre_classification_dropout is not None and pre_classification_dropout > 0:
            self.pre_classification_dropout = nn.Dropout(p=pre_classification_dropout)
        else:
            self.pre_classification_dropout = None
        self.pre_classification_layers = nn.Sequential(*classifier_layers)
        self.classifier = nn.Linear(out_s, num_ans_candidates)

    def __batch_norm(self, x, num_objs, flat_emb_dim):
        x = x.view(-1, flat_emb_dim)
        x = self.batch_norm_mmc(x)
        x = x.view(-1, num_objs, flat_emb_dim)
        return x

    def forward(self, x, q):
        """

        :param v: B x num_objs x emb_size
        :param q: B x emb_size
        :return:
        """
        x = self.q_expand_v_cat(q,x)
        curr_size = x.size()
        if not self.disable_batch_norm_for_late_fusion:
            x = x.view(-1, curr_size[2])
            x = self.batch_norm_before_aggregation(x)
            x = x.view(curr_size)
        # x = self.aggregator_dropout(x)
        x_aggregated = self.aggregator(x)
        if self.pre_classification_dropout is not None:
            x_aggregated = self.pre_classification_dropout(x_aggregated)
        final_emb = self.pre_classification_layers(x_aggregated)
        logits = self.classifier(final_emb)
        return logits

    def q_expand_v_cat(self,q, v, mask=True):
        q = q.view(q.size(0), 1, q.size(1))
        repeat_vals = (-1, v.shape[1], -1)
        q_expand = q.expand(*repeat_vals).clone()  # [batch_size, num_rois, q_dim]
        if mask:
            try:
                v_sum = v.sum(-1)  # 对每一个对象的特征求和
                mask_index = torch.nonzero(v_sum == 0, as_tuple=False)  # 特征和为0的对象的索引地址
            except AssertionError as e:
                print(e)
            if mask_index.dim() > 1:
                # 若有特征和为0的对象
                q_expand[mask_index[:, 0], mask_index[:, 1]] = 0  # 对应的q_expand=0
        v_cat_q = torch.cat((v, q_expand), dim=-1)
        return v_cat_q
