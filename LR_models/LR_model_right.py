"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.nn import Parameter

from LR_models.classifier import SimpleClassifier
from LR_models.fusion import BAN, BUTD, MuTAN
from LR_models.language_model import WordEmbedding, TextEmbedding, \
    TextSelfAttention, QuestionEmbedding
from LR_models.relation_encode_right import ExplicitRelationEncoder, ImplicitRelationEncoder
from ReGAT_models.fc import FCNet


class LR_model(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation,
                  right_classifier, joint_R, glimpse,
                 fusion, relation_type, use_count=True):
        super(LR_model, self).__init__()
        self.name = "L_model%s" % (relation_type)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.v_relation = v_relation
        self.right_classifier = right_classifier
        self.use_count = use_count
        self.joint_R = joint_R if joint_R is not None else None

    def forward(self, v, c, b, q, implicit_pos_emb_L, implicit_pos_emb_R, sem_adj_matrix_L=None,
                sem_adj_matrix_R=None,
                spa_adj_matrix_L=None, spa_adj_matrix_R=None):
        """Forward
        v: [batch, num_objs, cap_dim] visual_feat
        Q: [batch_size, t2i_dim] t2i_feat
        c: [batch, num_objs, cap_dim] captions
        b: [batch, num_objs, b_dim] boundding_boxes
        q: [batch_size, seq_length] questions
        pos: [batch_size, num_objs, nongt_dim, emb_dim] position
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]

        return: logits, not probs
        """
        # L
        ########question_embedding##########
        question = rnn.pack_sequence(q)
        question, lens = rnn.pad_packed_sequence(question)
        question = question.permute(1, 0)
        w_emb = self.w_emb(question)
        # q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_emb_seq = self.q_emb.forward_all(w_emb, lens)  # [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq)  # [batch, num_hid]
        v_mask = self.make_mask(v)
        cap_mask = self.make_mask(c)

        # [batch_size, num_rois, out_dim]*2
        v_emb = self.v_relation.forward(v, b, implicit_pos_emb_R)
        if self.right_classifier:
            # v_emb, _ = self.joint_R(v_emb, Q)
            v_emb, att = self.joint_R(v_emb, q_emb_seq, b)
            # cap_emb, _ = self.joint_L(cap_emb, q_emb_self_att)
            v_logits = self.right_classifier(v_emb)


        out = {'logits': (None, None, v_logits), 'q_emb': q_emb_self_att, 'att': att}

        return out

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0)


def build_LR_model_right(dataset, args):
    print("Building ReGAT ReGAT_models with %s relation and %s fusion method" %
          (args.relation_type, args.fusion))
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = TextEmbedding(300 if 'c' not in args.op else 600,
                          args.num_hid, 1, False, .0)
    q_att = TextSelfAttention(args.num_hid, .2)
    use_count=args.use_count
    if args.relation_type == "semantic":
        v_relation = ExplicitRelationEncoder(
            dataset.v_dim, args.num_hid, args.relation_dim,
            args.dir_num, args.sem_label_num,
            num_heads=args.num_heads,
            num_steps=args.num_steps, nongt_dim=args.nongt_dim,
            residual_connection=args.residual_connection,
            label_bias=args.label_bias)
    elif args.relation_type == "spatial":
        v_relation = ExplicitRelationEncoder(
            dataset.v_dim, args.num_hid, args.relation_dim,
            args.dir_num, args.spa_label_num,
            num_heads=args.num_heads,
            num_steps=args.num_steps, nongt_dim=args.nongt_dim,
            residual_connection=args.residual_connection,
            label_bias=args.label_bias)
    else:
        v_relation = ImplicitRelationEncoder(
            dataset.v_dim, args.num_hid, args.relation_dim,
            args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
            num_heads=args.num_heads, num_steps=args.num_steps,
            residual_connection=args.residual_connection,
            label_bias=args.label_bias,use_count=use_count)

    right_classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                        dataset.num_ans_candidates, 0.5)
    gamma = 0
    # joint_R = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    joint_R = BAN(args.relation_dim, args.num_hid, args.ban_gamma,use_counter=False)

    return LR_model(dataset, w_emb, q_emb, q_att, v_relation,
                    right_classifier, joint_R, gamma, args.fusion,
                    args.relation_type,use_count)
