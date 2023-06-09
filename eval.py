import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import utils
from ReGAT_models.position_emb import prepare_graph_variables
from metrics import Metrics, accumulate_metrics, TBX
from train import instance_bce_with_logits, compute_score_with_ensemble
from vqa_utils import VqaUtils, PerTypeMetric
import pickle


def compute_score_with_logits(logits, labels, device):
    # argmax
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores


def save_true(logits, labels, device):
    logits = torch.max(logits, 1)[1].data
    labels = torch.max(labels, 1)[1].data
    return logits == labels


@torch.no_grad()
def evaluate_by_logits_key(model, dataloader, args, device):
    model.eval()
    relation_type = "implicit"

    entropy = None
    # if args.fusion == "ban":
    #    entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    # with open(os.path.join(args.data_root, args.feature_subdir, 'answer_ix_map.json')) as f:
    #     answer_ix_map = json.load(f)
    joint_loss, cap_loss, v_loss = 0, 0, 0
    all_preds = []
    score0 = 0
    scores = {}
    for i in range(0, 11, 2):
        for j in range(0, 11, 2):
            a = i / 10
            b = j / 10
            c = 1 - a - b
            if a + b <= 1:
                scores[str(a) + ',' + str(b) + ',' + "{:^2.1f}".format(c)] = 0
    num_examples = 0
    with tqdm(total=len(dataloader), ncols=120) as pbar:
        for i, (v, norm_bb, q, target, cap, question_types,
                question_ids, image_id, bb) in enumerate(dataloader):

            num_objects = v.size(1)

            v = Variable(v).to(device)
            norm_bb = Variable(norm_bb).to(device)
            q = Variable(q).to(device)
            target = Variable(target).to(device)
            cap = Variable(cap).to(device)

            sem_adj_matrix, spa_adj_matrix = None, None
            pos_emb_L, sem_adj_matrix_L, spa_adj_matrix_L = prepare_graph_variables(
                relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
                args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
                args.sem_label_num, device)

            pos_emb_R, sem_adj_matrix_R, spa_adj_matrix_R = pos_emb_L, sem_adj_matrix_L, spa_adj_matrix_L

            # pred, att = model(v, t2i, cap, norm_bb, q)
            pred = model(v, cap, norm_bb, q, pos_emb_L, pos_emb_R,
                         sem_adj_matrix_L, sem_adj_matrix_R,
                         spa_adj_matrix_L, spa_adj_matrix_R)
            joint_logits, cap_logits, v_logits = pred['logits']
            if args.dataset == 'vqa':
                joint_loss = instance_bce_with_logits(joint_logits, target)
                cap_loss = instance_bce_with_logits(cap_logits, target)
                v_loss = instance_bce_with_logits(v_logits, target)
            else:
                joint_loss = F.cross_entropy(joint_logits, target.to(torch.long))
                cap_loss = F.cross_entropy(cap_logits, target.to(torch.long))
                v_loss = F.cross_entropy(v_logits, target.to(torch.long))
            loss = (joint_loss + cap_loss + v_loss)
            # joint_soft = F.softmax(joint_logits, -1)
            # cap_soft = F.softmax(cap_logits, -1)
            # v_soft = F.softmax(v_logits, -1)
            joint_soft = F.sigmoid(joint_logits)
            cap_soft = F.sigmoid(cap_logits)
            v_soft = F.sigmoid(v_logits)
            bl = save_true(joint_soft, target, device)
            # for j in range(4):
            #     if bl[j]:
            #         pickle.dump((pred['att'][0][j], pred['att'][1][j], pred['att'][2][j]),
            #                     open("result/s-dmls/att/att_" + str(question_ids[j])+ ".pkl", "wb"))
            for i in range(0, 11, 2):
                for j in range(0, 11, 2):
                    a = i / 10
                    b = j / 10
                    if a + b <= 1:
                        c = 1 - a - b
                        pred1 = joint_soft * a + cap_soft * b + v_soft * c
                        scores[str(a) + ',' + str(b) + ',' + "{:^2.1f}".format(c)] += compute_score_with_logits(pred1,
                                                                                                                target,
                                                                                                                device).sum()
            pred0 = (joint_soft + cap_soft + v_soft) / 3
            score0 += compute_score_with_logits(pred0, target, device).sum()
            num_examples += v.size(0)

            pbar.set_postfix(score0='{:^7.3f}'.format(score0 / num_examples * 100))
            pbar.update()

    for i in range(0, 11, 2):
        for j in range(0, 11, 2):
            a = i / 10
            b = j / 10
            c = 1 - a - b
            if a + b <= 1:
                scores[str(a) + ',' + str(b) + ',' + "{:^2.1f}".format(c)] /= num_examples
    return score0 / num_examples, scores


def evaluate(model, eval_loader, args, device=torch.device("cuda")):
    if args.checkpoint != "":
        print("Loading weights from %s" % (args.checkpoint))
        if not os.path.exists(args.checkpoint):
            raise ValueError("No such checkpoint exists!")
        checkpoint = torch.load(args.checkpoint)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

    score0, scores = evaluate_by_logits_key(model, eval_loader, args, device)
    logger = utils.Logger(os.path.join('eval', 'log.txt'))
    print(scores)
    logger.write('%s: score0=%7.3f' % (args.checkpoint, score0 * 100))
    logger.write(scores)
