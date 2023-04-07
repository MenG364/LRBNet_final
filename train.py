"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
import math
import os
import pickle
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
from vqa_utils import VqaUtils, PerTypeMetric
from torch.cuda.amp import autocast, GradScaler


def compute_score_with_ensemble(pred1, pred2, pred3, labels):
    logits = [torch.max(pred1, 1)[1].data, torch.max(pred2, 1)[1].data, torch.max(pred3, 1)[1].data]
    output = []
    batch = pred1.size(0)
    for i in range(batch):
        out = [0 for _ in range(3129)]
        for j in range(3):
            out[logits[j][i].item()] += 1
        m = max(out)
        if m == 1:
            output.append(logits[0][i].item())
        else:
            output.append(out.index(m))
    output = torch.as_tensor(output).to(labels.device)
    if len(labels.size()) == 2:
        one_hots = torch.zeros(*labels.size()).to(labels.device)
        output = output.view(-1, 1)
        one_hots.scatter_(1, output, 1)
        scores = (one_hots * labels).sum()
    else:
        scores = (output == labels.data).sum()
    return scores


def save_metrics_n_model(metrics, model, optimizer, args, is_best):
    """
    Saves all the metrics, parameters of models and parameters of optimizer.
    If current score is the highest ever, it also saves this model as the best model
    """
    metrics_n_model = metrics.copy()
    metrics_n_model["model_state_dict"] = model.state_dict()
    metrics_n_model["optimizer_state_dict"] = optimizer.state_dict()
    metrics_n_model["args"] = args

    with open(os.path.join(args.expt_save_dir, 'latest-model.pth'), 'wb') as lmf:
        torch.save(metrics_n_model, lmf)

    if is_best:
        with open(os.path.join(args.expt_save_dir, 'best-model.pth'), 'wb') as bmf:
            torch.save(metrics_n_model, bmf)

    return metrics_n_model


def label_smoothing(one_hot, n_class, eps=0.1):
    return one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    # labels = label_smoothing(labels, 2)
    loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == "mean":
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels, device):
    # argmax
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores


class lr_schedule_cosine():
    def __init__(self, T_0, T_mult=1, eta_max=1., eta_min=0., last_epoch=-1, restart=True,
                 warm_up_schedule=None):
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = last_epoch
        self.lr_warm_up_schedule = [] if warm_up_schedule == None else warm_up_schedule
        self.warm_up_step = len(self.lr_warm_up_schedule)
        self.T_0 = T_0 - len(self.lr_warm_up_schedule) - 1
        self.T_i = T_0 - len(self.lr_warm_up_schedule) - 1
        self.restart = restart

    def compute_restart(self):
        self.T_cur += 1
        if self.T_cur == self.T_i:
            lr = self.eta_min
            self.T_i *= self.T_mult
            self.T_cur = -1
        elif self.T_cur == 0:
            lr = self.eta_max
        else:
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return lr

    def compute(self):
        self.T_cur += 1
        if self.T_cur == 0:
            lr = self.eta_max
        else:
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return lr

    def get_lr(self, epoch):
        if epoch < len(self.lr_warm_up_schedule):
            lr_select = self.lr_warm_up_schedule[epoch]
        else:
            if self.restart:
                lr_select = self.compute_restart()
            else:
                lr_select = self.compute()
        return lr_select


def train(model, train_loader, eval_loader, args, device=torch.device("cuda")):
    metrics_stats_list = []
    val_per_type_metric_list = []
    best_val_score = 0
    best_val_epoch = 0
    N = len(train_loader.dataset)
    lr_default = args.base_lr
    num_epochs = args.epochs

    lr_decay_epochs = range(args.lr_decay_start, num_epochs,
                            args.lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,
                            1.5 * lr_default, 2.0 * lr_default]

    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=lr_default, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 5, eta_min=0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=5, T_mult=1,eta_min=0.00001, last_epoch=-1)
    warm_up_schedule = []
    lr_d = lr_schedule_cosine(num_epochs, T_mult=1, eta_max=1, eta_min=1e-1, restart=True,
                              warm_up_schedule=warm_up_schedule)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_d.get_lr)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f,'
                 % (lr_default, args.lr_decay_step,
                    args.lr_decay_rate) + 'grad_clip=%.2f' % args.grad_clip)
    logger.write('LR decay epochs: ' + ','.join(
        [str(i) for i in lr_decay_epochs]))
    last_eval_score, eval_score = 0, 0
    relation_type = 'implicit'

    out = args.output.split('/')[0]
    train_TBX = TBX(args.seed)
    start_epoch = 0
    if args.checkpoint != "":
        print("Loading weights from %s" % (args.checkpoint))
        if not os.path.exists(args.checkpoint):
            raise ValueError("No such checkpoint exists!")
        checkpoint = torch.load(args.checkpoint)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)

        # opt_state_dict = checkpoint.get('optimizer_state_dict', checkpoint)
        # optim.load_state_dict(opt_state_dict)
        # optim.param_groups = checkpoint.get('optimizer_state_dict', checkpoint)['param_groups']
        # start_epoch = checkpoint['epoch'] + 1

    # if args.checkpoint != "":
    #     start_epoch = args.checkpoint.split('_')[1].split('.')[0]
    scaler = GradScaler()
    for epoch in range(start_epoch, num_epochs + 1):
        count, average_loss, att_entropy = 0, 0, 0
        logger.write('===========================')
        t = time.time()
        if epoch < len(gradual_warmup_steps):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %f' %
                         optim.param_groups[-1]['lr'])
        elif (epoch in lr_decay_epochs or
              eval_score < last_eval_score and args.lr_decay_based_on_val):
            for i in range(len(optim.param_groups)):
                optim.param_groups[i]['lr'] *= args.lr_decay_rate
            logger.write('decreased lr: %f' % optim.param_groups[-1]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[-1]['lr'])

        last_eval_score = eval_score

        is_best = False
        train_metrics, val_metrics = Metrics(), Metrics()

        batch_multiplier = args.grad_accu_steps
        mini_batch_count = batch_multiplier
        # evaluate_by_logits_key(model, eval_loader, epoch, args, val_metrics, train_TBX, device,
        #                        batch_multiplier)
        # break
        with tqdm(total=len(train_loader), ncols=160) as pbar:
            for i, (v, norm_bb, q, target, cap, _, _, _, bb) in enumerate(train_loader):
                batch_size = v.size(0)
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

                # with autocast():
                pred = model(v, cap, norm_bb, q, pos_emb_L, pos_emb_R,
                             sem_adj_matrix_L, sem_adj_matrix_R,
                             spa_adj_matrix_L, spa_adj_matrix_R)

                # 3loss
                joint_logits, cap_logits, v_logits = pred['logits']
                if args.model == 'model' or args.model == 'left_right_model' or args.model == 'MMDLS':
                    if (args.dataset == 'vqa' or args.dataset == 'vqa1.0'):
                        joint_loss = instance_bce_with_logits(joint_logits, target)
                        cap_loss = instance_bce_with_logits(cap_logits, target)
                        v_loss = instance_bce_with_logits(v_logits, target)
                        pred1 = (torch.sigmoid(joint_logits) + torch.sigmoid(cap_logits) + torch.sigmoid(v_logits)) / 3
                    else:
                        joint_loss = F.cross_entropy(joint_logits, target.to(torch.long))
                        cap_loss = F.cross_entropy(cap_logits, target.to(torch.long))
                        v_loss = F.cross_entropy(v_logits, target.to(torch.long))
                        pred1 = (F.softmax(joint_logits, -1) + F.softmax(cap_logits, -1) + F.softmax(v_logits, -1)) / 3
                    loss = (joint_loss + cap_loss + v_loss) / 3
                    pred = (joint_logits + cap_logits + v_logits) / 3
                elif args.model == 'left_model' or args.model == 'MMDLS_left':
                    cap_loss = instance_bce_with_logits(cap_logits, target)
                    joint_loss = None
                    v_loss = None
                    pred = cap_logits
                    pred1 = torch.sigmoid(cap_logits)
                    loss = cap_loss
                elif args.model == 'right_model' or args.model == 'MMDLS_right':
                    v_loss = instance_bce_with_logits(v_logits, target)
                    joint_loss = None
                    cap_loss = None
                    pred = v_logits
                    pred1 = torch.sigmoid(v_logits)
                    loss = v_loss

                loss /= batch_multiplier
                loss.backward()
                # scaler.scale(loss).backward()

                mini_batch_count -= 1
                train_metrics.update_per_batch(model, target, (loss, joint_loss, cap_loss, v_loss),
                                               (pred, joint_logits, cap_logits, v_logits, pred1), v.shape[0],
                                               batch_multiplier,
                                               device)

                train_TBX.log_per_batch('train', (train_metrics.loss / train_metrics.num_examples,
                                                  train_metrics.loss1[0] / train_metrics.num_examples,
                                                  train_metrics.loss1[1] / train_metrics.num_examples,
                                                  train_metrics.loss1[2] / train_metrics.num_examples),
                                        (100 * train_metrics.raw_score / train_metrics.num_examples,
                                         100 * train_metrics.raw_score1[0] / train_metrics.num_examples,
                                         100 * train_metrics.raw_score1[1] / train_metrics.num_examples,
                                         100 * train_metrics.raw_score1[2] / train_metrics.num_examples,
                                         100 * train_metrics.raw_score1[3] / train_metrics.num_examples,),
                                        100 * loss / batch_size, 100 * train_metrics.batch_score / batch_size)
                if mini_batch_count == 0:
                    optim.step()
                    # scaler.step(optim)
                    # scaler.update()
                    optim.zero_grad()
                    mini_batch_count = batch_multiplier
                pbar.set_description('epoch{}'.format(epoch))
                pbar.set_postfix(loss0='{:^7.3f}'.format(train_metrics.loss / train_metrics.num_examples),
                                 loss_j='{:^7.3f}'.format(train_metrics.loss1[0] / train_metrics.num_examples),
                                 loss_c='{:^7.3f}'.format(train_metrics.loss1[1] / train_metrics.num_examples),
                                 loss_v='{:^7.3f}'.format(train_metrics.loss1[2] / train_metrics.num_examples),
                                 score='{:^7.3f}'.format(
                                     100 * train_metrics.raw_score / train_metrics.num_examples),
                                 lr='{:^7.6f}'.format(optim.param_groups[-1]['lr']))

                pbar.update(1)

    train_metrics.update_per_epoch()
    train_TBX.log_per_epoch('train', (
        train_metrics.loss, train_metrics.loss1[0], train_metrics.loss1[1], train_metrics.loss1[2]),
                            (train_metrics.score, train_metrics.score1[0], train_metrics.score1[1],
                             train_metrics.score1[2],
                             train_metrics.score1[3]),
                            train_metrics.norm, lr=optim.param_groups[-1]['lr'])
    # scheduler.step()

    if eval_loader is not None:
        # eval_score, bound, entropy = evaluate(
        #     model, eval_loader, device, args)
        val_results = evaluate_by_logits_key(model, eval_loader, epoch, args, val_metrics, train_TBX, device,
                                             batch_multiplier)

        if val_metrics.score > best_val_score:
            best_val_score = val_metrics.score
            best_val_epoch = epoch
            is_best = True

        save_val_metrics = not args.test or not args.test_does_not_have_answers
        if save_val_metrics:
            print("Best val score {} at epoch {}".format(best_val_score, best_val_epoch))
            print(f"### Val from Logits {val_metrics.score}")

        save_val_metrics = not args.test or not args.test_does_not_have_answers
        val_per_type_metric_list.append(val_results['per_type_metric'].get_json())

        metrics = accumulate_metrics(epoch, train_metrics, val_metrics, val_results['per_type_metric'],
                                     best_val_score, best_val_epoch,
                                     save_val_metrics)
        metrics_stats_list.append(metrics)
        metrics_n_model = save_metrics_n_model(metrics, model, optim, args, is_best)
        VqaUtils.save_stats(metrics_stats_list, val_per_type_metric_list, val_results['all_preds'],
                            args.expt_save_dir,
                            split=args.test_split, epoch=epoch)

    logger.write('epoch %d, time: %.4f' % (epoch, time.time() - t))
    logger.write('\ttrain_loss: %.4f, norm: %.4f, score: %.4f'
                 % (train_metrics.loss, train_metrics.norm, train_metrics.raw_score))

    if eval_loader is not None:
        eval_score = val_metrics.score
        logger.write('\teval score: %.4f (%.4f)'
                     % (eval_score, 100 * val_metrics.upper_bound))
        entropy = val_results['entropy']
        if entropy is not None:
            info = ''
            for i in range(entropy.size(0)):
                info = info + ' %.4f' % entropy[i]
            logger.write('\tentropy: ' + info)

    if (eval_loader is not None) \
            or (eval_loader is None and epoch >= args.saving_epoch):
        logger.write("saving current model weights to folder")
        model_path = os.path.join(args.output, 'model_%d.pth' % epoch)
        opt = optim if args.save_optim else None
        utils.save_model(model_path, model, epoch, opt)

    if args.test:
        VqaUtils.save_preds(val_results['all_preds'], args.expt_save_dir, args.test_split, epoch)
        print("Test completed!")


@torch.no_grad()
def evaluate_by_logits_key(model, dataloader, epoch, args, val_metrics, val_TBX, device, batch_multiplier,
                           logits_key='logits'):
    per_type_metric = PerTypeMetric(epoch=epoch)
    model.eval()
    relation_type = 'implicit'

    entropy = None
    # if args.fusion == "ban":
    #    entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    # with open(os.path.join(args.data_root, args.feature_subdir, 'answer_ix_map.json')) as f:
    #     answer_ix_map = json.load(f)
    joint_loss, cap_loss, v_loss = 0, 0, 0
    all_preds = []
    ensemble_score = 0
    label2ans_path = os.path.join(args.data_folder, 'cache',
                                  'trainval_label2ans.pkl')
    label2ans = pickle.load(open(label2ans_path, 'rb'))
    with tqdm(total=len(dataloader), ncols=80) as pbar:
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
            if args.model == 'model' or args.model == 'left_right_model' or args.model == 'MMDLS':
                if (args.dataset == 'vqa' or args.dataset == 'vqa1.0'):
                    joint_loss = instance_bce_with_logits(joint_logits, target)
                    cap_loss = instance_bce_with_logits(cap_logits, target)
                    v_loss = instance_bce_with_logits(v_logits, target)
                    pred1 = (torch.sigmoid(joint_logits) + torch.sigmoid(cap_logits) + torch.sigmoid(v_logits)) / 3
                else:
                    joint_loss = F.cross_entropy(joint_logits, target.to(torch.long))
                    cap_loss = F.cross_entropy(cap_logits, target.to(torch.long))
                    v_loss = F.cross_entropy(v_logits, target.to(torch.long))
                    pred1 = (F.softmax(joint_logits, -1) + F.softmax(cap_logits, -1) + F.softmax(v_logits, -1)) / 3
                loss = (joint_loss + cap_loss + v_loss) / 3
                pred = (joint_logits + cap_logits + v_logits) / 3
                ensemble_score += compute_score_with_ensemble(joint_logits, cap_logits, v_logits, target).item()
            elif args.model == 'left_model' or args.model == 'MMDLS_left':
                cap_loss = instance_bce_with_logits(cap_logits, target)
                joint_loss = None
                v_loss = None
                pred = cap_logits
                pred1 = torch.sigmoid(cap_logits)
                loss = cap_loss
            elif args.model == 'right_model' or args.model == 'MMDLS_left':
                v_loss = instance_bce_with_logits(v_logits, target)
                joint_loss = None
                cap_loss = None
                pred = v_logits
                pred1 = torch.sigmoid(v_logits)
                loss = v_loss

            val_metrics.update_per_batch(model, target, (loss, joint_loss, cap_loss, v_loss),
                                         (pred, joint_logits, cap_logits, v_logits, pred1), v.shape[0],
                                         batch_multiplier, device, logits_key)

            pred_ans_ixs = pred.max(1)[1]

            # Create predictions file
            if args.dataset.lower() != 'vqa' and args.dataset.lower() != 'vqa1.0':
                target1 = torch.zeros(*pred.size()).cuda()
                target1.scatter_(1, target.view(-1, 1).to(torch.long), 1)
                target = target1
            for curr_ix, pred_ans_ix in enumerate(pred_ans_ixs):
                pred_ans = label2ans[int(pred_ans_ix)]
                all_preds.append({
                    'question_id': str(question_ids[curr_ix]),
                    'answer': str(pred_ans)
                })
                if not args.test or not args.test_does_not_have_answers:
                    per_type_metric.update_for_question_type(question_types[curr_ix],
                                                             target[curr_ix].cpu().data.numpy(),
                                                             pred[curr_ix].cpu().data.numpy())
            pbar.update(1)
    val_metrics.update_per_epoch()
    val_TBX.log_per_epoch('val', (val_metrics.loss, val_metrics.loss1[0], val_metrics.loss1[1], val_metrics.loss1[2]),
                          (val_metrics.score, val_metrics.score1[0], val_metrics.score1[1], val_metrics.score1[2],
                           val_metrics.score1[3]), ensemble_socre=ensemble_score / val_metrics.num_examples)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return {
        'all_preds': all_preds,
        'per_type_metric': per_type_metric,
        'entropy': entropy
    }


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()
    relation_type = dataloader.dataset.relation_type
    score = 0
    upper_bound = 0
    num_data = 0
    N = len(dataloader.dataset)
    entropy = None
    if model.module.fusion == "ban":
        entropy = torch.Tensor(model.module.glimpse).zero_().to(device)
    pbar = tqdm(total=len(dataloader), ncols=100)

    for i, (v, norm_bb, q, target, cap, t2i, _, question_type_id, _, bb, spa_adj_matrix,
            sem_adj_matrix) in enumerate(dataloader):
        batch_size = v.size(0)
        num_objects = v.size(1)

        v = Variable(v).to(device)
        norm_bb = Variable(norm_bb).to(device)
        q = Variable(q).to(device)
        target = Variable(target).to(device)
        cap = Variable(cap).to(device)
        t2i = Variable(t2i).to(device)

        pos_emb_L, sem_adj_matrix_L, spa_adj_matrix_L = prepare_graph_variables(
            relation_type, bb, sem_adj_matrix, spa_adj_matrix, num_objects,
            args.nongt_dim, args.imp_pos_emb_dim, args.spa_label_num,
            args.sem_label_num, device)
        pos_emb_R, sem_adj_matrix_R, spa_adj_matrix_R = pos_emb_L, sem_adj_matrix_L, spa_adj_matrix_L
        pred, att = model(v, t2i, cap, norm_bb, q, question_type_id, pos_emb_L, pos_emb_R,
                          sem_adj_matrix_L, sem_adj_matrix_R,
                          spa_adj_matrix_L, spa_adj_matrix_R)

        batch_score = compute_score_with_logits(
            pred, target, device).sum()
        score += batch_score
        upper_bound += (target.max(1)[0]).sum()
        num_data += pred.size(0)
        if att is not None and 0 < model.module.glimpse \
                and entropy is not None:
            entropy += calc_entropy(att.data)[:model.module.glimpse]
        pbar.update(1)
    # pbar.close()
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)
    model.train()
    return score, upper_bound, entropy


def calc_entropy(att):
    # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p + eps).log()).sum(2).sum(0)  # g
