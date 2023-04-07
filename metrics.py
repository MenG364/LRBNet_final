import torch
import torch.nn as nn
import time
import os
import json
from tensorboardX import SummaryWriter


def compute_score_with_logits(preds, labels, device, logits_key='logits'):
    # argmax
    logits = preds[logits_key] if isinstance(preds ,dict) else preds
    logits = torch.max(logits, 1)[1].data
    logits = logits.view(-1, 1)
    one_hots = torch.zeros(*labels.size()).to(device)
    one_hots.scatter_(1, logits, 1)
    scores = (one_hots * labels)
    return scores


class Metrics:
    """
    Stores accuracy (score), loss and timing info
    """

    def __init__(self):
        self.loss = 0
        self.loss1 = [0, 0, 0]
        self.raw_score = 0
        self.raw_score1 = [0, 0, 0, 0, 0]
        self.score = 0
        self.score1 = [0, 0, 0, 0, 0]
        self.normalized_score = 0
        self.start_time = time.time()
        self.end_time = 0
        self.total_norm = 0
        self.count_norm = 0
        self.num_examples = 0
        self.upper_bound = 0
        self.norm = 0
        self.batch_score = 0
        self.batch_score1 = [0, 0, 0, 0]

        self.reset_start_time()

    def update_per_batch(self, model, answers, loss, pred, curr_size, batch_multiplier, device, logits_key='logits'):

        self.total_norm += nn.utils.clip_grad_norm_(model.parameters(), 0.25).item()
        self.count_norm += 1

        if len(answers.size())==2:
            upper_bound = answers.max(1)[0].sum().item()

            self.batch_score = compute_score_with_logits(pred[0], answers.data, device, logits_key).sum().item()
            for i in range(1, len(pred)):
                if pred[i] != None:
                    self.batch_score1[i - 1] = compute_score_with_logits(pred[i], answers, device,logits_key).sum().item()
        else:
            upper_bound = answers.size()[0]

            self.batch_score = (pred[0].argmax(1) == answers.data).sum().item()
            for i in range(1, len(pred)):
                if pred[i] != None:
                    self.batch_score1[i - 1] = (pred[i].argmax(1) == answers.data).sum().item()

        self.upper_bound += upper_bound
        self.loss += loss[0].item() * curr_size * batch_multiplier
        for i in range(1, len(loss)):
            if loss[i] != None:
                self.loss1[i - 1] += loss[i].item() * curr_size * batch_multiplier
        self.raw_score += self.batch_score
        for i in range(len(self.batch_score1)):
            self.raw_score1[i] += self.batch_score1[i]
        self.num_examples += curr_size


    def update_per_epoch(self):
        self.loss /= self.num_examples
        for i in range(len(self.loss1)):
            self.loss1[i] /= self.num_examples
        self.raw_score = 100 * self.raw_score / self.num_examples
        for i in range(len(self.batch_score1)):
            self.raw_score1[i] = 100 * self.raw_score1[i] / self.num_examples

        self.upper_bound = self.upper_bound / self.num_examples
        self.normalized_score = self.raw_score / self.upper_bound
        self.score = self.raw_score
        for i in range(len(self.raw_score1)):
            self.score1[i] = self.raw_score1[i]

        self.norm = self.total_norm / self.count_norm
        self.end_time = time.time()

    def print(self, epoch):
        print("Epoch {} Score {:.4f} Loss {}".format(epoch, 100 * self.raw_score / self.num_examples,
                                                     self.loss / self.num_examples))

    def reset_start_time(self):
        self.start_time = time.time()


def accumulate_metrics(epoch, train_metrics, val_metrics, val_per_type_metric,
                       best_val_score,
                       best_val_epoch, save_val_metrics=True):
    stats = {
        "epoch": epoch,

        "train_loss": float(train_metrics.loss),
        "train_raw_score": float(train_metrics.raw_score),
        "train_normalized_score": float(train_metrics.normalized_score),
        "train_upper_bound": float(train_metrics.upper_bound),
        "train_score": float(train_metrics.score),
        "train_num_examples": train_metrics.num_examples,

        "train_time": train_metrics.end_time - train_metrics.start_time,
        "val_time": val_metrics.end_time - val_metrics.start_time
    }
    if save_val_metrics:
        stats["val_raw_score"] = float(val_metrics.raw_score)
        stats["val_normalized_score"] = float(val_metrics.normalized_score)
        stats["val_upper_bound"] = float(val_metrics.upper_bound)
        stats["val_loss"] = float(val_metrics.loss)
        stats["val_score"] = float(val_metrics.score)
        stats["val_num_examples"] = val_metrics.num_examples
        stats["val_per_type_metric"] = val_per_type_metric.get_json()

        stats["best_val_score"] = float(best_val_score)
        stats["best_epoch"] = best_val_epoch

    print(json.dumps(stats, indent=4))
    return stats


class TBX:
    """
       Stores accuracy (score), loss and timing info
       """

    def __init__(self, seed):
        self.batch_count = 0
        self.train_epoch = 0
        self.val_epoch = 0
        # self.writer = SummaryWriter('saved_models/run/exp' + str(seed), comment='VQA')
        self.writer = SummaryWriter("saved_models" + '/run_new2/exp' + str(seed),comment='VQA')

    def add_graph(self, model, input):
        self.writer.add_graph(model, input)

    def log_per_batch(self, name, loss, raw_score, loss_per_batch, score_per_batch):
        self.writer.add_scalar(name + "/batch_loss", loss[0], self.batch_count)
        self.writer.add_scalar(name + "/batch_joint_loss", loss[1], self.batch_count)
        self.writer.add_scalar(name + "/batch_cap_loss", loss[2], self.batch_count)
        self.writer.add_scalar(name + "/batch_v_loss", loss[3], self.batch_count)

        self.writer.add_scalar(name + "/batch_score", raw_score[0], self.batch_count)
        self.writer.add_scalar(name + "/batch_jscore", raw_score[1], self.batch_count)
        self.writer.add_scalar(name + "/batch_cscore", raw_score[2], self.batch_count)
        self.writer.add_scalar(name + "/batch_vscore", raw_score[3], self.batch_count)
        self.writer.add_scalar(name + "/batch_zscore", raw_score[4], self.batch_count)

        self.writer.add_scalar(name + "/loss_per_batch", loss_per_batch, self.batch_count)
        self.writer.add_scalar(name + "/score_per_batch", score_per_batch, self.batch_count)
        self.batch_count += 1

    def log_per_epoch(self, name, loss, raw_score, norm=None,lr=None,ensemble_socre=None):
        self.writer.add_scalar(name + "/loss_per_epoch", loss[0], self.train_epoch if name=='train' else self.val_epoch)
        self.writer.add_scalar(name + "/joint_loss_per_epoch", loss[1], self.train_epoch if name=='train' else self.val_epoch)
        self.writer.add_scalar(name + "/cap_loss_per_epoch", loss[2], self.train_epoch if name=='train' else self.val_epoch)
        self.writer.add_scalar(name + "/v_loss_per_epoch", loss[3], self.train_epoch if name=='train' else self.val_epoch)

        self.writer.add_scalar(name + "/score_per_epoch", raw_score[0], self.train_epoch if name=='train' else self.val_epoch)
        self.writer.add_scalar(name + "/jscore_per_epoch", raw_score[1], self.train_epoch if name=='train' else self.val_epoch)
        self.writer.add_scalar(name + "/cscore_per_epoch", raw_score[2], self.train_epoch if name=='train' else self.val_epoch)
        self.writer.add_scalar(name + "/vscore_per_epoch", raw_score[3], self.train_epoch if name=='train' else self.val_epoch)
        self.writer.add_scalar(name + "/zscore_per_epoch", raw_score[4], self.train_epoch if name=='train' else self.val_epoch)
        if ensemble_socre is not None:
            self.writer.add_scalar(name + "/escore_per_epoch", ensemble_socre,self.train_epoch if name == 'train' else self.val_epoch)
        if name == 'train':
            self.writer.add_scalar('lr' + "/lr", lr, self.train_epoch)

        if norm is not None:
            self.writer.add_scalar(name + "/norm_per_epoch", norm, self.train_epoch if name=='train' else self.val_epoch)
        if name == 'train':
            self.train_epoch += 1
        else:
            self.val_epoch += 1
