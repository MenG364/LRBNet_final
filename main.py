import argparse
import json
import math
import os
import random
from os.path import join, exists

import torch
import torch.nn as nn
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, ConcatDataset, random_split

import utils
from LR_models.LR_model import build_LR_model
from LR_models.LR_model_left import build_LR_model_left
from LR_models.LR_model_right import build_LR_model_right
from LR_models.LR_model_left_right import build_LR_model_left_right

from eval import evaluate
from mini_mdls.mini_mdls import build_mmdls_model
from mini_mdls.mini_mdls_left import build_mmdls_model_left
from mini_mdls.mini_mdls_right import build_mmdls_model_right
from config.parser import parse_with_config
from dataset import Dictionary, VisualGenomeFeatureDataset, VQAFeatureDataset
from test import test
from train import train
from utils import trim_collate


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=2)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))


def parse_args():
    parser = argparse.ArgumentParser()
    '''
    For training logistics
    '''
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay_start', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.25)
    parser.add_argument('--lr_decay_step', type=int, default=2)
    parser.add_argument('--lr_decay_based_on_val', action='store_true',
                        help='Learning rate decay when val score descreases')
    parser.add_argument('--grad_accu_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=0.25)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=128)
    parser.add_argument('--output', type=str, default='saved_models2/')
    parser.add_argument('--save_optim', action='store_true',
                        help='save optimizer')
    parser.add_argument('--log_interval', type=int, default=-1,
                        help='Print log for certain steps')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test_split', type=str, default='test')
    parser.add_argument('--test_does_not_have_answers', action='store_true')

    '''
    loading trained models
    '''
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--pretrain_model', type=str, default="")

    '''
    For dataset
    '''
    parser.add_argument('--dataset', type=str, default='vqa',
                        choices=["vqa", "gqa", "clevr"])
    parser.add_argument('--data_folder', type=str, default='./data')
    parser.add_argument('--use_both', action='store_true',
                        help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true',
                        help='use visual genome dataset to train?')
    parser.add_argument('--adaptive', action='store_true',
                        help='adaptive or fixed number of regions')
    '''
    Model
    '''
    parser.add_argument('--relation_type', type=str, default='implicit',
                        choices=["spatial", "semantic", "implicit"])
    parser.add_argument('--fusion', type=str, default='mutan',
                        choices=["ban", "butd", "mutan"])
    parser.add_argument('--tfidf', action='store_true',
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help="op used in tfidf word embedding")
    parser.add_argument('--num_hid', type=int, default=1024)
    '''
    Fusion Hyperparamters
    '''
    parser.add_argument('--ban_gamma', type=int, default=1, help='glimpse')
    parser.add_argument('--mutan_gamma', type=int, default=2, help='glimpse')
    '''
    Hyper-params for relations
    '''
    # hyper-parameters for implicit relation
    parser.add_argument('--imp_pos_emb_dim', type=int, default=64,
                        help='geometric embedding feature dim')

    # hyper-parameters for explicit relation
    parser.add_argument('--spa_label_num', type=int, default=11,
                        help='number of edge labels in spatial relation graph')
    parser.add_argument('--sem_label_num', type=int, default=15,
                        help='number of edge labels in \
                              semantic relation graph')

    # shared hyper-parameters
    parser.add_argument('--dir_num', type=int, default=2,
                        help='number of directions in relation graph')
    parser.add_argument('--relation_dim', type=int, default=1024,
                        help='relation feature dim')
    parser.add_argument('--nongt_dim', type=int, default=20,
                        help='number of objects consider relations per image')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='number of attention heads \
                              for multi-head attention')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='number of graph propagation steps')
    parser.add_argument('--residual_connection', action='store_true',
                        help='Enable residual connection in relation encoder')
    parser.add_argument('--label_bias', action='store_true',
                        help='Enable bias term for relation labels \
                              in relation encoder')
    parser.add_argument('--expt_name', type=str, required=False)

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--fusion_num_layer', type=int, default=1)
    parser.add_argument('--model', type=str, default="model",
                        choices=["model", "left_model", "right_model", "left_right_model"])
    parser.add_argument('--use_count', action='store_true',
                        help='use counter')

    args = parse_with_config(parser)
    args.expt_save_dir = os.path.join('saved_models', args.expt_name)
    return args


if __name__ == '__main__':
    args = parse_args()

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available," +
                         "this code currently only support GPU.")
    n_device = torch.cuda.device_count()
    print("Found %d GPU cards for training" % (n_device))
    device = torch.device("cuda")
    batch_size = args.batch_size * n_device

    torch.backends.cudnn.benchmark = True

    if args.seed != -1:
        print("Predefined randam seed %d" % args.seed)
    else:
        # fix seed
        args.seed = random.randint(1, 10000)
        print("Choose random seed %d" % args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if "ban" == args.fusion:
        fusion_methods = args.fusion + "_" + str(args.ban_gamma)
    else:
        fusion_methods = args.fusion

    dictionary = Dictionary.load_from_file(
        join(args.data_folder, 'glove/dictionary.pkl'))
    dictionary1 = Dictionary.load_from_file(
        join(args.data_folder, 'glove/dictionary.pkl'))

    if args.test:
        test_dset = VQAFeatureDataset(
            args.dataset, 'test2015', dictionary, args.relation_type,
            adaptive=args.adaptive, pos_emb_dim=args.imp_pos_emb_dim,
            dataroot=args.data_folder)
    else:
        train_dset = VQAFeatureDataset(
            args.dataset, 'train', dictionary, args.relation_type,
            adaptive=args.adaptive, pos_emb_dim=args.imp_pos_emb_dim,
            dataroot=args.data_folder)
    val_dset = VQAFeatureDataset(
        args.dataset, 'val', dictionary, args.relation_type, adaptive=args.adaptive,
        pos_emb_dim=args.imp_pos_emb_dim, dataroot=args.data_folder)

    if args.model == 'model':
        model = build_LR_model(val_dset, args)
    elif args.model == 'left_model':
        model = build_LR_model_left(val_dset, args)
    elif args.model == 'right_model':
        model = build_LR_model_right(val_dset, args)
    elif args.model == 'left_right_model':
        model = build_LR_model_left_right(val_dset, args)
    elif args.model == 'MMDLS':
        model = build_mmdls_model(val_dset, args)
    elif args.model == 'MMDLS_left':
        model = build_mmdls_model_left(val_dset, args)
    elif args.model == 'MMDLS_right':
        model = build_mmdls_model_right(val_dset, args)

    # model.apply(weights_init)

    tfidf = None
    weights = None
    # if args.tfidf and (args.dataset == 'vqa' or args.dataset == 'vqa1.0'):
    #     tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'],
    #                                           dictionary1, dataroot=args.data_folder)
    model.w_emb.init_embedding(join(args.data_folder,
                                    'glove/glove6b_init_300d.npy'),
                               tfidf, weights)

    model = nn.DataParallel(model).to(device)

    if args.pretrain_model != "":
        # print("Loading weights from %s" % (args.pretrain_model))
        # if not os.path.exists(args.pretrain_model):
        #    raise ValueError("No such checkpoint exists!")
        # checkpoint = torch.load(args.pretrain_model)
        # state_dict = checkpoint.get('model_state_dict', checkpoint)
        # matched_state_dict = {}
        # unexpected_keys = set()
        # missing_keys = set()
        # for name, param in model.named_parameters():
        #    missing_keys.add(name)
        # for key, data in state_dict.items():
        #    if 'alpha' in key or 'beta' in key or 'gama' in key:
        #        continue
        #    elif key in missing_keys:
        #        matched_state_dict[key] = data
        #        missing_keys.remove(key)
        #    else:
        #        unexpected_keys.add(key)
        # print("Unexpected_keys:", list(unexpected_keys))
        # print("Missing_keys:", list(missing_keys))
        # model.load_state_dict(matched_state_dict, strict=False)

        # print("Loading weights from %s" % (args.pretrain_model))
        # if not os.path.exists(args.pretrain_model):
        #    raise ValueError("No such checkpoint exists!")
        # checkpoint = torch.load(args.pretrain_model)
        # state_dict = checkpoint.get('model_state_dict', checkpoint)
        # matched_state_dict = {}
        # unexpected_keys = set()
        # missing_keys = set()
        # for name, param in model.named_parameters():
        #    missing_keys.add(name)
        # for key, data in state_dict.items():
        #    if key in missing_keys:
        #        matched_state_dict[key] = data
        #        missing_keys.remove(key)
        #    if key.replace('models.0', 'model') in missing_keys:
        #        new_key = key.replace('models.0', 'model')
        #        matched_state_dict[new_key] = data
        #        missing_keys.remove(new_key)
        #    else:
        #        unexpected_keys.add(key)
        # print("Unexpected_keys:", list(unexpected_keys))
        # print("Missing_keys:", list(missing_keys))
        # matched_state_dict['module.joint_embedding.v_att.logits.h_bias']=matched_state_dict['module.joint_embedding.v_att.logits.h_bias'].repeat(1,8,1,1)
        # matched_state_dict['module.joint_embedding.v_att.logits.h_mat_v']=matched_state_dict['module.joint_embedding.v_att.logits.h_mat_v'].repeat(1,8,1,1)
        # matched_state_dict['module.joint_L.v_att.logits.h_bias']=matched_state_dict['module.joint_L.v_att.logits.h_bias'].repeat(1,8,1,1)
        # matched_state_dict['module.joint_L.v_att.logits.h_mat_v']=matched_state_dict['module.joint_L.v_att.logits.h_mat_v'].repeat(1,8,1,1)
        # model.load_state_dict(matched_state_dict, strict=False)
        # print("Loading weights from %s" % (args.pretrain_model))
        # if not os.path.exists(args.pretrain_model):
        #    raise ValueError("No such pretrain_model exists!")
        # checkpoint = torch.load(args.pretrain_model)
        # state_dict = checkpoint.get('model_state', checkpoint)
        # matched_state_dict = {}
        # unexpected_keys = set()
        # missing_keys = set()
        # for name, param in model.named_parameters():
        #    missing_keys.add(name)
        # for key1 in missing_keys.copy():
        #    key = key1.replace('.net.', '.')
        #    key = key.replace('R_model', 'implicit_relation')
        #    key = key.replace('L_model', 'implicit_relation')
        #    key = key.replace('joint_L', 'joint_embedding')
        #    key = key.replace('joint_R', 'joint_embedding')
        #    key = key.replace('right_', '')
        #    key = key.replace('left_', '')
        #    key = key.replace('total_', '')
        #    if 'cap_emb.rnn.rnn' in key:
        #        new_key = key.replace('cap_emb.rnn.rnn', 'q_emb.rnn')
        #        if new_key in state_dict.keys():
        #            matched_state_dict[key1] = state_dict[new_key]
        #            missing_keys.remove(key1)
        #    elif 'rnn.rnn' in key:
        #        new_key = key.replace('rnn.rnn', 'rnn')
        #        if new_key in state_dict.keys():
        #            matched_state_dict[key1] = state_dict[new_key]
        #            missing_keys.remove(key1)
        #    elif 'cap_att' in key:
        #        new_key = key.replace('cap_att', 'q_att')
        #        if new_key in state_dict.keys():
        #            matched_state_dict[key1] = state_dict[new_key]
        #            missing_keys.remove(key1)
        #    elif '_L' in key:
        #        new_key = key.replace('_L', '')
        #        if new_key in state_dict.keys():
        #            matched_state_dict[key1] = state_dict[new_key]
        #            missing_keys.remove(key1)
        #    elif '_R' in key:
        #        new_key = key.replace('_R', '')
        #        if new_key in state_dict.keys():
        #            matched_state_dict[key1] = state_dict[new_key]
        #            missing_keys.remove(key1)
        #    elif key in state_dict.keys():
        #        matched_state_dict[key1] = state_dict[key]
        #        missing_keys.remove(key1)
        #    else:
        #        unexpected_keys.add(key1)
        # print("Unexpected_keys:", list(unexpected_keys))
        # print("Missing_keys:", list(missing_keys))
        # model.load_state_dict(matched_state_dict, strict=False)

        print("Loading weights from %s" % (args.pretrain_model))
        if not os.path.exists(args.pretrain_model):
            raise ValueError("No such pretrain_model exists!")
        checkpoint = torch.load(args.pretrain_model)
        state_dict = checkpoint.get('model_state', checkpoint)
        matched_state_dict = {}
        unexpected_keys = set()
        missing_keys = set()

        for name, param in model.named_parameters():
            missing_keys.add(name)
        for key1 in missing_keys.copy():
            key = key1.replace('joint_L', 'joint_embedding')
            key = key.replace('joint_R', 'joint_embedding')
            key = key.replace('right_', '')
            key = key.replace('left_', '')
            if 'cap_emb.rnn.rnn' in key:
                new_key = key.replace('cap_emb.rnn.rnn', 'q_emb.rnn')
                if new_key in state_dict.keys():
                    matched_state_dict[key1] = state_dict[new_key]
                    missing_keys.remove(key1)
            elif 'rnn.rnn' in key:
                new_key = key.replace('rnn.rnn', 'rnn')
                if new_key in state_dict.keys():
                    matched_state_dict[key1] = state_dict[new_key]
                    missing_keys.remove(key1)
            elif 'cap_att' in key:
                new_key = key.replace('cap_att', 'q_att')
                if new_key in state_dict.keys():
                    matched_state_dict[key1] = state_dict[new_key]
                    missing_keys.remove(key1)
            elif '_L' in key:
                new_key = key.replace('_L.1.', '')
                new_key = key.replace('_L.0.', '')
                new_key = key.replace('_L', '')
                if new_key in state_dict.keys():
                    matched_state_dict[key1] = state_dict[new_key]
                    missing_keys.remove(key1)
            elif '_R' in key:
                new_key = key.replace('_R.0.', '')
                new_key = key.replace('_R.1.', '')
                new_key = key.replace('_R', '')
                if new_key in state_dict.keys():
                    matched_state_dict[key1] = state_dict[new_key]
                    missing_keys.remove(key1)
            elif key in state_dict.keys():
                matched_state_dict[key1] = state_dict[key]
                missing_keys.remove(key1)
            else:
                unexpected_keys.add(key1)

        print("Unexpected_keys:", list(unexpected_keys))
        print("Missing_keys:", list(missing_keys))
        matched_state_dict['module.v_relation.v_transform.main.0.weight_v'] = F.pad(
            matched_state_dict['module.v_relation.v_transform.main.0.weight_v'],
            (0, 2054 - 2048,
             0, 0))
        matched_state_dict['module.joint_embedding.v_att.logits.h_bias'] = matched_state_dict[
            'module.joint_embedding.v_att.logits.h_bias'].repeat(1, 8, 1, 1)
        matched_state_dict['module.joint_embedding.v_att.logits.h_mat_v'] = matched_state_dict[
            'module.joint_embedding.v_att.logits.h_mat_v'].repeat(1, 8, 1, 1)
        matched_state_dict['module.joint_L.v_att.logits.h_bias'] = matched_state_dict[
            'module.joint_L.v_att.logits.h_bias'].repeat(1, 8, 1, 1)
        matched_state_dict['module.joint_L.v_att.logits.h_mat_v'] = matched_state_dict[
            'module.joint_L.v_att.logits.h_mat_v'].repeat(1, 8, 1, 1)
        matched_state_dict['module.v_relation.implicit_relation_R.self_weights.main.1.weight_v'] = matched_state_dict[
                                                                                                       'module.v_relation.implicit_relation_R.self_weights.main.1.weight_v'][
                                                                                                   :, :1024]

        matched_state_dict['module.joint_R.v_att.logits.h_bias'] = matched_state_dict[
            'module.joint_R.v_att.logits.h_bias'].repeat(1, 8, 1, 1)
        matched_state_dict['module.joint_R.v_att.logits.h_mat_v'] = matched_state_dict[
            'module.joint_R.v_att.logits.h_mat_v'].repeat(1, 8, 1, 1)
        if args.model != 'MMDLS':
            if args.dataset == "gqa":
                matched_state_dict['module.classifier.main.3.bias'] = matched_state_dict[
                                                                          'module.classifier.main.3.bias'][
                                                                      :1852]
                matched_state_dict['module.classifier.main.3.weight_v'] = matched_state_dict[
                                                                              'module.classifier.main.3.weight_v'][
                                                                          :1852]
                matched_state_dict['module.left_classifier.main.3.bias'] = matched_state_dict[
                                                                               'module.left_classifier.main.3.bias'][
                                                                           :1852]
                matched_state_dict['module.left_classifier.main.3.weight_v'] = matched_state_dict[
                                                                                   'module.left_classifier.main.3.weight_v'][
                                                                               :1852]
                matched_state_dict['module.right_classifier.main.3.bias'] = matched_state_dict[
                                                                                'module.right_classifier.main.3.bias'][
                                                                            :1852]
                matched_state_dict['module.right_classifier.main.3.weight_v'] = matched_state_dict[
                                                                                    'module.right_classifier.main.3.weight_v'][
                                                                                :1852]
            if args.dataset == 'clevr':
                matched_state_dict['module.classifier.main.3.bias'] = matched_state_dict[
                                                                          'module.classifier.main.3.bias'][
                                                                      :28]
                matched_state_dict['module.classifier.main.3.weight_v'] = matched_state_dict[
                                                                              'module.classifier.main.3.weight_v'][:28]
                matched_state_dict['module.left_classifier.main.3.bias'] = matched_state_dict[
                                                                               'module.left_classifier.main.3.bias'][
                                                                           :28]
                matched_state_dict['module.left_classifier.main.3.weight_v'] = matched_state_dict[
                                                                                   'module.left_classifier.main.3.weight_v'][
                                                                               :28]
                matched_state_dict['module.right_classifier.main.3.bias'] = matched_state_dict[
                                                                                'module.right_classifier.main.3.bias'][
                                                                            :28]
                matched_state_dict['module.right_classifier.main.3.weight_v'] = matched_state_dict[
                                                                                    'module.right_classifier.main.3.weight_v'][
                                                                                :28]
            if args.dataset == 'vqa1.0':
                matched_state_dict['module.classifier.main.3.bias'] = matched_state_dict[
                                                                          'module.classifier.main.3.bias'][
                                                                      :2185]
                matched_state_dict['module.classifier.main.3.weight_v'] = matched_state_dict[
                                                                              'module.classifier.main.3.weight_v'][
                                                                          :2185]
                matched_state_dict['module.left_classifier.main.3.bias'] = matched_state_dict[
                                                                               'module.left_classifier.main.3.bias'][
                                                                           :2185]
                matched_state_dict['module.left_classifier.main.3.weight_v'] = matched_state_dict[
                                                                                   'module.left_classifier.main.3.weight_v'][
                                                                               :2185]
                matched_state_dict['module.right_classifier.main.3.bias'] = matched_state_dict[
                                                                                'module.right_classifier.main.3.bias'][
                                                                            :2185]
                matched_state_dict['module.right_classifier.main.3.weight_v'] = matched_state_dict[
                                                                                    'module.right_classifier.main.3.weight_v'][
                                                                                :2185]
        else:
            matched_state_dict['module.v_relation.v_transform.main.0.weight_v'] = F.pad(
                matched_state_dict['module.v_relation.v_transform.main.0.weight_v'],
                (0, 2054 - 2048,
                 0, 0))

        # matched_state_dict['module.q_emb.rnn.rnn.weight_ih_l0'] = matched_state_dict['module.q_emb.rnn.rnn.weight_ih_l0'][:,:300]
        # matched_state_dict['module.cap_emb.rnn.rnn.weight_ih_l0'] = matched_state_dict['module.cap_emb.rnn.rnn.weight_ih_l0'][:,:300]
        model.load_state_dict(matched_state_dict, strict=False)

    # use train & val splits to optimize, only available for vqa, not vqa_cp
    if args.use_both and args.test == False:
        length = len(val_dset)
        trainval_concat_dset = ConcatDataset([train_dset, val_dset])
        if args.use_vg:
            trainval_concat_dsets_split = random_split(
                trainval_concat_dset,
                [int(0.2 * length),
                 len(trainval_concat_dset) - int(0.2 * length)])
        else:
            trainval_concat_dsets_split = random_split(
                trainval_concat_dset,
                [int(0.1 * length),
                 len(trainval_concat_dset) - int(0.1 * length)])
        concat_list = [trainval_concat_dsets_split[1]]

        # use a portion of Visual Genome dataset
        if args.use_vg:
            vg_train_dset = VisualGenomeFeatureDataset(args.dataset,
                                                       'train', train_dset.features, train_dset.normalized_bb,
                                                       train_dset.bb, dictionary,
                                                       train_dset.captions, train_dset.objects_num,
                                                       adaptive=train_dset.adaptive,
                                                       pos_boxes=train_dset.pos_boxes,
                                                       dataroot=args.data_folder)
            vg_val_dset = VisualGenomeFeatureDataset(args.dataset,
                                                     'val', val_dset.features, val_dset.normalized_bb,
                                                     val_dset.bb, dictionary,
                                                     val_dset.captions, val_dset.objects_num,
                                                     adaptive=val_dset.adaptive,
                                                     pos_boxes=val_dset.pos_boxes,
                                                     dataroot=args.data_folder)
            concat_list.append(vg_train_dset)
            concat_list.append(vg_val_dset)
        final_train_dset = ConcatDataset(concat_list)
        final_eval_dset = trainval_concat_dsets_split[0]
        train_loader = DataLoaderX(final_train_dset, batch_size, shuffle=True, pin_memory=True,
                                   num_workers=args.num_worker, collate_fn=trim_collate)
        eval_loader = DataLoaderX(final_eval_dset, batch_size, pin_memory=True,
                                  shuffle=False, num_workers=args.num_worker,
                                  collate_fn=trim_collate)
    else:
        if args.test:
            test_loader = DataLoaderX(test_dset, batch_size, shuffle=False, pin_memory=True,
                                      num_workers=args.num_worker, collate_fn=trim_collate)
        else:
            eval_loader = DataLoaderX(val_dset, batch_size, shuffle=False, pin_memory=True,
                                      num_workers=args.num_worker, collate_fn=trim_collate)
            train_loader = DataLoaderX(train_dset, batch_size, shuffle=False, pin_memory=True,
                                       num_workers=args.num_worker, collate_fn=trim_collate)

    output_meta_folder = join(args.output, "LRBNet_new_2%s" % args.relation_type)
    utils.create_dir(output_meta_folder)
    args.output = output_meta_folder + "/%s_%s_%s_%d" % (
        fusion_methods, args.relation_type,
        args.dataset, args.seed)
    if exists(args.output) and os.listdir(args.output):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output))
    utils.create_dir(args.output)
    with open(join(args.output, 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    logger = utils.Logger(join(args.output, 'log.txt'))

    if args.test:
        test(model, test_loader, args)
        test_dset.close_h5_file()
    elif args.val:
        evaluate(model, eval_loader, args, device)
    else:
        train(model, train_loader, eval_loader, args, device)
        train_dset.close_h5_file()
        # check_file='saved_models'
        # check=os.listdir(check_file)
        # for c in check:
        #     if c.endswith('.pth'):
        #         args.checkpoint=os.path.join(check_file,c)
        #         evaluate(model,eval_loader, args, device)

    val_dset.close_h5_file()
