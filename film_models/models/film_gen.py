#!/usr/bin/env python3

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from ReGAT_models.fc import FCNet
from film_models.models.layers import init_modules


class FiLMGen(nn.Module):
    def __init__(self,
                 output_batchnorm=False,
                 bidirectional=False,
                 encoder_type='FCNet',
                 decoder_type='linear',
                 act_type='linear',
                 gamma_option='linear',
                 gamma_baseline=1,
                 module_dim=128,
                 ):
        super(FiLMGen, self).__init__()
        self.encoder_type = encoder_type
        self.output_batchnorm = output_batchnorm
        self.bidirectional = bidirectional
        self.num_dir = 2 if self.bidirectional else 1
        self.gamma_option = gamma_option
        self.gamma_baseline = gamma_baseline
        self.module_dim = module_dim
        if act_type == 'linear':
            act_type = None

        self.func_list = {
            'linear': None,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'exp': torch.exp,
        }
        self.cond_feat_size = 2 * self.module_dim  # FiLM params per Graph node

        self.encoder_model = init_model(self.encoder_type, [self.module_dim, self.cond_feat_size], act=act_type,
                                        dropout=0, bias=True)
        if self.output_batchnorm:
            self.output_bn = nn.BatchNorm1d(self.module_dim, affine=True)
            self.output_bn1 = nn.BatchNorm1d(self.module_dim, affine=True)

        init_modules(self.modules())

    def forward(self, x):
        '''
            input:
                x: [batch_size, num_rois, v_dim]
                adj_matrix: [batch_size, num_rois, num_rois, num_labels]
            output:
                film: [batch, num_rois, v_dim * 2]

        '''
        film_pre_mod = self.encoder_model(x)
        film = self.modify_output(film_pre_mod, gamma_option=self.gamma_option,
                                  gamma_shift=self.gamma_baseline)
        gamma = film[:, :, :self.module_dim]
        beta = film[:, :, self.module_dim:]
        if self.output_batchnorm:
            gamma = gamma.view(-1, self.module_dim)
            beta = beta.view(-1, self.module_dim)
            gamma = self.output_bn(gamma)
            beta = self.output_bn1(beta)
            gamma = gamma.view(x.size(0), -1, self.module_dim)
            beta = beta.view(x.size(0), -1, self.module_dim)
        return [gamma,beta]

    def modify_output(self, out, gamma_option='linear', gamma_scale=1, gamma_shift=0,
                      beta_option='linear', beta_scale=1, beta_shift=0):
        gamma_func = self.func_list[gamma_option]
        beta_func = self.func_list[beta_option]

        gs = slice(0, self.module_dim)
        bs = slice(self.module_dim, 2 * self.module_dim)

        if gamma_func is not None:
            out[:, :, gs] = gamma_func(out[:, :, gs])
        if gamma_scale != 1:
            out[:, :, gs] = out[:, :, gs] * gamma_scale
        if gamma_shift != 0:
            out[:, :, gs] = out[:, :, gs] + gamma_shift
        if beta_func is not None:
            out[:, :, bs] = beta_func(out[:, :, bs])
        if beta_scale != 1:
            out[:, :, bs] = out[:, :, bs] * beta_scale
        if beta_shift != 0:
            out[:, :, bs] = out[:, :, bs] + beta_shift
        return out


def init_model(model_type, dims, act='ReLU', dropout=0, bias=True):
    if model_type == 'FCNet':
        return FCNet(dims, act=act, dropout=dropout, bias=bias)
    else:
        print('Model type ' + str(model_type) + ' not yet implemented.')
        raise (NotImplementedError)
