import torch
from torch import nn
from ramen_models.rnn import RNN
from components import nonlinearity


class LinearWithDropout(nn.Module):
    def __init__(self, in_s, out_s, dropout_type='1d', dropout_p=0):
        super(LinearWithDropout, self).__init__()
        self.linear = nn.Linear(in_s, out_s)
        self.dropout_p = dropout_p
        self.dropout_type = dropout_type
        if dropout_type == '2d':
            self.dropout = nn.Dropout2d(p=dropout_p)
        else:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        # x: B x num_objects x feat_dim
        batch_size, num_objs, feat_dim = x.shape
        x1 = self.linear(x)
        if self.dropout_p > 0:
            x1 = self.dropout(x1)
        return x1


class MultiModalCore(nn.Module):
    """
    Concatenates visual and linguistic features and passes them through an MLP.
    """

    def __init__(self, v_dim, q_emb_dim, mmc_sizes=None, input_dropout=0., disable_early_fusion=False,
                 mmc_nonlinearity='Swish', mmc_dropout=0):
        super(MultiModalCore, self).__init__()
        if mmc_sizes is None:
            mmc_sizes = [1024, 1024, 1024, 1024]
        self.v_dim = v_dim
        self.q_emb_dim = q_emb_dim
        self.mmc_sizes = mmc_sizes
        self.mmc_layers = []
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.disable_early_fusion = disable_early_fusion
        self.mmc_connection = 'residual'

        # Create MLP with early fusion in the first layer followed by batch norm
        for mmc_ix in range(len(mmc_sizes)):
            if mmc_ix == 0:
                if disable_early_fusion:
                    in_s = self.v_dim
                else:
                    in_s = self.v_dim + self.q_emb_dim
                # self.batch_norm_fusion = nn.BatchNorm1d(in_s)
            else:
                in_s = mmc_sizes[mmc_ix - 1]
            out_s = mmc_sizes[mmc_ix]
            # lin = nn.Linear(in_s, out_s)
            lin = LinearWithDropout(in_s, out_s, dropout_p=mmc_dropout)
            self.mmc_layers.append(lin)
            nonlin = getattr(nonlinearity, mmc_nonlinearity)()
            self.mmc_layers.append(nonlin)

        self.mmc_layers = nn.ModuleList(self.mmc_layers)
        self.batch_norm_mmc = nn.BatchNorm1d(self.mmc_sizes[-1])

        #Aggregation
        # if not self.config.disable_late_fusion:
        #     out_s += config.q_emb_dim
        #     if not self.config.disable_batch_norm_for_late_fusion:
        #         self.batch_norm_before_aggregation = nn.BatchNorm1d(out_s)
        # self.aggregator = RNN(out_s, config.mmc_aggregator_dim, nlayers=config.mmc_aggregator_layers,
        #                       bidirect=True)

    def __batch_norm(self, x, num_objs, flat_emb_dim):
        x = x.view(-1, flat_emb_dim)
        x = self.batch_norm_mmc(x)
        x = x.view(-1, num_objs, flat_emb_dim)
        return x

    def forward(self, x, labels=None):
        """

        :param v: B x num_objs x emb_size
        :param b: B x num_objs x 6
        :param q: B x emb_size
        :param labels
        :return:
        """
        x = self.input_dropout(x)
        num_objs = x.shape[1]
        emb_size = x.shape[2]

        curr_lin_layer = -1

        # Pass through MMC
        for mmc_layer in self.mmc_layers:
            # if isinstance(mmc_layer, nn.Linear):
            if isinstance(mmc_layer, LinearWithDropout):
                curr_lin_layer += 1
                mmc_out = mmc_layer(x)
                x_new = mmc_out
                if curr_lin_layer > 0:
                    if self.mmc_connection == 'residual':
                        x_new = x + mmc_out
                if x_new is None:
                    x_new = mmc_out
                x = x_new
            else:
                x = mmc_layer(x)

        x = x.view(-1, self.mmc_sizes[-1])
        x = self.batch_norm_mmc(x)
        x = x.view(-1, num_objs, self.mmc_sizes[-1])

        # if not self.config.disable_late_fusion:
        #     x = torch.cat((x, q), dim=2)
        #     curr_size = x.size()
        #     if not self.config.disable_batch_norm_for_late_fusion:
        #         x = x.view(-1, curr_size[2])
        #         x = self.batch_norm_before_aggregation(x)
        #         x = x.view(curr_size)
        #     # x = self.aggregator_dropout(x)
        #     x_aggregated = self.aggregator(x)

        return x

    def update_dropouts(self, input_dropout_p=None, hidden_dropout_p=None):
        self.input_dropout.p = input_dropout_p
        for mmc_layer in self.mmc_layers:
            if isinstance(mmc_layer, LinearWithDropout):
                mmc_layer.dropout.p = hidden_dropout_p
