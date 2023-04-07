import torch
import torch.nn as nn

from components import nonlinearity
from components.language_model import QuestionEmbedding
from components.language_model import WordEmbedding
from components.multi_modal_core import MultiModalCore


class Ramen(nn.Module):
    def __init__(self, config):
        super(Ramen, self).__init__()
        self.config = config
        self.mmc_net = MultiModalCore(config)
        self.w_emb = WordEmbedding(config.w_emb_size, 300)
        self.w_emb.init_embedding(config.glove_file)
        self.q_emb = QuestionEmbedding(300, self.config.q_emb_dim, 1, bidirect=True, dropout=0,
                                       rnn_type=config.question_rnn_type,
                                       dropout_before_rnn=config.question_dropout_before_rnn,
                                       dropout_after_rnn=config.question_dropout_after_rnn)

        clf_in_size = config.mmc_aggregator_dim * 2
        classifier_layers = []
        for ix, size in enumerate(config.classifier_sizes):
            in_s = clf_in_size if ix == 0 else config.classifier_sizes[ix - 1]
            out_s = size
            lin = nn.Linear(in_s, out_s)
            classifier_layers.append(lin)
            classifier_layers.append(getattr(nonlinearity, config.classifier_nonlinearity)())
            classifier_layers.append(nn.Dropout(p=config.classifier_dropout))

        if config.pre_classification_dropout is not None and config.pre_classification_dropout > 0:
            self.pre_classification_dropout = nn.Dropout(p=config.pre_classification_dropout)
        else:
            self.pre_classification_dropout = None
        self.pre_classification_layers = nn.Sequential(*classifier_layers)
        self.classifier = nn.Linear(out_s, config.num_ans_candidates)

    def forward(self, v, b, q, a=None, qlen=None):
        """Forward

        v: [batch, num_objs, v_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits
        """
        batch_size, num_objs, v_emb_dim = v.size()
        b = b[:, :, :4]
        q = self.w_emb(q)
        q_emb = self.q_emb(q, qlen)
        # x, mmc_aggregated = self.mmc_net(v, b, q_emb)  # B x num_objs x num_hid and B x num_hid
        x = self.mmc_net(v, b, q_emb)  # B x num_objs x num_hid and B x num_hid


        if not self.config.disable_late_fusion:
            x = torch.cat((x, q), dim=2)
            curr_size = x.size()
            if not self.config.disable_batch_norm_for_late_fusion:
                x = x.view(-1, curr_size[2])
                x = self.batch_norm_before_aggregation(x)
                x = x.view(curr_size)
            # x = self.aggregator_dropout(x)
            x_aggregated = self.aggregator(x)

        if self.pre_classification_dropout is not None:
            mmc_aggregated = self.pre_classification_dropout(mmc_aggregated)
        final_emb = self.pre_classification_layers(mmc_aggregated)
        logits = self.classifier(final_emb)
        out = {'logits': logits, 'q_emb': q_emb}
        return out
