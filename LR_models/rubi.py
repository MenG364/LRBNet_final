import torch
import torch.nn as nn

from LR_models.mlp import MLP
from utils import grad_mul_const


class RUBiNet(nn.Module):
    """
    Wraps another model
    The original model must return a dictionnary containing the 'logits' key (predictions before softmax)
    Returns:
        - logits: the original predictions of the model
        - logits_q: the predictions from the question-only branch
        - logits_rubi: the updated predictions from the model by the mask.
    => Use `logits_rubi` and `logits_q` for the loss
    """

    def __init__(self, model, output_size, classif, end_classif=True):
        super().__init__()
        self.net = model
        self.c_1 = MLP(**classif)
        self.end_classif = end_classif
        if self.end_classif:
            self.c_2 = nn.Linear(output_size, output_size)

    def forward(self, v, Q, c, b, q, question_type, implicit_pos_emb_L, implicit_pos_emb_R, sem_adj_matrix_L=None,
                sem_adj_matrix_R=None,
                spa_adj_matrix_L=None, spa_adj_matrix_R=None, a=None, qlen=None):
        out = {}
        # model prediction
        net_out = self.net(v, Q, c, b, q, question_type, implicit_pos_emb_L, implicit_pos_emb_R, sem_adj_matrix_L=None,
                           sem_adj_matrix_R=None,
                           spa_adj_matrix_L=None, spa_adj_matrix_R=None)
        logits = net_out['logits']
        q_embedding = net_out['q_emb']  # N * q_emb
        q_embedding = grad_mul_const(q_embedding, 0.0)  # don't backpropagate through question encoder
        q_pred = self.c_1(q_embedding)
        fusion_pred = [logit * torch.sigmoid(q_pred) for logit in logits]

        if self.end_classif:
            q_out = self.c_2(q_pred)
        else:
            q_out = q_pred

        out['logits'] = net_out['logits']
        out['logits_rubi'] = fusion_pred
        out['logits_q'] = q_out
        #out['att'] = net_out['att']
        return out

    def process_answers(self, out, key=''):
        out = self._process_answers(out)
        out = self._process_answers(out, key='_rubi')
        out = self._process_answers(out, key='_q')
        return out

    def _process_answers(self, out, key=''):
        batch_size = out[f'logits{key}'].shape[0]
        _, pred = out[f'logits{key}'].data.max(1)
        pred.squeeze_()
        if batch_size != 1:
            out[f'answers{key}'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids{key}'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers{key}'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids{key}'] = [pred.item()]
        return out
