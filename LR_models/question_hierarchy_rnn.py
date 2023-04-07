import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn


class question_hierarchy_rnn(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """

    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='LSTM'):
        super().__init__()

        self.unigram_conv = nn.Conv1d(in_dim, in_dim, 1, stride=1, padding=0)
        self.bigram_conv = nn.Conv1d(in_dim, in_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(in_dim, in_dim, 3, stride=1, padding=2, dilation=2)
        self.max_pool = nn.MaxPool2d((3, 1))
        self.tanh = nn.Tanh()
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU \
            if rnn_type == 'GRU' else None
        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

    def forward(self, words, hidden, lens):
        # Image: B x 512 x 196
        # question, lens = rnn.pad_packed_sequence(question)
        # question = question.permute(1, 0)                  # Ques : B x L
        # words = self.embed(question).permute(0, 2, 1)

        words = words.permute(0, 2, 1)  # Words: B x L x dim
        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2)  # B x dim x L
        bigrams = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)  # B x dim x L
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2)  # B x dim x L
        words = words.permute(0, 2, 1)

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0, 2, 1)  # B x L x 512

        phrase_packed = nn.utils.rnn.pack_padded_sequence(phrase, lengths=lens.cpu(), batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        sentence_packed, hidden = self.rnn(phrase_packed, hidden)
        sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)  # B x L x 512

        return sentence, hidden
