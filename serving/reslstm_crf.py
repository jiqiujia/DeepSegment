import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from .CRF import CRF
import utils
# torch.manual_seed(1)
from utils.util_functions import *

from .StackedLSTM import StackedLSTM

class ResLSTM_CRF(torch.jit.ScriptModule):
    # __constants__ = ['device']

    def __init__(self, vocab_size, tagset_size, config):
        super(ResLSTM_CRF, self).__init__()
        self.device = str(config.device)
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.tagset_size = tagset_size + 2

        self.bidirectional = config.bidirectional
        self.num_direction = 1
        if self.bidirectional:
            self.num_direction = 2

        self.word_embeds = nn.Embedding(vocab_size, config.embedding_dim)
        self.emb_drop = nn.Dropout(config.emb_dropout)
        self.bilstm = nn.LSTM(config.embedding_dim, config.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.lstm = StackedLSTM(self.num_layers, input_size=config.hidden_dim, config=config)
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(config.hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, config)

        self.device = torch.jit.Attribute(self.device, str)
        self.num_direction = torch.jit.Attribute(self.num_direction, int)
        self.hidden_dim = torch.jit.Attribute(self.hidden_dim, int)
        self.num_layers = torch.jit.Attribute(self.num_layers, int)

    @torch.jit.script_method
    def _get_lstm_features(self, sentence, lengths):
        batch_size = sentence.size(0)
        bilstm_hidden = (torch.randn(self.num_direction, batch_size, self.hidden_dim // 2).to(self.device),
                torch.randn(self.num_direction, batch_size, self.hidden_dim // 2).to(self.device))
        embeds = self.emb_drop(self.word_embeds(sentence))
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        bilstm_out, _ = self.bilstm(embeds, bilstm_hidden)
        bilstm_out, _ = nn.utils.rnn.pad_packed_sequence(bilstm_out, batch_first=True, padding_value=float(utils.PAD))

        lstm_out = []
        for t in range(sentence.size(1)):
            out_t, lstm_hidden = self.lstm(bilstm_out[:, t])
            lstm_out.append(out_t)
        lstm_out = torch.stack(lstm_out, 0)
        lstm_out = lstm_out.transpose(0, 1)

        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood(self, sentence, tags, lengths):
        feats = self._get_lstm_features(sentence, lengths)
        forward_score = self.crf(feats, lengths)
        gold_score = self.crf.score(feats, tags, lengths)
        return forward_score - gold_score

    @torch.jit.script_method
    def forward(self, sentence, lengths):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, lengths)

        # Find the best path, given the features.
        score, tag_seq = self.crf.decode(lstm_feats, lengths)
        return score, tag_seq
