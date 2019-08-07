import torch.nn as nn
from .CRF import CRF
import utils
# torch.manual_seed(1)
from utils.util_functions import *

class BiLSTM_CRF(torch.jit.ScriptModule):
    __constants__ = ['num_tags']

    def __init__(self, vocab_size, tagset_size, config):
        super(BiLSTM_CRF, self).__init__()
        self.device = config.device
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size + 2
        self.start_idx = -2
        self.end_idx = -1
        self.batch_size = config.batch_size

        self.bidirectional = config.bidirectional
        self.num_direction = 1
        if self.bidirectional:
            self.num_direction = 2

        self.word_embeds = nn.Embedding(vocab_size, config.embedding_dim)
        self.emb_drop = nn.Dropout(config.emb_dropout)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim,
                            num_layers=config.num_layers, bidirectional=True,
                            dropout=config.dropout, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.num_direction * config.hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, config)

    def init_hidden(self, batch_size):
        # return (torch.zeros(self.num_direction * self.num_layers, batch_size, self.hidden_dim),
        #         torch.zeros(self.num_direction * self.num_layers, batch_size, self.hidden_dim))
        return (torch.randn(self.num_direction * self.num_layers, batch_size, self.hidden_dim),
                torch.randn(self.num_direction * self.num_layers, batch_size, self.hidden_dim))


    def _get_lstm_features(self, sentence, lengths):
        batch_size = sentence.size(0)
        self.hidden = self.init_hidden(batch_size)
        embeds = self.emb_drop(self.word_embeds(sentence))
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, padding_value=utils.PAD)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood(self, sentence, tags, lengths):
        feats = self._get_lstm_features(sentence, lengths)
        forward_score = self.crf(feats, lengths)
        gold_score = self.crf.score(feats, tags, lengths)
        return forward_score - gold_score

    @torch.jit.script_method
    def forward(self, sentence, lengths):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence, lengths)

        # Find the best path, given the features.
        score = self.crf.decode(lstm_feats, lengths)
        return score