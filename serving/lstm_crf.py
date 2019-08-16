import torch.nn as nn
from serving.CRF import CRF
import utils
# torch.manual_seed(1)
from utils.util_functions import *

class BiLSTM_CRF(torch.jit.ScriptModule):
    __constants__ = ['num_tags', 'hidden', 'num_layers', 'embedding_dim',
                     'batch_first', 'num_direction', 'hidden_dim']

    def __init__(self, vocab_size, tagset_size, config):
        super(BiLSTM_CRF, self).__init__()
        self.device = config.device
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size + 2
        self.batch_size = config.batch_size
        self.batch_first = True

        self.bidirectional = config.bidirectional
        self.num_direction = 2

        self.word_embeds = nn.Embedding(vocab_size, config.embedding_dim)
        self.emb_drop = nn.Dropout(config.emb_dropout)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim,
                            num_layers=config.num_layers, bidirectional=True,
                            dropout=config.dropout, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.num_direction * config.hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, config)

    @torch.jit.script_method
    def forward(self, sentence, lengths):
        # Get the emission scores from the BiLSTM
        batch_size = sentence.shape[0]
        hidden = (torch.randn(self.num_direction * self.num_layers, batch_size, self.hidden_dim),
                torch.randn(self.num_direction * self.num_layers, batch_size, self.hidden_dim))
        embeds = self.emb_drop(self.word_embeds(sentence))
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=self.batch_first)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=self.batch_first, padding_value=float(utils.PAD))
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        # lstm_feats = self._get_lstm_features(sentence, lengths)

        # Find the best path, given the features.
        score = self.crf.decode(lstm_feats, lengths)
        return score

if __name__ == '__main__':
    import opts
    import yaml
    from argparse import ArgumentParser, Namespace
    from utils import misc_utils

    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))
    device, devices_id = misc_utils.set_cuda(config)
    config.device = device
    model = BiLSTM_CRF(2, 2, config)

    # model.forward([[1,1]], [2])