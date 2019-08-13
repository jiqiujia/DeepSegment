import torch.nn as nn
from .CRF import CRF
import utils
from models.transformer import TransformerEncoder
# torch.manual_seed(1)
from utils.util_functions import *


class TransformerCRF(nn.Module):
    # __constants__ = ['device']

    def __init__(self, vocab_size, tagset_size, config):
        super(TransformerCRF, self).__init__()
        self.device = config.device
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size + 2
        self.batch_size = config.batch_size

        config.src_vocab_size = vocab_size
        config.emb_size = config.embedding_dim
        config.hidden_size = config.hidden_dim
        self.transformer = TransformerEncoder(config=config, padding_idx=utils.PAD)

        self.hidden2tag = nn.Linear(config.hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, config)

    def neg_log_likelihood(self, sentence, tags, lengths):
        sentence = sentence.transpose(0, 1)
        feats = self.transformer(sentence)
        feats = self.hidden2tag(feats)
        feats = feats.transpose(0, 1)
        forward_score = self.crf(feats, lengths)
        gold_score = self.crf.score(feats, tags, lengths)
        return forward_score - gold_score

    def forward(self, sentence, lengths):
        sentence = sentence.transpose(0, 1)
        feats = self.transformer(sentence)
        feats = self.hidden2tag(feats)
        feats = feats.transpose(0, 1)

        # Find the best path, given the features.
        score, tag_seq = self.crf.decode(feats, lengths)
        return score, tag_seq
