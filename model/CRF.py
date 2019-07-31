import torch
import torch.nn as nn
import utils
from utils.util_functions import *

class CRF(nn.Module):
    def __init__(self, num_tags, config):
        super().__init__()
        self.num_tags = num_tags
        self.batch_size = config.batch_size
        self.start_tag = -1
        self.end_tag = -2

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        self.trans.data[self.start_tag, :] = -10000. # no transition to SOS
        self.trans.data[:, self.end_tag] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, utils.PAD] = -10000. # no transition from PAD except to PAD
        self.trans.data[utils.PAD, :] = -10000. # no transition to PAD except from EOS
        self.trans.data[utils.PAD, self.end_tag] = 0.
        self.trans.data[utils.PAD, utils.PAD] = 0.

    def forward(self, h, lengths): # forward algorithm
        # initialize forward variables in log space
        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        score = torch.full((self.batch_size, self.num_tags), -10000.)
        score[:, self.start_tag] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = torch.where(mask_t, score_t, score) #score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[self.end_tag])
        return score # partition function

    def score(self, h, y, lengths): # calculate the score of a given sequence
        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        score = torch.Tensor(self.batch_size).fill_(0.)
        h = h.unsqueeze(3)
        y = torch.cat((torch.full((self.batch_size, 1), self.start_tag, dtype=torch.long), y), dim=-1)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += torch.where(mask_t, emit_t + trans_t, torch.zeros_like(emit_t))
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[self.end_tag, last_tag]
        return score

    def decode(self, h, lengths): # Viterbi decoding
        max_len = torch.max(lengths).item()
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        # initialize backpointers and viterbi variables in log space
        bptr = torch.LongTensor()
        score = torch.Tensor(self.batch_size, self.num_tags).fill_(-10000.)
        score[:, self.start_tag] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = torch.where(mask_t, score_t, score)  # score_t * mask_t + score * (1 - mask_t)
        score += self.trans[self.end_tag]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path