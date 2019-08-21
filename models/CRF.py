import torch
import torch.nn as nn
import utils
from utils.util_functions import *
import torch.nn.functional as F


class CRF(nn.Module):
    # __constants__ = ['device']

    def __init__(self, num_tags, config):
        super().__init__()
        self.num_tags = num_tags
        self.start_tag = -2
        self.end_tag = -1
        self.device = config.device

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        # self.trans.data[utils.UNK, :] = -10000. # no transition to UNK
        # self.trans.data[:, utils.UNK] = -10000. # no transition from UNK
        # self.trans.data[self.start_tag, :] = -10000. # no transition to SOS
        # self.trans.data[:, self.end_tag] = -10000. # no transition from EOS except to PAD
        # self.trans.data[:, utils.PAD] = -10000. # no transition from PAD except to PAD
        # self.trans.data[utils.PAD, :] = -10000. # no transition to PAD except from EOS
        # self.trans.data[utils.PAD, self.end_tag] = 0.
        # self.trans.data[utils.PAD, utils.PAD] = 0.

    def forward(self, h, lengths):  # forward algorithm
        # initialize forward variables in log space
        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len).to(self.device) < lengths.unsqueeze(1)
        score = torch.full((h.shape[0], self.num_tags), -10000.).to(self.device)
        score[:, self.start_tag] = 0.
        trans = self.trans.unsqueeze(0)  # [1, C, C]
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans  # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C, C] -> [B, C]
            score = torch.where(mask_t, score_t, score)  # score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[self.end_tag])
        return score  # partition function

    def score(self, h, y, lengths):  # calculate the score of a given sequence
        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len).to(self.device) < lengths.unsqueeze(1)
        score = torch.Tensor(h.shape[0]).fill_(0.).to(self.device)
        h = h.unsqueeze(3)
        y = torch.cat((torch.full((h.shape[0], 1), self.start_tag, dtype=torch.long).to(self.device), y), dim=-1)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += torch.where(mask_t, emit_t + trans_t, torch.zeros_like(emit_t))
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[self.end_tag, last_tag]
        return score

    def decode(self, h, lengths):  # Viterbi decoding
        h = h.cpu()
        lengths = lengths.cpu()
        trans = self.trans.cpu()
        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        # initialize backpointers and viterbi variables in log space
        bptr = torch.zeros((h.shape[0], max_len, self.num_tags), dtype=torch.int32)
        score = torch.ones(h.shape[0], self.num_tags).fill_(-10000.)
        score[:, self.start_tag] = 0.

        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + trans  # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2)  # best previous scores and tags
            score_t += h[:, t]  # plus emission scores
            bptr[:, t] = bptr_t  # .unsqueeze(1)
            # bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = torch.where(mask_t, score_t, score)  # score_t * mask_t + score * (1 - mask_t)
        score += trans[self.end_tag]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(h.shape[0]):
            x = best_tag[b]  # best tag
            for bptr_t in reversed(bptr[b, :lengths[b]]):
                x = bptr_t[x].item()
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_score, best_path

    # 要注意每一维的涵义，以及trans矩阵是从哪一维forward到哪一维
    def decode_nbest(self, h, lengths, nbest):  # Viterbi decoding
        batch_size = h.shape[0]
        max_len = h.shape[1]
        tag_size = h.shape[2]
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        # initialize backpointers and viterbi variables in log space
        bptr = torch.zeros((batch_size, max_len, self.num_tags, nbest), dtype=torch.long)  # nbest from_tags for to_tags
        score = torch.ones(batch_size, self.num_tags, nbest).fill_(-10000.) # 当前每个tag的nbest个结果
        score[:, self.start_tag, :] = 0.

        for t in range(h.size(1)):  # recursion through the sequence
            mask_t = mask[:, t].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest)
            # nbest 放在最后一维，考虑下面要 //nbest
            score_t = score.view(batch_size, 1, tag_size, nbest).expand(batch_size, tag_size, tag_size, nbest) + \
                      self.trans.view(1, tag_size, tag_size, 1).expand(batch_size, tag_size, tag_size, nbest)  # [B, 1, NB, C] -> [B, C, NB, C]
            score_t = score_t.view(batch_size, tag_size, tag_size * nbest)
            score_t, bptr_t = torch.topk(score_t, nbest, dim=2)  # best previous scores and tags  # [B, C, NB]
            score_t += h[:, t].view(batch_size, tag_size, 1)  # plus emission scores
            bptr[:, t] = bptr_t  # .unsqueeze(1)
            bptr[:, t].masked_fill_(torch.bitwise_not(mask_t), 0)
            score = torch.where(mask_t, score_t, score)  # score_t * mask_t + score * (1 - mask_t)
        score_last = score.view(batch_size, 1, tag_size, nbest).expand(batch_size, tag_size, tag_size, nbest) + \
                      self.trans.view(1, tag_size, tag_size, 1).expand(batch_size, tag_size, tag_size, nbest)  # [B, 1, NB, C] -> [B, C, NB, C]
        score = score_last.view(batch_size, tag_size, tag_size * nbest)
        best_score, best_tags = torch.topk(score, nbest, dim=2)  # [B, NB] 最后一个状态，每个batch得到nbest个标签
        best_score = best_score[:, self.end_tag]
        best_tags = best_tags[:, self.end_tag].long()

        # back-tracking
        # decode_idx = torch.zeros(batch_size, max_len, nbest, dtype=torch.int32)
        best_path = [[tags // nbest] for tags in best_tags]
        for b in range(h.shape[0]):
            tags = best_tags[b]  # best tags
            for bptr_t in reversed(bptr[b, :lengths[b]]):
                tags = torch.gather(bptr_t.reshape(-1), 0, tags)
                best_path[b].append(tags // nbest)
            best_path[b].pop()
            best_path[b].reverse()

        batch_nbest_paths = []
        for bpath in best_path:
            nbest_paths = []
            for i in range(nbest):
                path = []
                for tags in bpath:
                    path.append(tags[i].item())
                nbest_paths.append(path)
            batch_nbest_paths.append(nbest_paths)
        return best_score, batch_nbest_paths
