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

    def forward(self, h, lengths): # forward algorithm
        # initialize forward variables in log space
        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len).to(self.device) < lengths.unsqueeze(1)
        score = torch.full((h.shape[0], self.num_tags), -10000.).to(self.device)
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
        mask = torch.arange(max_len).expand(len(lengths), max_len).to(self.device) < lengths.unsqueeze(1)
        score = torch.Tensor(h.shape[0]).fill_(0.).to(self.device)
        h = h.unsqueeze(3)
        y = torch.cat((torch.full((h.shape[0], 1), self.start_tag, dtype=torch.long).to(self.device), y), dim=-1)
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
        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        # initialize backpointers and viterbi variables in log space
        bptr = torch.zeros((h.shape[0], max_len, self.num_tags), dtype=torch.int32)
        score = torch.ones(h.shape[0], self.num_tags).fill_(-10000.)
        score[:, self.start_tag] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr[:, t] = bptr_t#.unsqueeze(1)
            # bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = torch.where(mask_t, score_t, score)  # score_t * mask_t + score * (1 - mask_t)
        score += self.trans[self.end_tag]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(h.shape[0]):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b, :y]):
                x = bptr_t[x].item()
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_score, best_path

    def decode_nbest(self, feats, lengths, nbest):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                lengths: (batch, 1)
            output:
                decode_idx: (batch, nbest, seq_len) decoded sequence
                path_score: (batch, nbest) corresponding score for each sequence (to be implementated)
                nbest decode for sentence with one token is not well supported, to be optimized
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert (tag_size == self.num_tags)

        max_len = torch.max(lengths)
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        ## calculate sentence length for each sentence
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.trans.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :].clone()  # bat_size * to_target_size
        ## initial partition [batch_size, tag_size]
        partition_history.append(partition.view(batch_size, tag_size, 1).expand(batch_size, tag_size, nbest))
        # iter over last scores
        for idx, cur_values in seq_iter:
            if idx == 1:
                cur_values = cur_values.view(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size,
                                                                                                           tag_size,
                                                                                                           1).expand(
                    batch_size, tag_size, tag_size)
            else:
                # previous to_target is current from_target
                # partition: previous results log(exp(from_target)), #(batch_size * nbest * from_target)
                # cur_values: batch_size * from_target * to_target
                cur_values = cur_values.view(batch_size, tag_size, 1, tag_size).expand(batch_size, tag_size, nbest,
                                                                                       tag_size) + partition.contiguous().view(
                    batch_size, tag_size, nbest, 1).expand(batch_size, tag_size, nbest, tag_size)
                ## compare all nbest and all from target
                cur_values = cur_values.view(batch_size, tag_size * nbest, tag_size)
                # print "cur size:",cur_values.size()
            partition, cur_bp = torch.topk(cur_values, nbest, 1)
            ## cur_bp/partition: [batch_size, nbest, tag_size], id should be normize through nbest in following backtrace step
            # print partition[:,0,:]
            # print cur_bp[:,0,:]
            # print "nbest, ",idx
            if idx == 1:
                cur_bp = cur_bp * nbest
            partition = partition.transpose(2, 1)
            cur_bp = cur_bp.transpose(2, 1)

            # print partition
            # exit(0)
            # partition: (batch_size * to_target * nbest)
            # cur_bp: (batch_size * to_target * nbest) Notice the cur_bp number is the whole position of tag_size*nbest, need to convert when decode
            partition_history.append(partition)
            ## cur_bp: (batch_size,nbest, tag_size) topn source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            ## mask[idx] ? mask[idx-1]
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1, 1).expand(batch_size, tag_size, nbest), 0)
            # print cur_bp[0]
            back_points.append(cur_bp)
        ### add score to final end_tag
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, tag_size, nbest).transpose(1,
                                                                                                                 0).contiguous()  ## (batch_size, seq_len, nbest, tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = lengths.view(batch_size, 1, 1, 1).expand(batch_size, 1, tag_size, nbest) - 1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, nbest, 1)
        ### calculate the score from last partition to end state (and then select the end_tag from it)
        last_values = last_partition.expand(batch_size, tag_size, nbest, tag_size) + self.trans.view(1, tag_size,
                                                                                                           1,
                                                                                                           tag_size).expand(
            batch_size, tag_size, nbest, tag_size)
        last_values = last_values.view(batch_size, tag_size * nbest, tag_size)
        end_partition, end_bp = torch.topk(last_values, nbest, 1)
        ## end_partition: (batch, nbest, tag_size)
        end_bp = end_bp.transpose(2, 1)
        # end_bp: (batch, tag_size, nbest)
        pad_zero = torch.zeros(batch_size, tag_size, nbest).long()
        # if self.gpu:
        pad_zero = pad_zero.to(self.device)
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size, nbest)

        ## select end ids in end_tag
        pointer = end_bp[:, self.end_tag, :]  ## (batch_size, nbest)
        insert_last = pointer.contiguous().view(batch_size, 1, 1, nbest).expand(batch_size, 1, tag_size, nbest)
        back_points = back_points.transpose(1, 0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last[0]
        # exit(0)
        ## copy the ids of last position:insert_last to back_points, though the last_position index
        ## last_position includes the length of batch sentences
        # print "old:", back_points[9,0,:,:]
        back_points.scatter_(1, last_position, insert_last)
        ## back_points: [batch_size, seq_length, tag_size, nbest]
        # print "new:", back_points[9,0,:,:]
        # exit(0)
        # print pointer[2]
        '''
        back_points: in simple demonstratration
        x,x,x,x,x,x,x,x,x,7
        x,x,x,x,x,4,0,0,0,0
        x,x,6,0,0,0,0,0,0,0
        '''

        back_points = back_points.transpose(1, 0).contiguous()
        # print back_points[0]
        ## back_points: (seq_len, batch, tag_size, nbest)
        ## decode from the end, padded position ids are 0, which will be filtered in following evaluation
        decode_idx = torch.LongTensor(seq_len, batch_size, nbest)
        # if self.gpu:
        decode_idx = decode_idx.to(self.device)
        decode_idx[-1] = pointer.data / nbest
        # print "pointer-1:",pointer[2]
        # exit(0)
        # use old mask, let 0 means has token
        for idx in range(len(back_points) - 2, -1, -1):
            # print "pointer: ",idx,  pointer[3]
            # print "back:",back_points[idx][3]
            # print "mask:",mask[idx+1,3]
            new_pointer = torch.gather(back_points[idx].view(batch_size, tag_size * nbest), 1,
                                       pointer.contiguous().view(batch_size, nbest))
            decode_idx[idx] = new_pointer.data / nbest
            # # use new pointer to remember the last end nbest ids for non longest
            pointer = new_pointer + pointer.contiguous().view(batch_size, nbest) * mask[idx].view(batch_size, 1).expand(
                batch_size, nbest).long()

        # exit(0)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        ## decode_idx: [batch, seq_len, nbest]
        # print decode_idx[:,:,0]
        # print "nbest:",nbest
        # print "diff:", decode_idx[:,:,0]- decode_idx[:,:,4]
        # print decode_idx[:,0,:]
        # exit(0)

        ### calculate probability for each sequence
        scores = end_partition[:, :, self.end_tag]
        ## scores: [batch_size, nbest]
        max_scores, _ = torch.max(scores, 1)
        minus_scores = scores - max_scores.view(batch_size, 1).expand(batch_size, nbest)
        path_score = F.softmax(minus_scores, 1)
        ## path_score: [batch_size, nbest]
        # exit(0)
        return path_score, decode_idx