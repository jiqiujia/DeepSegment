import io
import os
import yaml
import torch
import re
import numpy as np

import logging
import torch
import utils
from models import BiLSTM_CRF, ResLSTM_CRF, TransformerCRF
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_valid_input(texts):
    return len(texts) > 0 and isinstance(texts, list) and all(isinstance(s, str) and s.strip() for s in texts)


ILLEGAL_REGEX = r"[^-\u4e00-\u9fff0-9a-zA-Z.、/*]"


class Server(object):

    def __init__(self, config):
        self.config = config

        self.src_vocab = utils.Dict()
        self.src_vocab.loadFile(os.path.join(config.data, "src.vocab"))
        self.tgt_vocab = utils.Dict()
        self.tgt_vocab.loadFile(os.path.join(config.data, "tgt.vocab"))

        self.num_tags = self.tgt_vocab.size() + 2

        if config.model == 'bilstm_crf':
            self.model = BiLSTM_CRF(self.src_vocab.size(), self.tgt_vocab.size(), config)
        elif config.model == 'reslstm_crf':
            self.model = ResLSTM_CRF(self.src_vocab.size(), self.tgt_vocab.size(), config)
        elif config.model == 'transformer_crf':
            self.model = TransformerCRF(self.src_vocab.size(), self.tgt_vocab.size(), config)
        else:
            self.model = None
            raise NotImplementedError(config.model + " not implemented!")
        checkpoint = torch.load(config.restore, lambda storage, loc: storage)
        print(self.model.state_dict().keys())
        print(checkpoint['model'].keys())
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.model.to(config.device)

        self.oovs = {0: 'B', 1: 'B'}

    def preprocess(self, x: str):
        x = re.sub("&amp;?", "&", x, flags=re.IGNORECASE)
        x = re.sub("&#x0a;?", "", x, flags=re.IGNORECASE)
        x = re.sub("&apos[;，]?", "'", x, flags=re.IGNORECASE)
        x = re.sub("&#x0020;?", "", x, flags=re.IGNORECASE)
        x = re.sub("&#x000a;?", "", x, flags=re.IGNORECASE)
        x = re.sub("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", "", x,
                   flags=re.IGNORECASE)
        x = re.sub("\\[[^\\]]{1,3}\\]", "", x)
        x = re.sub('[-.=]{2,}', '', x)
        x = re.sub(']]$', '', x)
        x = re.sub("", "", x)
        x = re.sub('×', 'x', x)
        return x

    def batchify(self, texts: List[str], batch_size=None):
        oriSrcList = []
        srcList = []
        srcIdList = []
        srcLenList = []
        tgtMaskList = []

        if batch_size is None:
            batch_size = self.config.batch_size

        cnt = 0
        for line in texts:
            ori_x = self.preprocess(line)
            oriSrcList.append(ori_x)

            x = ori_x.lower()
            x = re.sub(ILLEGAL_REGEX, "", x)
            srcList.append(x)

            chars = [c for c in x]
            ids = self.src_vocab.convertToIdx(chars, utils.UNK_WORD)
            srcIdList.append(ids)
            srcLenList.append(len(ids))

            tgtMask = self.get_constrained_tag_masks(ori_x, x)
            tgtMaskList.append(tgtMask)
            cnt += 1
        # packed rnn sequence needs lengths to be in decreasing order
        indices = np.argsort(srcLenList)[::-1]
        back_indices = np.argsort(indices)
        oriSrcList = [oriSrcList[i] for i in indices]
        srcList = [srcList[i] for i in indices]
        srcIdList = [srcIdList[i] for i in indices]
        srcLenList = [srcLenList[i] for i in indices]
        tgtMaskList = [tgtMaskList[i] for i in indices]

        addOne = 1 if (len(srcList) % batch_size) else 0
        batch_num = len(srcList) // batch_size + addOne
        print('batchNum ', batch_num)

        for i in range(batch_num):
            print("batch ", i)
            startIdx = i * batch_size
            endIdx = min((i + 1) * batch_size, len(srcList))
            ori_xs = oriSrcList[startIdx:endIdx]
            xs = srcList[startIdx:endIdx]
            x_ids = srcIdList[startIdx:endIdx]
            lengths = srcLenList[startIdx:endIdx]
            tgt_masks = tgtMaskList[startIdx:endIdx]
            batch_back_indices = back_indices[startIdx:endIdx]

            yield ori_xs, xs, x_ids, lengths, tgt_masks, batch_back_indices

    def get_aligned_stses(self, ori_x: str, x: str, y: str):
        oi = xi = 0
        seg_stses = []
        tmpx = []
        while oi < len(ori_x) and xi < len(x):
            # if ori_x[oi] in ',，。！!?？【】':
            #     oi += 1
            #     continue
            if y[xi] == 'b' and xi > 0:
                while ori_x[oi].lower() != x[xi]:
                    tmpx += [ori_x[oi]]
                    oi += 1
                seg_stses.append(''.join(tmpx))
                tmpx = []
            if ori_x[oi].lower() == x[xi]:
                xi += 1
            tmpx += [ori_x[oi]]
            oi += 1
        if oi < len(ori_x):
            tmpx += ori_x[oi:]
        if len(tmpx) > 0:
            seg_stses.append(''.join(tmpx))

        return seg_stses

    def get_constrained_tag_masks(self, ori_x, x):
        oi = xi = 0
        mask = np.zeros(self.num_tags, dtype=np.float32)
        mask[self.tgt_vocab.lookup('b')] = 1
        masks = []
        b_mark = False
        while oi < len(ori_x) and xi < len(x):
            if ori_x[oi].lower() == x[xi]:
                xi += 1
                if b_mark:
                    masks.append(mask)
                else:
                    masks.append(np.ones_like(mask))
            if ori_x[oi] in ",，;；":
                b_mark = True
            else:
                b_mark = False
            oi += 1
        masks = np.stack(masks)
        return masks

    def batch_predict_line(self, texts: List[str], batch_size=None):

        stses_list = []
        back_indices = []
        for ori_xs, xs, x_ids, lengths, tgt_masks, batch_back_indice in self.batchify(texts, batch_size):

            maxLen = max(len(x) for x in x_ids)
            x_ids = [x + [utils.PAD] * (maxLen - len(x)) for x in x_ids]
            tgt_masks = [np.concatenate((mask, np.ones((maxLen - mask.shape[0], self.num_tags)))) for mask in tgt_masks]
            tgt_masks = np.stack(tgt_masks)

            x_ids = torch.tensor(x_ids).to(self.config.device)
            lengths = torch.tensor(lengths).to(self.config.device)
            tgt_masks = torch.tensor(tgt_masks).float()#.to(self.config.device)

            resList = []
            with torch.no_grad():
                score, tag_seq = self.model(x_ids, lengths, self.config.nbest, tgt_masks)
                if self.config.nbest <= 1:
                    for tags in tag_seq:
                        candidates = [''.join(self.tgt_vocab.convertToLabels(tags, utils.PAD, oovs=self.oovs))]
                        resList.append(candidates)
                else:
                    tag_seq = tag_seq.cpu().numpy()
                    for nbest_tags in tag_seq:
                        nbest_tags = np.transpose(nbest_tags)
                        # print(nbest_tags)
                        candidates = [''.join(self.tgt_vocab.convertToLabels(tags, utils.PAD, oovs=self.oovs)) for tags
                                      in nbest_tags]
                        resList.append(candidates)

            for i, res in enumerate(resList):
                stses = self.get_aligned_stses(ori_xs[i], xs[i], res[0])
                stses_list.append(stses)
            back_indices.append(batch_back_indice)
        back_indices = np.concatenate(back_indices)
        stses_list = [stses_list[i] for i in back_indices]
        return stses_list


if __name__ == '__main__':
    from argparse import Namespace
    import opts
    import yaml
    from utils import misc_utils

    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))

    device, devices_id = misc_utils.set_cuda(config)
    config.device = device

    server = Server(config)
    print(server.batch_predict_line(
        ["test, test", "为什么你的脸一到换季就干，手也干，起皮，过敏！更换季节是一部分原因更多的是因为洗面奶含碱性太高 经常用碱性洗面奶洗脸洗掉了保湿的皮脂膜！"]))

    with io.open("testOut.txt", 'w+', encoding='utf-8') as fout:
        with io.open("../data/deepsegment/testdata/randomDescs.txt", encoding='utf-8') as fin:
            lines = [line.strip().split('\t')[0] for line in fin.readlines()]
            res_stses = server.batch_predict_line(lines, 32)
            for ori, res in zip(lines, res_stses):
                fout.write(ori + '\t' + str(len(ori)) + '\n')
                for sts in res:
                    fout.write(sts + '\t' + str(len(ori)) + '\n')
                fout.write('#\t' + str(len(ori)) + '\n')
