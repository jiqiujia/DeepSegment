import io
import os
import yaml
import torch
import re
import numpy as np

import logging
import torch
import json

import opts
import utils
from models import BiLSTM_CRF, ResLSTM_CRF, TransformerCRF
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_valid_input(texts):
    return len(texts) > 0 and isinstance(texts, list) and all(isinstance(s, str) and s.strip() for s in texts)


ILLEGAL_REGEX = r"[^-\u4e00-\u9fff0-9a-zA-Z.、/]"


class Server(object):

    def __init__(self, config):
        self.config = config

        self.src_vocab = utils.Dict()
        self.src_vocab.loadFile(os.path.join(config.data, "src.vocab"))
        self.tgt_vocab = utils.Dict()
        self.tgt_vocab.loadFile(os.path.join(config.data, "tgt.vocab"))

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
        x = re.sub("&#x0020;?", "", x, flags=re.IGNORECASE)
        x = re.sub("&#x000a;?", "", x, flags=re.IGNORECASE)
        x = re.sub("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", "", x, flags=re.IGNORECASE)
        x = re.sub("\\[[^\\]]{1,3}\\]", "", x)
        x = re.sub('[-.=]{2,}', '', x)
        return x

    def batchify(self, texts: List[str]):
        oriSrcList = []
        srcList = []
        srcIdList = []
        srcLenList = []

        batch_size = self.config.batch_size

        cnt = 0
        for line in texts:
            line = self.preprocess(line)
            oriSrcList.append(line)

            line = line.lower()
            line = re.sub(ILLEGAL_REGEX, "", line)
            srcList.append(line)

            chars = [c for c in line]
            ids = self.src_vocab.convertToIdx(chars, utils.UNK_WORD)
            srcIdList.append(ids)
            srcLenList.append(len(ids))
            # catList.append(cat)
            cnt += 1
        # packed rnn sequence needs lengths to be in decreasing order
        indices = np.argsort(srcLenList)[::-1]
        oriSrcList = [oriSrcList[i] for i in indices]
        srcList = [srcList[i] for i in indices]
        srcIdList = [srcIdList[i] for i in indices]
        srcLenList = [srcLenList[i] for i in indices]
        # catList = [catList[i] for i in indices]

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

            yield ori_xs, xs, x_ids, lengths

    def get_aligned_stses(self, ori_x: str, x: str, y: str):
        oi = xi = 0
        seg_stses = []
        tmpx = []
        while oi < len(ori_x) and xi < len(x):
            if ori_x[oi] in ',，。！!?？【】':
                oi += 1
                continue
            if y[xi] == 'b' and xi > 0:
                while ori_x[oi].lower() != x[xi].lower():
                    oi += 1
                seg_stses.append(''.join(tmpx))
                tmpx = []
            if ori_x[oi].lower() == x[xi].lower():
                xi += 1
            tmpx += [ori_x[oi]]
            oi += 1
        if oi < len(ori_x) and len(tmpx) > 0:
            tmpx += ori_x[oi:]
            seg_stses.append(''.join(tmpx))

        return seg_stses

    def batch_predict_line(self, texts: List[str]):

        stses_list = []
        for ori_xs, xs, x_ids, lengths in self.batchify(texts):

            maxLen = max(len(x) for x in x_ids)
            x_ids = [x + [utils.PAD] * (maxLen - len(x)) for x in x_ids]

            x_ids = torch.tensor(x_ids).to(self.config.device)
            lengths = torch.tensor(lengths).to(self.config.device)

            resList = []
            with torch.no_grad():
                score, tag_seq = self.model(x_ids, lengths, self.config.nbest)
                if self.config.nbest <= 1:
                    for tags in tag_seq:
                        candidates = [''.join(self.tgt_vocab.convertToLabels(tags, utils.PAD))]
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
    print(server.batch_predict_line(["为什么你的脸一到换季就干，手也干，起皮，过敏！更换季节是一部分原因更多的是因为洗面奶含碱性太高 经常用碱性洗面奶洗脸洗掉了保湿的皮脂膜！"]))