from argparse import ArgumentParser, Namespace
import io
import os
import yaml
import torch
import re
import numpy as np

import opts
import utils
from models import BiLSTM_CRF, ResLSTM_CRF, TransformerCRF
from utils import misc_utils

opt = opts.model_opts()
config = yaml.load(open(opt.config, "r"))
config = Namespace(**config, **vars(opt))

device, devices_id = misc_utils.set_cuda(config)
config.device = device

src_vocab = utils.Dict()
src_vocab.loadFile(os.path.join(config.data, "src.vocab"))
tgt_vocab = utils.Dict()
tgt_vocab.loadFile(os.path.join(config.data, "tgt.vocab"))

if config.model == 'bilstm_crf':
    model = BiLSTM_CRF(src_vocab.size(), tgt_vocab.size(), config)
elif config.model == 'reslstm_crf':
    model = ResLSTM_CRF(src_vocab.size(), tgt_vocab.size(), config)
elif config.model == 'transformer_crf':
    model = TransformerCRF(src_vocab.size(), tgt_vocab.size(), config)
else:
    model = None
    raise NotImplementedError(config.model + " not implemented!")
checkpoint = torch.load(config.restore, lambda storage, loc: storage)
print(model.state_dict().keys())
print(checkpoint['model'].keys())
model.load_state_dict(checkpoint['model'])
model.eval()

model.to(device)
# TODO: half precision has limited support on cpu
# model = model.half()

# x = torch.ones(32, 20).long()
# lengths = torch.ones(32).long() * 20
# print(model(x, lengths))

ILLEGAL_REGEX = r"[^-\u4e00-\u9fff0-9a-zA-Z.、/]"


def preprocess(x: str):
    x = x.lower()
    x = re.sub("&amp;?", "&", x)
    x = re.sub("&#x0a;?", "", x)
    x = re.sub("&#x0020;?", "", x)
    x = re.sub("&#x000a;?", "", x)
    x = re.sub("(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", "", x)
    x = re.sub("\\[[^\\]]{1,3}\\]", "", x)
    x = re.sub(ILLEGAL_REGEX, "", x)
    x = re.sub('\\.{2,}', '', x)
    return x

oovs = {0: 'B', 1:'B'}

# testCats = ['cloth']
with io.open("testOut.txt", 'w+', encoding='utf-8') as fout:
    with io.open("../data/deepsegment/testdata/randomDescs.txt", encoding='utf-8') as fin:
        oriSrcList = []
        srcList = []
        srcIdList = []
        srcLenList = []
        # catList = []

        batch_size = config.batch_size
        cnt = 0
        for line in fin.readlines():
            arr = line.strip().split("\t")
            line = preprocess(arr[0])
            # cat = arr[1]
            chars = [c for c in line]
            ids = src_vocab.convertToIdx(chars, utils.UNK_WORD)
            # print(chars, ids)

            oriSrcList.append(arr[0])
            srcList.append(line)
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

        # srcList = srcList[:8]
        resList = []
        addOne = 1 if (len(srcList) % batch_size) else 0
        batch_num = len(srcList) // batch_size + addOne
        print('batchNum ', batch_num)
        for i in range(batch_num):
            print("batch ", i)
            startIdx = i * batch_size
            endIdx = min((i + 1) * batch_size, len(srcList))
            xs = srcIdList[startIdx:endIdx]
            lengths = srcLenList[startIdx:endIdx]

            maxLen = max(len(x) for x in xs)
            xs = [x + [utils.PAD] * (maxLen - len(x)) for x in xs]

            xs = torch.tensor(xs).to(device)
            lengths = torch.tensor(lengths).to(device)

            with torch.no_grad():
                score, tag_seq = model(xs, lengths, config.nbest)
                if config.nbest < 1:
                    for tags in tag_seq:
                        candidates = [''.join(tgt_vocab.convertToLabels(tags, utils.PAD))]
                        resList.append(candidates)
                else:
                    # tag_seq = tag_seq.cpu().numpy()
                    for nbest_tags in tag_seq:
                        # nbest_tags = np.transpose(nbest_tags)
                        candidates = [''.join(tgt_vocab.convertToLabels(tags, utils.PAD, oovs=oovs)) for tags in nbest_tags]
                        resList.append(candidates)

        for oriSrc, src, res in zip(oriSrcList, srcList, resList):
            fout.write(oriSrc + '\n')
            split_stses_list = []
            for nbest_y in res:
                split_stses = []
                tmp = []
                for i, (x, y) in enumerate(zip(src, nbest_y)):
                    if y == 'b' and i > 0:
                        split_stses.append(''.join(tmp))
                        tmp = []
                    tmp.append(x)
                if len(tmp) > 0:
                    split_stses.append(''.join(tmp))
                    tmp = []
                split_stses_list.append(split_stses)
            max_sts_num = max([len(split_stses) for split_stses in split_stses_list])
            for si in range(max_sts_num):
                tmp = []
                for i in range(len(split_stses_list)):
                    if si < len(split_stses_list[i]):
                        tmp.append(split_stses_list[i][si])
                    else:
                        tmp.append('')
                fout.write('\t'.join(tmp) + '\n')
            fout.write('\n')
