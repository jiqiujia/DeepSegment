from argparse import ArgumentParser, Namespace
import io
import os
import yaml
import torch
import re
import numpy as np

import opts
import utils
from model import BiLSTM_CRF
from utils import misc_utils

opt = opts.model_opts()
config = yaml.load(open(opt.config, "r"))
config = Namespace(**config, **vars(opt))

src_vocab = utils.Dict()
src_vocab.loadFile(os.path.join(config.data, "src.vocab"))
tgt_vocab = utils.Dict()
tgt_vocab.loadFile(os.path.join(config.data, "tgt.vocab"))

model = BiLSTM_CRF(src_vocab.size(), tgt_vocab.size(), config)
checkpoint = torch.load(config.restore, lambda storage, loc: storage)
model.load_state_dict(checkpoint['model'])
model.eval()

device, devices_id = misc_utils.set_cuda(config)
config.device = device
model.to(device)

ILLEGAL_REGEX = r"[^\u4e00-\u9fff0-9a-zA-Z]"
# testCats = ['cloth']
with io.open("testOut.txt", 'w+', encoding='utf-8') as fout:
    with io.open("randomDescs.txt", encoding='utf-8') as fin:
        srcList = []
        srcIdList = []
        srcLenList = []
        catList = []

        batch_size = 10
        for line in fin.readlines():
            arr = line.strip().split("\t")
            line = re.sub(ILLEGAL_REGEX, "", arr[0])
            cat = arr[1]
            chars = [c for c in line]
            ids = src_vocab.convertToIdx(chars, utils.UNK_WORD)
            ids = ids[::-1] # 由于训练时反转了，这里也需要反转，见data_helper的padding函数
            # print(chars, ids)

            srcList.append(line)
            srcIdList.append(ids)
            srcLenList.append(len(ids))
            catList.append(cat)
        # packed rnn sequence needs lengths to be in decreasing order
        indices = np.argsort(srcLenList)[::-1]
        srcList = [srcList[i] for i in indices]
        srcIdList = [srcIdList[i] for i in indices]
        srcLenList = [srcLenList[i] for i in indices]
        catList = [catList[i] for i in indices]

        # srcList = srcList[:2]
        resList = []
        addOne = 1 if (len(srcList) % batch_size) else 0
        for i in range(len(srcList) // batch_size + addOne):
            print("batch ", i)
            startIdx = i * batch_size
            endIdx = min((i + 1) * batch_size, len(srcList))
            xs = srcIdList[startIdx:endIdx]
            lengths = srcLenList[startIdx:endIdx]

            maxLen = max(len(x) for x in xs)
            xs = [x + [0] * (maxLen - len(x)) for x in xs]

            xs = torch.tensor(xs).to(device)
            lengths = torch.tensor(lengths).to(device)

            with torch.no_grad():
                score, tag_seq = model(xs, lengths)
                for tags in tag_seq:
                    candidates = ''.join(tgt_vocab.convertToLabels(tags, utils.EOS))
                    resList.append(candidates)

        for src, res in zip(srcList, resList):
            fout.write(src + '\n')
            for x, y in zip(src, res):
                fout.write(x +'\t'+ y +'\n')
            fout.write("\n")

        fout.write("\n####################################\n\n")