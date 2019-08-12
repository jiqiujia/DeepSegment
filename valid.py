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

device, devices_id = misc_utils.set_cuda(config)
config.device = device

src_vocab = utils.Dict()
src_vocab.loadFile(os.path.join(config.data, "src.vocab"))
tgt_vocab = utils.Dict()
tgt_vocab.loadFile(os.path.join(config.data, "tgt.vocab"))

model = BiLSTM_CRF(src_vocab.size(), tgt_vocab.size(), config)
checkpoint = torch.load(config.restore, lambda storage, loc: storage)
print(model.state_dict().keys())
print(checkpoint['model'].keys())
model.load_state_dict(checkpoint['model'])
model.eval()

model.to(device)

# testCats = ['cloth']
with io.open("validOut2.txt", 'w+', encoding='utf-8') as fout:
    with io.open(os.path.join(config.data, 'valid.txt'), encoding='utf-8') as fin:
        srcList = []
        srcIdList = []
        srcLenList = []
        tgtList = []

        batch_size = config.batch_size
        cnt = 0
        tmpx = []
        tmpy = []
        tmpId = []
        for line in fin.readlines():
            if line.strip() == '':
                if len(tmpx) > 0:
                    tmpSrc = ''.join(src_vocab.convertToLabels(tmpx, None))
                    srcList.append(tmpSrc)
                    srcIdList.append(tmpx)
                    tgtList.append(tmpy)
                    srcLenList.append(len(tmpx))
                    tmpx = []
                    tmpy = []
            else:
                arr = line.strip().split("\t")
                x = int(arr[0])
                y = int(arr[1])
                tmpx.append(x)
                tmpy.append(y)
                cnt += 1
        if len(tmpx) > 0:
            tmpSrc = src_vocab.convertToLabels(tmpx, None)
            srcList.append(tmpSrc)
            srcIdList.append(tmpx)
            tgtList.append(tmpy)
            srcLenList.append(len(tmpx))

        # packed rnn sequence needs lengths to be in decreasing order
        indices = np.argsort(srcLenList)[::-1][-1000:]
        srcList = [srcList[i] for i in indices]
        srcIdList = [srcIdList[i] for i in indices]
        srcLenList = [srcLenList[i] for i in indices]
        tgtList = [tgtList[i] for i in indices]

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
                score, tag_seq = model(xs, lengths)
                for tags in tag_seq:
                    candidates = ''.join(tgt_vocab.convertToLabels(tags, utils.EOS))
                    resList.append(candidates)

        for src, res, gt in zip(srcList, resList, tgtList):
            fout.write(src + '\n')
            idx = 0
            stses = []
            for x, y in zip(src, gt):
                if y == 1 and idx > 0:
                    stses.append('，')
                stses.append(x)
                idx += 1
            fout.write(''.join(stses) + '\n')

            tmp = []
            idx = 0
            for x, y in zip(src, res):
                if y == 'b' and idx > 0:
                    fout.write(''.join(tmp) + '\n')
                    tmp = []
                tmp.append(x)
                idx += 1
            fout.write(''.join(tmp) + "\n\n")

        fout.write("\n####################################\n\n")
