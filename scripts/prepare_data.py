# -*- coding: UTF-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import io
import re
import random
from argparse import ArgumentParser
import utils

parser = ArgumentParser(description="train.py")
parser.add_argument('--input', type=str, help='input file path')
parser.add_argument('--outdir', type=str, help='output dir')
parser.add_argument('--min_freq', type=int, default=0, help='vocab frequency threshold')
parser.add_argument('--valid_size', type=float, default=0.1, help='validation set size')
opt = parser.parse_args()

logger = utils.init_logger("torch", logging_path='')

STS_SEPARATOR_REGEX = '[,，！!。？?:：;；~]|[-.=]{2,}'
punctuations = ',，！!。？?;；'
special_punctuations = '[:：~]'

sts_multi_separator_regex = re.compile('[-.。=,，！!？?:：;；~…]{2,}')
alpha_digit_regex = re.compile('[0-9a-zA-Z]')

def split_to_stses(line):
    stses = sts_multi_separator_regex.split(line)
    res = []
    for sts in stses:
        start = -1
        for i in range(len(sts)):
            if sts[i] in punctuations:
                res.append(sts[start+1:i])
                start = i
            elif sts[i] in special_punctuations:
                if i < len(sts) - 1 and i > 0 and not (alpha_digit_regex.match(sts[i-1]) and alpha_digit_regex.match(sts[i+1])):
                    res.append(sts[start+1:i])
                    start = i
                elif i==0:
                    start = i
                elif i==len(sts) - 1:
                    res.append(sts[start+1:i])
                    start = i
        if start < len(sts) - 1:
            res.append(sts[start+1:])
    return res


def train_val_split(X, y, valid_size=0.1, random_state=1101, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
    logger.info('train val split')
    data = [(data_x, data_y) for data_x, data_y in zip(X, y)]
    N = len(data)
    test_size = int(N * valid_size) if valid_size < 1 else valid_size

    if shuffle:
        random.seed(random_state)
        random.shuffle(data)

    valid = data[:test_size]
    train = data[test_size:]
    return train, valid


if __name__ == '__main__':
    srcList = []
    tgtList = []
    with io.open(opt.input, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip().lower()
            if line.count(':') >= 3 or line.count(' ') > 3:
                continue
            if '【' in line or '】' in line:
                continue
            if "mb" in opt.input:
                line = line.split('\t')[0]
            else:
                line = line.split('\t')[-1]
            stses = split_to_stses(line)
            print(stses)
            tmp = ''.join(stses)
            if len(tmp) < 30:
                continue
            tags = [['I'] * len(sts) for sts in stses if len(sts) > 0]
            for tag in tags:
                if len(tag) > 0:
                    tag[0] = 'B'
                    tag[-1] = 'E'
            src = ''.join(stses)
            tgt = ''.join([''.join(tag) for tag in tags])
            assert len(src) == len(tgt), 'unequal length of src and tgt'
            srcList.append(src)
            tgtList.append(tgt)

    # srcList = srcList[:100]
    # tgtList = tgtList[:100]
    # dump raw files
    # with io.open(opt.outdir, 'w+', encoding='utf-8') as fout:
    #     for src, tgt in zip(srcList, tgtList):
    #         for ch, label in zip(src, tgt):
    #             fout.write(ch +'\t'+ label +'\n')
    #         fout.write('\n')

    print('building dictionary...')
    srcDict = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD], lower=True)
    for stses in srcList:
        for ch in stses:
            srcDict.add(ch)
    srcDict = srcDict.prune(srcDict.size(), opt.min_freq)
    srcDict.writeFile(os.path.join(opt.outdir, 'src.vocab'))

    tgtDict = utils.Dict([utils.PAD_WORD], lower=False)
    tgtDict.add('B')
    tgtDict.add('I')
    tgtDict.add('E')
    tgtDict.writeFile(os.path.join(opt.outdir, 'tgt.vocab'))

    logger.info("convert to idx...")
    srcIdList = []
    tgtIdList = []
    for src, tgt in zip(srcList, tgtList):
        srcIds = srcDict.convertToIdx(src, unkWord=utils.UNK_WORD)
        tgtIds = tgtDict.convertToIdx(tgt, unkWord=utils.UNK_WORD)
        srcIdList.append(srcIds)
        tgtIdList.append(tgtIds)

    trainData, validData = train_val_split(srcIdList, tgtIdList, valid_size=opt.valid_size)

    print('writing files...')
    with io.open(os.path.join(opt.outdir, 'train.txt'), 'w+', encoding='utf-8') as fout:
        for x, y in trainData:
            for sc, yc in zip(x, y):
                fout.write(str(sc) + '\t' + str(yc) + '\n')
            fout.write("\n")

    with io.open(os.path.join(opt.outdir, 'valid.txt'), 'w+', encoding='utf-8') as fout:
        for x, y in validData:
            for sc, yc in zip(x, y):
                fout.write(str(sc) + '\t' + str(yc) + '\n')
            fout.write("\n")
