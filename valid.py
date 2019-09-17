import os
import sys
import time
import io
from tqdm import tqdm

from torchtext import data, datasets
import opts
import yaml
from argparse import Namespace
from utils import init_logger, misc_utils, lr_scheduler
import utils

import torch
from torch.nn.init import xavier_uniform_
from models import BiLSTM_CRF, ResLSTM_CRF, TransformerCRF
import numpy as np


def to_int(x):
    # 如果加载进来的是已经转成id的文本
    # 此处必须将字符串转换成整型
    return [int(c) for c in x]


# save model
def save_model(path, model, optim, updates, config):
    model_state_dict = model.state_dict()
    optim_state_dict = optim.optimizer.state_dict()
    checkpoints = {
        "model": model_state_dict,
        "config": config,
        "updates": updates,
        "optim": optim_state_dict,
    }
    torch.save(checkpoints, path)


def getPR(pred, gt, label):
    all_b_tags = pred == tgt_vocab.lookup('B')
    all_b_labels = gt == tgt_vocab.lookup('B')
    intersection = np.sum(np.logical_and(all_b_labels, all_b_tags))
    return intersection, np.sum(all_b_tags), np.sum(all_b_labels)

def get_stses(x, y):
    res = []
    i = 0
    for xx, yy in zip(x,y):
        if yy == 'b' and i>0:
            res.append('，')
        res.append(xx)
        i += 1
    return ''.join(res)


if __name__ == '__main__':
    # Combine command-line arguments and yaml file arguments
    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))
    logger = init_logger("torch", logging_path='')
    logger.info(config.__dict__)

    device, devices_id = misc_utils.set_cuda(config)
    config.device = device

    TEXT = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                      include_lengths=True, pad_token=utils.PAD, preprocessing=to_int, )
    # init_token=utils.BOS, eos_token=utils.EOS)
    LABEL = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                       include_lengths=True, pad_token=utils.PAD, preprocessing=to_int, )
    # init_token=utils.BOS, eos_token=utils.EOS)

    fields = [("text", TEXT), ("label", LABEL)]
    validDataset = datasets.SequenceTaggingDataset(path=os.path.join(config.data, 'valid.txt'),
                                                   fields=fields)
    valid_iter = data.Iterator(validDataset,
                               batch_size=config.batch_size,
                               sort_key=lambda x: len(x.text),  # field sorted by len
                               sort=True,
                               sort_within_batch=True,
                               repeat=False
                               )

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
    model.to(device)

    if config.restore:
        print("loading checkpoint...\n")
        checkpoints = torch.load(
            config.restore, map_location=lambda storage, loc: storage
        )
    else:
        checkpoints = None

    if checkpoints is not None:
        model.load_state_dict(checkpoints["model"])

    print(repr(model) + "\n\n")
    model.eval()

    oovs = {0: 'B', 1: 'B'}
    oriList = []
    resList = []
    scoreList = []
    for batch in tqdm(valid_iter):
        inputs = batch.text[0].to(device)
        labels = batch.label[0].to(device)
        lengths = batch.text[1].to(device)

        with torch.no_grad():
            score, tag_seq = model(inputs, lengths, config.nbest, None)
            for s in score:
                scoreList.append(s.item())
            for input, label, tags in zip(inputs, labels, tag_seq):
                x = src_vocab.convertToLabels(input.numpy(), utils.PAD)
                y = tgt_vocab.convertToLabels(label.numpy(), utils.PAD)
                candidates = ''.join(tgt_vocab.convertToLabels(tags, utils.PAD, oovs=oovs))
                oriList.append(get_stses(x, y))
                resList.append(get_stses(x, [t for t in candidates]))
        # break

    with io.open('validOut.txt', 'w+', encoding='utf-8') as fout:
        for ori, res, score in zip(oriList, resList, scoreList):
            fout.write(ori + '\n')
            fout.write(res + '\t' + str(score) + '\n')
            fout.write('\n')
