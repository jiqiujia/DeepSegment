import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torchtext import data, datasets
import opts
import yaml
from argparse import Namespace
import misc_utils
from .utils import init_logger
import utils


def to_int(x):
    # 如果加载进来的是已经转成id的文本
    # 此处必须将字符串转换成整型
    return [int(c) for c in x]

if __name__ == '__main__':
    # Combine command-line arguments and yaml file arguments
    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))
    logger = init_logger("torch", logging_path='')

    writer = misc_utils.set_tensorboard(config)
    device, devices_id = misc_utils.set_cuda(config)

    TEXT = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                      include_lengths=True, pad_token=utils.PAD, preprocessing=to_int)
    LABEL = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                       include_lengths=True, pad_token=utils.PAD, preprocessing=to_int)

    fields = [("text", TEXT), ("label", LABEL)]
    trainDataset = datasets.SequenceTaggingDataset(path=os.path.join(config.data, 'train.txt'),
                                                   fields=fields)
    validDataset = datasets.SequenceTaggingDataset(path=os.path.join(config.data, 'valid.txt'),
                                                   fields=fields)
    train_iter, valid_iter = data.Iterator.splits((trainDataset, validDataset),
                                                  batch_sizes=(config.batch_size, config.batch_size))

    # valid_iter = data.Iterator(validDataset, batch_size=config.batch_size, device=device)

    logger.info('begin training...')
    for e in range(config.num_epoch):
        logger.info('epoch %d'.format(e))
        for batch in valid_iter:
            inputs = batch.text[0]
            label = batch.label
            length = batch.text[1]


            print(inputs, label, length)
