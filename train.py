import os
import sys
import time

from torchtext import data, datasets
import opts
import yaml
from argparse import Namespace
from utils import init_logger, misc_utils, lr_scheduler
import utils

import torch
from model import BiLSTM_CRF

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

if __name__ == '__main__':
    # Combine command-line arguments and yaml file arguments
    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))
    logger = init_logger("torch", logging_path='')

    # writer = misc_utils.set_tensorboard(config)
    device, devices_id = misc_utils.set_cuda(config)

    TEXT = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                      include_lengths=True, pad_token=utils.PAD, preprocessing=to_int,
                      init_token=utils.BOS, eos_token=utils.EOS)
    LABEL = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                       include_lengths=True, pad_token=utils.PAD, preprocessing=to_int,
                       init_token=utils.BOS, eos_token=utils.EOS)

    fields = [("text", TEXT), ("label", LABEL)]
    trainDataset = datasets.SequenceTaggingDataset(path=os.path.join(config.data, 'valid.txt'),
                                                   fields=fields)
    validDataset = datasets.SequenceTaggingDataset(path=os.path.join(config.data, 'valid.txt'),
                                                   fields=fields)
    train_iter, valid_iter = data.Iterator.splits((trainDataset, validDataset),
                                                  batch_sizes=(config.batch_size, config.batch_size))

    src_vocab = utils.Dict()
    src_vocab.loadFile(os.path.join(config.data, "src.vocab"))
    tgt_vocab = utils.Dict()
    tgt_vocab.loadFile(os.path.join(config.data, "tgt.vocab"))

    model = BiLSTM_CRF(src_vocab.size(), tgt_vocab.size(), config)

    optim = utils.Optim(
        config.optim,
        config.learning_rate,
        config.max_grad_norm,
        lr_decay=config.learning_rate_decay,
        start_decay_steps=config.start_decay_steps,
        beta1=config.beta1,
        beta2=config.beta2,
        decay_method=config.decay_method,
        warmup_steps=config.warmup_steps,
        model_size=config.hidden_dim,
    )
    optim.set_parameters(model.parameters())

    if config.schedule:
        scheduler = lr_scheduler.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

    if config.restore:
        print("loading checkpoint...\n")
        checkpoints = torch.load(
            config.restore, map_location=lambda storage, loc: storage
        )
    else:
        checkpoints = None

    params = {
        "updates": 0,
        "report_total_loss": 0,
        "report_total": 0,
        "report_correct": 0,
        "report_time": time.time(),
        "log_path": os.path.join(config.logdir, config.expname) + "/",
    }
    for metric in config.metrics:
        params[metric] = []
    if config.restore:
        params["updates"] = checkpoints["updates"]

    logger.info('begin training...')
    for e in range(config.num_epoch):
        logger.info('epoch %d'.format(e))
        for batch in valid_iter:
            model.zero_grad()

            inputs = batch.text[0]
            labels = batch.label[0]
            lengths = batch.text[1]

            loss = model.neg_log_likelihood(inputs, labels, lengths)
            loss = torch.mean(loss)
            loss.backward()
            optim.step()

        with torch.no_grad():
            model.eval()
            for batch in valid_iter:
                model.zero_grad()

                inputs = batch.text[0]
                label = batch.label
                length = batch.text[1]

                score, tag_seq = model(inputs)
