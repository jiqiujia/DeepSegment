import os
import sys
import time
from tqdm import tqdm

from torchtext import data, datasets
import opts
import yaml
from argparse import Namespace
from utils import init_logger, misc_utils, lr_scheduler
import utils

import torch
from torch.nn.init import xavier_uniform_
from model import BiLSTM_CRF
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

def eval(valid_iter, model, config, best_loss, tgt_vocab):
    with torch.no_grad():
        model.eval()
        report_num_correct = 0
        report_num_total = 0
        report_loss_total = 0
        num_updates = 0

        precision = 0.
        total_precision = 0
        recall = 0.
        total_recall = 0
        for batch in tqdm(valid_iter):
            model.zero_grad()

            inputs = batch.text[0].to(device)
            labels = batch.label[0].to(device)
            lengths = batch.text[1].to(device)

            score, tag_seq = model(inputs, lengths)
            all_tags = np.asarray([tag for tags in tag_seq for tag in tags])
            all_labels = torch.masked_select(labels, labels.ne(utils.PAD)).cpu().numpy()
            num_correct = np.sum(all_tags == all_labels)
            num_total = labels.ne(utils.PAD).sum().item()
            score = torch.sum(score).item() / num_total
            report_num_correct += num_correct
            report_num_total += num_total
            report_loss_total += score
            num_updates += 1

            all_b_tags = all_tags == tgt_vocab.lookup('B')
            all_b_labels = all_labels == tgt_vocab.lookup('B')
            intersection = np.sum(np.logical_and(all_b_labels, all_b_tags))
            precision += intersection
            total_precision += np.sum(all_b_tags)
            recall += intersection
            total_recall += np.sum(all_b_labels)
            # print(inputs[0])
            # print(labels[0])
            # print(tag_seq[0])
            # break
        cur_loss = report_loss_total / num_updates
        cur_acc = report_num_correct / report_num_total
        precision = precision / total_precision
        recall = recall / total_recall
        logger.info("valid loss {}, acc {}, precision {}, recall {}".format(cur_loss, cur_acc, precision, recall))
        if config.mode == 'train':
            writer.add_scalar("valid/loss", cur_loss, params["updates"])
            writer.add_scalar("valid/acc", cur_acc, params["updates"])
            writer.add_scalar("valid/precision", precision, params["updates"])
            writer.add_scalar("valid/recall", recall, params["updates"])
            save_model(
                params["log_path"] + "checkpoint.pt",
                model,
                optim,
                params["updates"],
                config,
            )
            if cur_loss < best_loss:
                best_loss = cur_loss
                save_model(
                    params["log_path"] + "best_checkpoint.pt",
                    model,
                    optim,
                    params["updates"],
                    config,
                )
            return best_loss


if __name__ == '__main__':
    # Combine command-line arguments and yaml file arguments
    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))
    logger = init_logger("torch", logging_path='')
    logger.info(config.__dict__)

    writer = misc_utils.set_tensorboard(config)
    device, devices_id = misc_utils.set_cuda(config)
    config.device = device

    TEXT = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                      include_lengths=True, pad_token=utils.PAD, preprocessing=to_int,)
                      # init_token=utils.BOS, eos_token=utils.EOS)
    LABEL = data.Field(sequential=True, use_vocab=False, batch_first=True, unk_token=utils.UNK,
                       include_lengths=True, pad_token=utils.PAD, preprocessing=to_int,)
                       # init_token=utils.BOS, eos_token=utils.EOS)

    fields = [("text", TEXT), ("label", LABEL)]
    trainDataset = datasets.SequenceTaggingDataset(path=os.path.join(config.data, 'train.txt'),
                                                   fields=fields)
    validDataset = datasets.SequenceTaggingDataset(path=os.path.join(config.data, 'valid.txt'),
                                                   fields=fields)
    train_iter, valid_iter = data.BucketIterator.splits((trainDataset, validDataset),
                                                        batch_sizes=(config.batch_size, config.batch_size),
                                                        sort_key=lambda x: len(x.text),  # field sorted by len
                                                        sort_within_batch=True
                                                        )

    src_vocab = utils.Dict()
    src_vocab.loadFile(os.path.join(config.data, "src.vocab"))
    tgt_vocab = utils.Dict()
    tgt_vocab.loadFile(os.path.join(config.data, "tgt.vocab"))

    model = BiLSTM_CRF(src_vocab.size(), tgt_vocab.size(), config)
    model.to(device)

    if config.param_init != 0.0:
        for p in model.parameters():
            p.data.uniform_(-config.param_init, config.param_init)
    if config.param_init_glorot:
        for p in model.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

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
    if checkpoints is not None:
        model.load_state_dict(checkpoints["model"])
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

    if config.mode == 'train':
        logger.info('begin training...')
        best_loss = 1000000000
        for e in range(config.num_epoch):
            logger.info('epoch {}'.format(e))
            if config.schedule:
                scheduler.step()
            model.train()
            for batch in tqdm(train_iter):
                model.zero_grad()

                inputs = batch.text[0].to(device)
                labels = batch.label[0].to(device)
                lengths = batch.text[1].to(device)

                loss = model.neg_log_likelihood(inputs, labels, lengths)
                num_total = labels.ne(utils.PAD).sum().item()
                loss = torch.sum(loss) / num_total
                loss.backward()
                optim.step()

                params["report_total_loss"] += loss.item()
                params["report_total"] += num_total
                params["updates"] += 1

                if params["updates"] % config.report_interval == 0:
                    writer.add_scalar("train/{}", loss.item(), params["updates"])
                    logger.info("{} loss {}".format(params["updates"], loss.item()))
                    writer.add_scalar("train" + "/lr", optim.lr, params['updates'])
                # if params["updates"] % 1 == 0:
                #     break

            if config.epoch_decay:
                optim.updateLearningRate(e)

            best_loss = eval(valid_iter, model, config, best_loss, tgt_vocab)
    elif config.mode == 'eval':
        eval(valid_iter, model, config, 0, tgt_vocab)
