from argparse import ArgumentParser, Namespace
import io
import os
import yaml
import torch
import re
import numpy as np

import opts
import utils
from serving.lstm_crf import BiLSTM_CRF
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
# print(model.state_dict().keys())
# print(checkpoint['model'].keys())
state = model.state_dict()
state.update(checkpoint['model'])
model.load_state_dict(state)
model.eval()


batch_size = 1
length = 2
x = torch.ones(batch_size, length).long()
lengths = torch.ones(batch_size).long() * length

model.eval()

# TODO: failed because nn.utils.rnn.pack_padded_sequence is not supported
# see issue: https://github.com/pytorch/pytorch/issues/21282
traced_scripts_module = torch.jit.trace(model, (x, lengths), check_trace=False)
print(traced_scripts_module)

traced_scripts_module.save('traced_model.pt')

