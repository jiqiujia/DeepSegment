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
pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in state}
state.update(pretrained_dict)
model.load_state_dict(state)
model.eval()

batch_size = 1
length = 2
x = torch.ones(batch_size, length).long()
lengths = torch.ones(batch_size).long() * 2
output = model(x, lengths)
print('output', output)

torch.onnx.export(model,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "deepsegment.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the onnx version to export the model to
                  do_constant_folding=True,  # wether to execute constant folding for optimization
                  input_names=['input', 'lengths'],  # the model's input names
                  verbose=True,
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size', 1: 'lengths'},  # variable lenght axes
                                'lengths': {0: 'batch_size'},
                                'output': {0: 'batch_size'}},
                  example_outputs=output)
