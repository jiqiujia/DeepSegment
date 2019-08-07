import torch
import torch.nn as nn

print(torch.__version__)

# TODO: https://github.com/pytorch/pytorch/issues/23930
class test(torch.jit.ScriptModule):

    def __init__(self, rnn_dims=512):
        super().__init__()
        self.rnn = nn.LSTM(input_size=rnn_dims, hidden_size=rnn_dims, batch_first=True)
        # delattr(self.rnn, 'forward_packed')

    @torch.jit.script_method
    def forward(self, x):
        # h1 = (torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
        out, h1 = self.rnn(x, torch.zeros(1, 1, 512))

        return h1


net = test()

output = net(torch.randn(1, 1, 512))

print(output.shape)