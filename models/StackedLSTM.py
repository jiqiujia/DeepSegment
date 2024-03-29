import torch
from torch import nn


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, config):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.residual = config.residual
        self.hidden_dim = config.hidden_dim

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, self.hidden_dim))
            input_size = self.hidden_dim

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            output = h_1_i
            if i + 1 != self.num_layers:
                output = self.dropout(output)

            if self.residual and input.size(-1) == output.size(-1):
                input = output + input
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)
