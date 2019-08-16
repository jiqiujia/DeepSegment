import torch
from torch import nn


class StackedLSTM(torch.jit.ScriptModule):
    def __init__(self, num_layers, input_size, config):
        super(StackedLSTM, self).__init__()
        self.device = str(config.device)
        self.dropout = nn.Dropout(config.dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.residual = config.residual
        self.hidden_dim = config.hidden_dim

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, self.hidden_dim))
            input_size = self.hidden_dim

        self.device = torch.jit.Attribute(self.device, str)
        self.hidden_dim = torch.jit.Attribute(self.hidden_dim, int)
        self.num_layers = torch.jit.Attribute(self.num_layers, int)
        self.residual = torch.jit.Attribute(self.num_layers, bool)

    @torch.jit.script_method
    def forward(self, input):
        batch_size = input.shape[0]
        h_0, c_0 = (torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                    torch.randn(self.num_layers, batch_size, self.hidden_dim).to(self.device))
        h_1, c_1 = [], []
        i = 0
        for layer in self.layers:
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            output = h_1_i
            if i + 1 != self.num_layers:
                output = self.dropout(output)

            if self.residual and input.size(-1) == output.size(-1):
                input = output + input
            h_1 += [h_1_i]
            c_1 += [c_1_i]

            i += 1

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)
