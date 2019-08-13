import torch
import torch.nn as nn


# TODO: https://github.com/pytorch/pytorch/issues/23930
class test(torch.jit.ScriptModule):

    def __init__(self, vocab_size=10, rnn_dims=512):
        super().__init__()
        self.word_embeds = nn.Embedding(vocab_size, rnn_dims)
        self.emb_drop = nn.Dropout(0.1)
        self.rnn = nn.LSTM(input_size=rnn_dims, hidden_size=rnn_dims, batch_first=True,
                           num_layers=2, dropout=0.1)
        # delattr(self.rnn, 'forward_packed')

    @torch.jit.script_method
    def forward(self, x):
        h1 = (torch.zeros(2, 1, 512), torch.zeros(2, 1, 512))
        embeds = self.emb_drop(self.word_embeds(x))
        out, h1 = self.rnn(embeds, h1)

        return h1


model = test()

input = torch.ones((1,3)).long()
output = model(input)
print('output', output)

torch.onnx.export(model,  # model being run
                  input,
                  'test.onnx',
                  example_outputs=output)
# torch.jit.trace(net, (torch.ones((1,3)).long(), torch.ones((3,1))), check_trace=False)
# print(output.shape)