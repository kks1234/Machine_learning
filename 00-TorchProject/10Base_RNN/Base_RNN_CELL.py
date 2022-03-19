import torch
import torch.nn as nn

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

cell = nn.RNNCell(input_size, hidden_size)

dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, inp in enumerate(dataset):
    print("=" * 20, idx, "=" * 20)
    print("Input Size:", inp.shape)
    hidden = cell(inp, hidden)
    print("Outputs Size:", hidden.shape)
    print(hidden)
