import torch
import torch.nn as nn

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers  = 5

cell = nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers,batch_size, hidden_size)

out,hidden = cell(dataset,hidden)

print("Output Size:",out.shape)
print("Output:",out)
print("Hidden Size:",hidden.shape)
print("Hidden:",hidden)
