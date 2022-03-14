import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input_data = torch.randn(batch_size, in_channels, width, height)

conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, bias=False)

out = conv_layer(input_data)

print(input_data.shape)
print(out.shape)
print(conv_layer.weight.shape)
