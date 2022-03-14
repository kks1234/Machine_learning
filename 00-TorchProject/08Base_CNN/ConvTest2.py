import torch

input_data = [
    3, 4, 6, 5, 7,
    2, 4, 6, 8, 2,
    1, 6, 7, 8, 4,
    9, 7, 4, 6, 2,
    3, 7, 5, 4, 1,
]
kernel_set = [
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
]

input_ = torch.Tensor(input_data).view(1, 1, 5, 5)
kernel_ = torch.Tensor(kernel_set).view(1, 1, 3, 3)

conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0, bias=False)
conv_layer.weight.data = kernel_.data

out = conv_layer(input_)
print(out)
