import torch

input_data = [
    3, 4, 6, 5,
    2, 4, 6, 8,
    1, 6, 7, 8,
    9, 7, 4, 6,
]

input_ = torch.Tensor(input_data).view(1, 1, 4, 4)

Max_polling_layer = torch.nn.MaxPool2d(kernel_size=2)

out = Max_polling_layer(input_)
print(out)
