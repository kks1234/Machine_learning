import torch

xx = [1.0, 2.0, 3.0]
yy = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0, 2.0])
b = torch.Tensor([1.0])
w.requires_grad = True
b.requires_grad = True


def forward(_x):
    return w[0] * _x * _x + w[1] * _x + b


def loss(_x, _y):
    y_pred = forward(_x)
    return (y_pred - _y) ** 2


if __name__ == '__main__':
    print("predict (before training) ", 4, forward(4).item())
    for epoch in range(100):
        for x, y in zip(xx, yy):
            tloss = loss(x, y)
            tloss.backward()
            w.data = w.data - 0.01 * w.grad.data
            b.data = b.data - 0.01 * b.grad.data
            print("\tgrad:", x, y, w.grad.data[0].item(), w.grad.data[1].item(), b.grad.item())
            w.grad.data.zero_()
            b.grad.data.zero_()
        print("Process:", epoch, tloss.item())
    print("predict (after training) ", 4, forward(4).item())
